from utils import *

class BinaryTextClassifier(nn.Module):
    def __init__(self, transformer_model, device, freeze_transformer=True, normalize_PP_T=True):
        super(BinaryTextClassifier, self).__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(transformer_model, weights_only=True).to(self.device)
        # Freeze the transformer model
        if freeze_transformer:
            for param in self.model.parameters():
                param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.fc = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.set_PP_T(normalize=normalize_PP_T)

    def compute_projection_matrix(self, X, k=100):
        """
        Compute the projection matrix P that projects any vector onto the space spanned by the rows of X.
        
        Args:
        X: Tensor of shape (N, D) representing N samples in D-dimensional space.
        
        Returns:
        P: Projection matrix of shape (N, N).
        """
        U, S, Vh = torch.linalg.svd(X)
        # take the top k eigenvectors of Vh
        Vk = Vh[:k,:]
        # Compute the projection matrix onto the row space:
        P_row = Vk.T @ Vk
        return P_row
    
    def set_PP_T(self, normalize=True):
        self.embeddings_matrix = self.get_static_embeddings_matrix()
        self.PP_T = self.compute_projection_matrix(self.embeddings_matrix)

    def get_static_embeddings_matrix(self):
        return self.model.get_input_embeddings().weight.to(self.device)
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        cls_token = outputs.last_hidden_state[:, 0]
        cls_token = self.fc(cls_token)
        return self.sigmoid(cls_token)
    
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data['comment_text'].iloc[idx]
        label = self.data['toxic'].iloc[idx]
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)
        inputs['label'] = torch.tensor(label)
        return inputs
    

def train_classifier(model, train_loader, optimizer, criterion, device, epochs=5, model_path='None'):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("Training the model...")
        model.train()
        for epoch in range(5):
            model.train()
            epoch_loss = 0
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(device)
                input_ids = input_ids.squeeze(1)
                attention_mask = batch['attention_mask'].to(device)
                attention_mask = attention_mask.view(input_ids.shape)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'Epoch {epoch+1}/{5}, Loss: {epoch_loss/len(train_loader)}')
        print('Finished Training')
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

def evaluate_classifier(model, test_loader, device):
    print("Evaluating the model...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            input_ids = input_ids.squeeze(1)
            attention_mask = batch['attention_mask'].to(device)
            attention_mask = attention_mask.view(input_ids.shape)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask)
            predicted = torch.round(outputs.squeeze())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Accuracy: {accuracy}')
    return accuracy