from utils import *


class AdvRunner:
    def __init__(self, model, criterion, optimizer, device, alpha=1e-3, suffix_len=20, suffix_char='!'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.alpha = alpha
        self.suffix_len = suffix_len

        self.embeddings_matrix = self.model.embeddings_matrix
        self.PP_T = self.model.PP_T.to(self.device)
        self.tokenizer = self.model.tokenizer
        self.model.eval()

        self.suffix = suffix_char * suffix_len

    # TODO: check if this is the correct projection (for now we ignore it)
    def project(self, inputs_embeds):
        inputs_embeds[:, -(self.suffix_len+1):-1] = torch.einsum(
            'ij,bsj->bsi', self.PP_T, inputs_embeds[:, -(self.suffix_len+1):-1])
        return inputs_embeds

    def decode(self, input_ids, skip_special_tokens=True):
        return self.tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens)

    def get_pred_and_loss(self, input_ids=None, attention_mask=None, inputs_embeds=None, label=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        pred = outputs.item()
        loss = self.criterion(outputs, label.unsqueeze(0).unsqueeze(0))
        return pred, loss
    
    def FGSM_step(self, inputs, verbose=False):
        # Decode original input text and append the suffix
        original_text = self.decode(inputs['input_ids'][0]) + self.suffix

        # Tokenize perturbed text
        tokenized = self.tokenizer(original_text, return_tensors='pt').to(self.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        label = inputs['label'].float().to(self.device)

        # Get original text loss
        original_text_pred, original_text_loss = self.get_pred_and_loss(input_ids=input_ids,
                                                                            attention_mask=attention_mask,
                                                                            label=label)
        # Get embeddings and enable gradient tracking
        inputs_embeds = self.embeddings_matrix[input_ids].clone().detach().to(self.device).requires_grad_(True)
        # Forward pass with embeddings
        original_emb_pred, original_emb_loss = self.get_pred_and_loss(inputs_embeds=inputs_embeds,
                                                                        attention_mask=attention_mask,
                                                                        label=label)
        original_emb_loss.backward() # Compute gradients

        if verbose:
            print(f'Original Text: {original_text}')
            print(f'Original Label: {label.item()}')
            print(f'Original Prediction: {original_text_pred}')
            print(f'Original Text Loss: {original_text_loss.item()}')
            print(f'Original Embedding Prediction: {original_emb_pred}')
            print(f'Original embedding loss: {original_emb_loss.item()}')

        # Extract gradients sign corresponding to suffix tokens only
        perturbation = inputs_embeds.grad[:, -(self.suffix_len):-1].sign()

        # Apply perturbation
        with torch.no_grad():
            inputs_embeds[:, -(self.suffix_len):-1] += self.alpha * perturbation
            # inputs_embeds = self.project(inputs_embeds)

        # Map perturbed embeddings back to discrete tokens (nearest embeddings by L2 distance)
        with torch.no_grad():
            distances = torch.cdist(inputs_embeds, self.embeddings_matrix)
            perturbed_input_ids = distances.argmin(dim=-1)

        # Decode new perturbed text
        perturbed_text = self.decode(perturbed_input_ids[0])

        # Get perturbed text loss
        perturbed_text_pred, perturbed_loss = self.get_pred_and_loss(input_ids=perturbed_input_ids,
                                                                        attention_mask=attention_mask,
                                                                        label=label)
        # Evaluate perturbed inputs_embeds on the model
        perturbed_emb_pred, perturbed_emb_loss = self.get_pred_and_loss(inputs_embeds=inputs_embeds,
                                                                        attention_mask=attention_mask,
                                                                        label=label)
        if verbose:
            print(f'Perturbed Text: {perturbed_text}')
            print(f'Perturbed Prediction: {perturbed_text_pred}')
            print(f'Perturbed text loss: {perturbed_loss.item()}')
            print(f'Perturbed Embedding Prediction: {perturbed_emb_pred}')
            print(f'Perturbed embedding loss: {perturbed_emb_loss.item()}')

        return original_text, original_text_pred, original_text_loss, original_emb_pred, original_emb_loss, \
                perturbed_text, perturbed_text_pred, perturbed_loss, perturbed_emb_pred, perturbed_emb_loss
    
    def PGD_step(self):
        pass


class TextAdvDataset(Dataset):
    # TODO: max_length is a problem maybe!
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data['comment_text'].iloc[idx]
        label = self.data['toxic'].iloc[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True)
        inputs['label'] = torch.tensor(label)
        return inputs


def calculate_perturbed_accuracy(dict_attack_results):
    correct = 0
    for key in dict_attack_results.keys():
        pred_label = dict_attack_results[key]['perturbed_text_pred'] > 0.5
        true_label = dict_attack_results[key]['true_label']
        if pred_label == true_label:
            correct += 1
    return correct / len(dict_attack_results)

def run_FGSM_attack(advrunner, adv_test_loader, verbose=False):
    print("Running FGSM attack...")
    dict_attack_results = {}
    for inputs in tqdm(adv_test_loader):
        single_input = {key: inputs[key][0] for key in inputs.keys()} # The format it works in (batch_size=1 here)
        original_text, original_text_pred, original_text_loss, original_emb_pred, original_emb_loss, \
                perturbed_text, perturbed_text_pred, perturbed_loss, perturbed_emb_pred, perturbed_emb_loss = \
                advrunner.FGSM_step(single_input, verbose=verbose)
        dict_attack_results[original_text] = {
            'original_text_pred': original_text_pred,
            'original_text_loss': original_text_loss.item(),
            'original_emb_pred': original_emb_pred,
            'original_emb_loss': original_emb_loss.item(),
            'perturbed_text': perturbed_text,
            'perturbed_text_pred': perturbed_text_pred,
            'perturbed_loss': perturbed_loss.item(),
            'perturbed_emb_pred': perturbed_emb_pred,
            'perturbed_emb_loss': perturbed_emb_loss.item(),
            'true_label': single_input['label'].item()
        }
    return dict_attack_results