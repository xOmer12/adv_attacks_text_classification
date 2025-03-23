from utils import *
from classifier import *
from advrunner import *

import argparse

# import os
# # set which GPU to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    parser = argparse.ArgumentParser(description='Train a binary text classifier')
    parser.add_argument('--task', type=str, help='Path to the training data', default='toxic_classification')
    parser.add_argument('--train_path', type=str, help='Path to the training data', default='data/train.csv')
    parser.add_argument('--test_path', type=str, help='Path to the test data', default='data/test.csv')
    parser.add_argument('--test_label_path', type=str, help='Name of the model to train', default='data/test_labels.csv')
    parser.add_argument('--test_frac', type=float, help='Fraction of the test data to use', default=1)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--verbose', type=bool, help='Verbosity', default=False)
    
    # Classifier Parameters
    parser.add_argument('--model_name', type=str, help='Name of the model to train', default='bert-base-uncased')
    parser.add_argument('--train_batch_size', type=int, help='Training batch size', default=32)
    parser.add_argument('--train_epochs', type=int, help='Number of training epochs for the classifier', default=5)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the classifier', default=0.001)
    parser.add_argument('--device', type=str, help='Device to train the classifier on', default='cuda:0')
    parser.add_argument('--model_dir', type=str, help='Directory to save the trained model', default='models')
    parser.add_argument('--max_token_length', type=int, help='Maximum token length for the tokenizer', default=512) # 512 for bert-base-uncased

    # Attack Parameters
    parser.add_argument('--attack', type=str, help='Type of attack to run', default='FGSM') # TODO: Add more attacks
    parser.add_argument('--alpha', type=float, help='Step size for the attack', default=0.5)
    parser.add_argument('--PGD_iterations', type=int, help='Number of iterations for the PGD attack', default=10)
    parser.add_argument('--return_iter_results', type=bool, help='Return the results for each iteration of the PGD attack', default=False)
    parser.add_argument('--suffix_len', type=int, help='Length of the suffix to add to the text', default=20)
    parser.add_argument('--suffix_char', type=str, help='Character to use for the suffix', default=' !')
    parser.add_argument('--results_dir', type=str, help='Directory to save the results', default='results')
    parser.add_argument('--text_proj_freq', type=int, help='project using embedding matrix in a given frequency', default=1)

    args = parser.parse_args()
    max_length = args.max_token_length - args.suffix_len
    device = torch.device(args.device)

    # set seed manually for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load the data
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    test_labels_df = pd.read_csv(args.test_label_path)

    train_df = train_df[['id', 'comment_text', 'toxic']]
    negative_sample_train = train_df[train_df['toxic'] == 0].sample(frac=0.1)
    positive_sample_train = train_df[train_df['toxic'] == 1]
    train_df = pd.concat([negative_sample_train, positive_sample_train])
    test_labels_df = test_labels_df[['id', 'toxic']]

    test_df = pd.merge(test_df, test_labels_df, on='id', how='inner')
    test_df = test_df[test_df['toxic'] != -1]
    # take a tenth of the test data
    test_df = test_df.sample(frac=args.test_frac)

    # Load the model and tokenizer
    model = BinaryTextClassifier('bert-base-uncased', device).to(device)
    train_dataset = TextDataset(train_df, model.tokenizer, max_length=max_length)
    test_dataset = TextDataset(test_df, model.tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    classifier_name = f"{args.model_name}_{args.task}_classifier.pth"
    # Create the model directory if it doesn't exist
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_path = os.path.join(args.model_dir, classifier_name)

    train_classifier(model, train_loader, optimizer, criterion, device, epochs=args.train_epochs, model_path=model_path)
    # clean_accuracy = evaluate_classifier(model, test_loader, device)
    clean_accuracy = 1 # TODO: remove

    # Attack
    adv_test_dataset = TextAdvDataset(test_df, model.tokenizer, max_length=max_length)
    adv_test_loader = DataLoader(adv_test_dataset, batch_size=1, shuffle=False)

    advrunner = AdvRunner(model, criterion, optimizer, device, alpha=args.alpha, suffix_len=args.suffix_len, suffix_char=args.suffix_char)
    # Run the attack
    if args.attack == 'FGSM':
        dict_attack_results = run_FGSM_attack(advrunner, adv_test_loader, verbose=args.verbose)
        perturbed_accuracy = calculate_perturbed_accuracy(dict_attack_results)
        print(f"Clean Accuracy: {clean_accuracy}")
        print(f"Perturbed Accuracy: {perturbed_accuracy}")

        # save the results
        results_dir = args.results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_name = f"{args.model_name}_{args.task}_{args.attack}_{args.alpha}_results.pkl"
        results_path = os.path.join(results_dir, results_name)
        with open(results_path, 'wb') as f:
            pickle.dump(dict_attack_results, f)
        print(f"Results saved to {results_path}")
    
    elif args.attack == 'PGD':
        dict_attack_results = run_PGD_attack(advrunner, adv_test_loader, verbose=args.verbose, num_iter=args.PGD_iterations, text_proj_freq=args.text_proj_freq, return_iter_results=args.return_iter_results)
        perturbed_accuracy = calculate_perturbed_accuracy(dict_attack_results)
        print(f"Clean Accuracy: {clean_accuracy}")
        print(f"Perturbed Accuracy: {perturbed_accuracy}")

        # save the results
        results_dir = args.results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_name = f"{args.model_name}_{args.task}_{args.attack}_{args.alpha}_results.pkl"
        results_path = os.path.join(results_dir, results_name)
        with open(results_path, 'wb') as f:
            pickle.dump(dict_attack_results, f)
        print(f"Results saved to {results_path}")
        
    else:
        print(f"Attack {args.attack} not implemented yet")


if __name__ == '__main__':
    main()



