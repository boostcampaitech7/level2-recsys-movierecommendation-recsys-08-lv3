import argparse
import torch
import os
import pandas as pd
from src.preprocessing import basic_data_load, context_data_split, context_data_side_merge
from src.utils import seed_everything, initialize_model, save_model
from src.train import train_deepfm, train_slim, train_ease  # 학습 함수 가져오기
from sklearn.metrics import precision_score, recall_score

def save_predictions(predictions, output_path="results/predictions/predictions.csv"):
    """
    Save predictions to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df = pd.DataFrame(predictions, columns=["user_id", "item_id", "predicted_score"])
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

def evaluate_model(model, valid_loader, device, top_k=10, output_path="results/predictions/predictions.csv"):
    """
    Evaluate the model and save predictions.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            user_ids, item_ids, genres, writers, directors, years, labels = [t.to(device) for t in batch]
            predictions = model(user_ids, item_ids, genres, writers, directors, years).squeeze()
            all_predictions.extend(zip(user_ids.cpu().numpy(), item_ids.cpu().numpy(), predictions.cpu().numpy()))
            all_labels.extend(labels.cpu().numpy())
    
    # Example evaluation metrics (Precision@K, Recall@K)
    precision = precision_score(all_labels, [1 if p > 0.5 else 0 for _, _, p in all_predictions])
    recall = recall_score(all_labels, [1 if p > 0.5 else 0 for _, _, p in all_predictions])
    print(f"Precision@{top_k}: {precision:.4f}, Recall@{top_k}: {recall:.4f}")

    # Save predictions to results/predictions/
    save_predictions(all_predictions, output_path=output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True, help="Mode to run: train or evaluate")
    args = parser.parse_args()

    # Load configuration and set seed
    import yaml

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    seed_everything(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    data = basic_data_load(config['data'])
    data = context_data_split(config['data'], data)
    data = context_data_side_merge(config['data'], data)

    # Initialize model
    model_name = config['model']['name']
    model = initialize_model(model_name, config['model'], data, device)

    if args.mode == "train":
        # Train model
        if model_name == "DeepFM":
            trained_model = train_deepfm(model, data['train_loader'], data['valid_loader'], config['train'], device)
        elif model_name == "SLIM":
            trained_model = train_slim(model, data)
        elif model_name == "EASE":
            trained_model = train_ease(model, data)

        # Save model
        save_model(trained_model, f"results/models/{model_name}_weights.pth")
        print(f"Model saved at results/models/{model_name}_weights.pth")

    elif args.mode == "evaluate":
        # Load model weights
        model.load_state_dict(torch.load(f"results/models/{model_name}_weights.pth"))
        model.to(device)

        # Evaluate model
        evaluate_model(model, data['valid_loader'], device, output_path=f"results/predictions/{model_name}_predictions.csv")