import argparse
import os
import torch
import logging
import yaml
from src.preprocessing import basic_data_load, context_data_split, context_data_side_merge
from src.utils import seed_everything, initialize_model, save_model
from src.train import train_deepfm, train_slim, train_ease  
from src.evaluate import evaluate_model 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True, help="Mode to run: train or evaluate")
    args = parser.parse_args()

    # Load configuration and set seed
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    seed_everything(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up logging
    os.makedirs("results/logs", exist_ok=True)
    logging.basicConfig(
        filename="results/logs/main.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Script started.")

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
        logging.info(f"Model saved at results/models/{model_name}_weights.pth")

    elif args.mode == "evaluate":
        # Load model weights
        model.load_state_dict(torch.load(f"results/models/{model_name}_weights.pth"))
        model.to(device)

        # Evaluate model
        evaluate_model(model, data['valid_loader'], device)