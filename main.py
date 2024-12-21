import argparse
import torch
from src.preprocessing import basic_data_load, context_data_split, context_data_side_merge
from src.utils import seed_everything, initialize_model
from src.evaluate import evaluate_model  # evaluate_model을 evaluate.py에서 가져옴
import yaml

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

    # Load and preprocess data
    data = basic_data_load(config['data'])
    data = context_data_split(config['data'], data)
    data = context_data_side_merge(config['data'], data)

    # Initialize model
    model_name = config['model']['name']
    model = initialize_model(model_name, config['model'], data, device)

    if args.mode == "evaluate":
        # Load model weights
        model.load_state_dict(torch.load(config['train']['save_path']))
        model.to(device)

        # Evaluate model
        evaluate_model(model, data['valid_loader'], device)