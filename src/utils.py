import os
import random
import numpy as np
import torch
from src.models import UnifiedDeepFM, SLIMModel, EASE

def seed_everything(seed):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model, path):
    """
    Save model weights to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def initialize_model(model_name, params, data, device):
    """
    Initialize the appropriate model based on the configuration.
    """
    if model_name == "DeepFM":
        return UnifiedDeepFM(
            input_dims=[
                len(data['label2idx']['user']),
                len(data['label2idx']['item']),
                len(data['label2idx']['genre']) + 1,
                len(data['label2idx']['writer']),
                len(data['label2idx']['director']),
                10  # Example for year buckets
            ],
            embedding_dim=params['embed_dim'],
            mlp_dims=params['mlp_dims'],
            drop_rate=params['dropout']
        ).to(device)
    elif model_name == "SLIM":
        return SLIMModel(
            num_items=len(data['label2idx']['item']),
            l1_reg=params['l1_reg'],
            l2_reg=params['l2_reg'],
            alpha=params['alpha'],
            max_iter=params['max_iter'],
            device=device
        )
    elif model_name == "EASE":
        return EASE(_lambda=params['lambda'])
    else:
        raise ValueError(f"Unknown model: {model_name}")