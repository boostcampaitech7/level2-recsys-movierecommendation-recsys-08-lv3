import torch
import numpy as np
import pandas as pd
from torch.nn.utils import Dataset
import os
from scipy.sparse import csr_matrix

class InteractionDataset(Dataset):
    def __init__(self, interaction_matrix):
        # interaction_matrix는 numpy 배열 또는 PyTorch 텐서여야 함
        if isinstance(interaction_matrix, np.ndarray):
            self.data = torch.FloatTensor(interaction_matrix)
        elif isinstance(interaction_matrix, torch.Tensor):
            self.data = interaction_matrix
        else:
            raise ValueError("interaction_matrix must be a numpy array or PyTorch tensor")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------------- 데이터 전처리 함수 ----------------------
def basic_data_load(args):
    data_path = args.datapath
    rating_data_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    rating_data_df.drop(['time'], axis=1, inplace=True)
    rating_data_df['interaction']=1
    
    label2idx = {}
    idx2label = {}

    for col in rating_data_df.columns:
        unique_label = rating_data_df[col].unique()
        label2idx[col] = {label: idx for idx, label in enumerate(unique_label)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_label)}
    
    rating_data_df['user'] = rating_data_df['user'].map(label2idx['user'])
    rating_data_df['item'] = rating_data_df['item'].map(label2idx['item'])

    data = {
        'total': rating_data_df,
        'label2idx': label2idx,
        'idx2label': idx2label
    }

    total, interaction = data['total'],data['total']['interaction'].to_numpy()
    users, items, num_users,num_items = total['user'].to_numpy(),total['item'].to_numpy(), len(total['user'].unique()),len(total['item'].unique())

    X = create_csr_matrix(users, items, interaction, num_users, num_items)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_torch = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

    data['basic']= X_torch

    return data

def create_csr_matrix(users, items, values, num_users, num_items):
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

# ---------------------- 데이터 전처리 함수 ----------------------
def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    """
    PyTorch sparse tensor로 변환
    """
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))