import torch
import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def basic_data_load(args):
    '''
        사이드 인포메이션을 쓰지않는 모델을 위한 기본데이터를 불러일으킵니다
    '''
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

def basic_data_split(args,data):
    """
        train, valid 를 따로 총 데이터에서 나누는 역할을합니다.
    """
    # Temporary lists to store DataFrames
    train_list = []
    valid_list = []

    # Split each user's data into train and valid
    for user, group in data['total'].groupby('user'):
        train, valid = train_test_split(group, test_size=args.split_size, random_state=args.seed)
        train_list.append(train)
        valid_list.append(valid)

    # Concatenate all user-specific train and valid DataFrames
    train_data = pd.concat(train_list, axis=0, ignore_index=True)
    valid_data = pd.concat(valid_list, axis=0, ignore_index=True)
    # Add train and valid splits to the data dictionary

    data['train'] = train_data
    data['valid'] = valid_data

    # 유저 아이템 총 길이는 동일해야함
    total_users = len(data['total']['user'].unique())
    total_items = len(data['total']['item'].unique())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, X_torch_train = create_matrices(train_data)
    data['train'] = X_torch_train

    _, X_torch_valid = create_matrices(valid_data)
    data['valid'] = X_torch_valid

    return data

def create_matrices(df,total_users,total_items):
    
    users,items,interactions= df['user'].to_numpy(), df['item'].to_numpy(), df['interaction'].to_numpy()
    csr_matrix =create_csr_matrix(users,items,interactions,total_users,total_items)
    torch_matrix = create_torch_sparse_matrix(users, items, interactions, total_users, total_items, device=device)
    return csr_matrix, torch_matrix



def create_csr_matrix(users, items, values, num_users, num_items):
    '''
        행렬 형태의 데이터 전처리함수입니다.
    '''
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    """
    PyTorch sparse tensor로 변환
    """
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))