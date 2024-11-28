import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def seed_everything(args):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        seed=args.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def data_load(args):
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
    
    data = {
        'total': rating_data_df,
        'label2idx': label2idx,
        'idx2label': idx2label
    }
    #user 와 item categori화 (deepfm) 필요.
    return data

# 최종 train_test_split 도출
def train_valid_split(data, args):
    """
    """
    # Temporary lists to store DataFrames
    train_list = []
    valid_list = []

    # Split each user's data into train and valid
    for user, group in data['total'].groupby('user'):
        train, valid = train_test_split(group, test_size=args.test_size, random_state=args.seed)
        train_list.append(train)
        valid_list.append(valid)

    # Concatenate all user-specific train and valid DataFrames
    train_data = pd.concat(train_list, axis=0, ignore_index=True)
    valid_data = pd.concat(valid_list, axis=0, ignore_index=True)

    # Add train and valid splits to the data dictionary
    data['train'] = train_data
    data['valid'] = valid_data

    return data

def sample_negative_items(data, seed, num_negative):
    """
    Parameters:
    - data (DataFrame): 전체 사용자-아이템 상호작용 데이터
    - seed (int): 랜덤 시드 값 (재현성을 위해 사용)
    - num_negative (int): 생성할 부정 샘플의 수

    Returns:
    - data_total (DataFrame): 부정 샘플이 포함된 전체 데이터프레임
    """
    rng = np.random.default_rng(seed)  # 난수 생성기를 시드와 함께 초기화(재현성 보장)
    items = set(data['item'].unique())  # 전체 아이템 집합
    total = []

    for user, group in data.groupby('user'):
        interacted_items = set(group['item'])  # 사용자가 이미 본 아이템 추출
        non_interacted_items = list(items - interacted_items)  # 사용자가 보지 않은 아이템 추출

        if len(non_interacted_items) < num_negative:
            sampled_items = non_interacted_items
        else:
            sampled_items = rng.choice(non_interacted_items, size=num_negative, replace=False)

        # 부정 샘플 데이터프레임 생성
        negative_samples = pd.DataFrame({
            'user': [user] * len(sampled_items),
            'item': sampled_items,
            'interaction': [0] * len(sampled_items)  # 부정 샘플은 interaction=0
        })

        total.append(negative_samples)

    # 원본 데이터와 부정 샘플을 합침
    data_total = pd.concat([data] + total, ignore_index=True)

    return data_total

def data_sideinfo (args,data):
    data_path= args.datapath
    item2idx=data['label2idx']['item']
    print('---genre---')
    genres_df= pd.read_csv(os.path.join(data_path,'genres.tsv'),sep='\t')
    genre2idx={}
    for idx,genre in enumerate(set(genres_df['genre'])):
        genre2idx[genre]=idx+1
    genres_df['genre']=genres_df['genre'].apply(lambda x: [genre2idx[x]])

    item_lst=[]
    group_lst=[]
    for item,group in genres_df.groupby('item',sort=False):
        item_lst.append(item)
        group_lst.append(group['genre'].sum(axis=0))
    A=pd.DataFrame(item_lst,columns=['item'])
    B = pd.DataFrame({'genre': group_lst})
    genre_df=pd.concat([A,B],axis=1)
    genre_df['item'] = genre_df['item'].apply(lambda x: item2idx[x])

    # print('----wrtier----')
    # writer_df = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')

    # # Create a mapping from writer to a unique index
    # writer2idx = {}
    # for idx, writer in enumerate(set(writer_df['writer'])):
    #     writer2idx[writer] = idx

    # # Replace writer names with their corresponding indices
    # writer_df['writer'] = writer_df['writer'].apply(lambda x: [writer2idx[x]])

    # # Group by items to aggregate writers
    # item_lst = []
    # group_lst = []
    # for item, group in writer_df.groupby('item', sort=False):
    #     item_lst.append(item)
    #     group_lst.append(group['writer'].sum(axis=0))  # Combine writer indices into a list

    # # Create the final DataFrame
    # A = pd.DataFrame(item_lst, columns=['item'])
    # B = pd.DataFrame({'writer': group_lst})
    # writer_df = pd.concat([A, B], axis=1)
    # writer_df['item'] = writer_df['item'].apply(lambda x: item2idx[x])
    
    # print('----director----')
    # director_df = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

    # # Create a mapping from director to a unique index
    # director2idx = {}
    # for idx, director in enumerate(set(director_df['director'])):
    #     director2idx[director] = idx

    # # Replace director names with their corresponding indices
    # director_df['director'] = director_df['director'].apply(lambda x: [director2idx[x]])

    # # Group by items to aggregate directors
    # item_lst = []
    # group_lst = []
    # for item, group in director_df.groupby('item', sort=False):
    #     item_lst.append(item)
    #     group_lst.append(group['director'].sum(axis=0))  # Combine director indices into a list

    # # Create the final DataFrame
    # A = pd.DataFrame(item_lst, columns=['item'])
    # B = pd.DataFrame({'director': group_lst})
    # director_df = pd.concat([A, B], axis=1)

    # print('----year----')
    # year_df = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    # year_df['year']=year_df['year'].apply(lambda x: year2decade(x))
    # year_df['item'] = year_df['item'].apply(lambda x: item2idx[x])

    data['label2idx']['genre']=genre2idx
    # data['label2idx']['writer']=writer2idx
    # data['label2idx']['director']=director2idx

    data['genre']=genre_df
    # data['writer']=writer_df
    # data['director']=director_df
    # data['year']=year_df

    return data


# def year2decade(x):
#     if 1920 <= x < 1930:
#         return int(0)
#     elif 1930 <= x < 1940:
#         return int(1)
#     elif 1940 <= x < 1950:
#         return int(2)
#     elif 1950 <= x < 1960:
#         return int(3)
#     elif 1960 <= x < 1970:
#         return int(4)
#     elif 1970 <= x < 1980:
#         return int(5)
#     elif 1980 <= x < 1990:
#         return int(6)
#     elif 1990 <= x < 2000:
#         return int(7)
#     elif 2000 <= x < 2010:
#         return int(8)
#     elif 2010 <= x < 2020:
#         return int(9)
#     else:
#         return -1  # Return -1 for years outside the range

def side_merge(args,data):

    if not args.predict:
        data['train'] = pd.merge(data['train'],genre_df,how='left',on='item')
        data['valid']= pd.merge(data['valid'],genre_df,how='left',on='item')

        data['train'] = pd.merge(data['train'],writer_df,how='left',on='item')
        data['valid']= pd.merge(data['valid'],writer_df,how='left',on='item')

        data['train'] = pd.merge(data['train'], director_df,how='left',on='item')
        data['valid']= pd.merge(data['valid'], director_df,how='left',on='item')

        data['train'] = pd.merge(data['train'], year_df,how='left',on='item')
        data['valid'] = pd.merge(data['valid'], year_df,how='left',on='item')
        train_df,valid_df=data['train'],data['valid']

        train_df=train_df.fillna('unknown')
        train_df['year'] = train_df['year'].apply(lambda row: int(row) if row!='unknown' else 'unknown')
        train_df['director'] = train_df['director'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        train_df['writer'] = train_df['writer'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        valid_df = valid_df.fillna('unknown')
        valid_df['year'] = valid_df['year'].apply(
            lambda row: int(row) if row != 'unknown' else 'unknown'
        )
        valid_df['director'] = valid_df['director'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        valid_df['writer'] = valid_df['writer'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        data['train'],data['valid']= train_df,valid_df
        return data
    else:
        data['total'] = pd.merge(data['total'],genre_df,how='left',on='item')
        data['total'] = pd.merge(data['total'],writer_df,how='left',on='item')
        data['total'] = pd.merge(data['total'], director_df,how='left',on='item')
        data['total'] = pd.merge(data['total'], year_df,how='left',on='item')
        train_df=data['total']
        train_df=train_df.fillna('unknown')
        train_df['year'] = train_df['year'].apply(lambda row: int(row) if row!='unknown' else 'unknown')
        train_df['director'] = train_df['director'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        train_df['writer'] = train_df['writer'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        data['total']=train_df
        return data
import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): A pandas DataFrame containing user, item, rating, and genre columns.
        """
        self.user_ids = torch.tensor(data['user'].values, dtype=torch.long)
        self.item_ids = torch.tensor(data['item'].values, dtype=torch.long)
        self.interaction = torch.tensor(data['interaction'].values, dtype=torch.float32)
        self.genres = data['genre'].apply(lambda x: torch.tensor(x, dtype=torch.long)).tolist()  # List of tensors

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
            'interaction': self.interaction[idx],
            'genre': self.genres[idx]  # A tensor of genre indices
        }

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad genres and create a batch.
    Args:
        batch (list of dict): List of samples from the dataset.
    
    Returns:
        dict: A batch with padded genres.
    """
    user_ids = torch.stack([sample['user_id'] for sample in batch])
    item_ids = torch.stack([sample['item_id'] for sample in batch])
    interaction = torch.stack([sample['interaction'] for sample in batch])

    # Pad the genre sequences
    genres = [sample['genre'] for sample in batch]  # List of genre tensors
    padded_genres = pad_sequence(genres, batch_first=True, padding_value=0)  # Padding with 0

    return {
        'user_id': user_ids,
        'item_id': item_ids,
        'interaction': interaction,
        'genre': padded_genres
    }
import torch
import torch.nn as nn

import torch
from torch import nn

class NCF_GM(nn.Module):
    def __init__(self, num_user, num_item, num_genres, embedding_dim=32, side_emb_dim=16, dropout_rate=0.2):
        super(NCF_GM, self).__init__()
        
        # Separate embeddings for NCF
        self.ncf_user_embedding = nn.Embedding(num_user, embedding_dim)
        self.ncf_item_embedding = nn.Embedding(num_item, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, side_emb_dim, padding_idx=0)
        
        # Separate embeddings for GM
        self.gm_user_embedding = nn.Embedding(num_user, embedding_dim)
        self.gm_item_embedding = nn.Embedding(num_item, embedding_dim)
        
        # Fully connected layers for NCF
        self.fc1 = nn.Linear(2 * embedding_dim + side_emb_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, user_ids, item_ids, genres):
        # ===== GM Component =====
        # Embed users and items for GM
        gm_user_emb = self.gm_user_embedding(user_ids)  # Shape: (batch_size, embedding_dim)
        gm_item_emb = self.gm_item_embedding(item_ids)  # Shape: (batch_size, embedding_dim)
        
        # Compute the interaction (dot product or generalized interaction)
        gm_output = torch.sum(gm_user_emb * gm_item_emb, dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # ===== NCF Component =====
        # Embed users and items for NCF
        ncf_user_emb = self.ncf_user_embedding(user_ids)  # Shape: (batch_size, embedding_dim)
        ncf_item_emb = self.ncf_item_embedding(item_ids)  # Shape: (batch_size, embedding_dim)
        
        # Embed genres and aggregate (mean)
        genre_emb = self.genre_embedding(genres)  # Shape: (batch_size, num_genres, side_emb_dim)
        genre_emb = genre_emb.mean(dim=1)  # Shape: (batch_size, side_emb_dim)

        # Concatenate embeddings
        ncf_input = torch.cat([ncf_user_emb, ncf_item_emb, genre_emb], dim=-1)  # Shape: (batch_size, 2*embedding_dim + side_emb_dim)
        
        # Pass through MLP layers
        x = self.relu(self.bn1(self.fc1(ncf_input)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        ncf_output = self.fc4(x)  # Shape: (batch_size, 1)

        # ===== Combine GM and NCF Outputs =====
        # Combine GM and NCF predictions
        output = torch.sigmoid(gm_output + ncf_output)  # Combine and apply sigmoid activation
        
        return output


def train_model(args,data):
    n_user=len(data['label2idx']['user'])
    n_item=len(data['label2idx']['item'])
    n_genre=len(data['label2idx']['genre'])+1
    # n_writer = len(data['label2idx']['writer'])
    # n_director = len(data['label2idx']['director'])
    # n_year=10

    if not args.predict:
        train_df,valid_df = data['train'],data['valid']
        print('-----train loading------')
        dataset = RecommendationDataset(train_df)

        # Use partial to pass the padding index to collate_fn
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn, num_workers=4)

        print('-----valid loading------')
        valid_dataset = RecommendationDataset(valid_df)
        valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, collate_fn=collate_fn)
    else:
        train_df=data['total']
        print('-----train loading------')
        dataset = RecommendationDataset(train_df)  # Your dataset
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn, num_workers=4)

    embedding_dim = 32
    side_emb_dim = 16
    print('---model----')
    model = NCF_GM(num_user=num_users, num_item=num_items, genre_item=num_genres, 
                embedding_dim=embedding_dim, side_emb_dim=side_emb_dim)

    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for implicit feedback
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    num_epochs = 10
    print('----train----')
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        
        for batch in tqdm(dataloader):
            # Move data to the same device as the model
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            interaction = batch['interaction'].to(device)  # Target values (1 for positive interaction, 0 for negative)
            genres = batch['genre'].to(device)   # Padded genre indices
        
            # Forward pass
            outputs = model(user_ids, item_ids, genres).squeeze()  # Predicted scores (sigmoid output)

            # Compute loss
            loss = criterion(outputs, interaction)

            # Backward pass
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
    if not args.preict:
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                user_ids, item_ids, genres, writers, directors, years, interaction = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                genres = genres.to(device)
                writers = writers.to(device)
                directors = directors.to(device)
                years = years.to(device)
                interaction = interaction.to(device)

                # Forward pass
                outputs = model(user_ids, item_ids, genres, writers, directors, years).squeeze()
                loss = criterion(outputs, interaction)
                val_loss += loss.item()

                # Compute accuracy (optional, for binary classification)
                preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions
                val_correct += (preds == interaction).sum().item()
                val_total += interaction.size(0)

        avg_val_loss = val_loss / len(valid_dataloader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    return model

import pandas as pd
from tqdm import tqdm

def generate_prediction_base(args,data):
    """
    Precompute metadata and generate prediction base for unseen items for each user.
    
    Args:
        data_total (DataFrame): Dataset containing all interactions between users and items.
        prediction_df (DataFrame): DataFrame containing user prediction information.
        genre_df, writer_df, director_df, year_df (DataFrames): Metadata DataFrames.

    Returns:
        DataFrame: Combined prediction DataFrame for all users.
    """
    # Step 1: Precompute metadata for all items
    all_items = data['total']['item'].unique()
    metadata_df = pd.DataFrame({'item': all_items})

    # Merge metadata
    for df in [data['genre']]:
        metadata_df = pd.merge(metadata_df, df, how='left', on='item')

    # Fill missing values and clean columns
    # metadata_df = metadata_df.fillna('unknown')
    # metadata_df['year'] = metadata_df['year'].apply(lambda row: int(row) if row != 'unknown' else 'unknown')
    # metadata_df['director'] = metadata_df['director'].apply(
    #     lambda x: [] if isinstance(x, str) and x == 'unknown' else x
    # )
    # metadata_df['writer'] = metadata_df['writer'].apply(
    #     lambda x: [] if isinstance(x, str) and x == 'unknown' else x
    # )

    # Step 2: Generate predictions for all users
    predict_base = []
    for user_id in tqdm(prediction_df['user'].unique(), desc="Generating predictions"):
        # Items already seen by the user
        seen_items = set(data_total[data_total['user'] == user_id]['item'])
        # Items not seen by the user
        unseen_items = set(metadata_df['item']) - seen_items

        # Create user-specific prediction DataFrame
        user_predict_df = pd.DataFrame({
            'user': [user_id] * len(unseen_items),
            'item': list(unseen_items),
            'interaction': [0] * len(unseen_items)  # Dummy interactions
        })

        # Merge precomputed metadata
        user_predict_df = pd.merge(user_predict_df, metadata_df, how='left', on='item')

        # Append to predict_base
        predict_base.append(user_predict_df)

    # Combine all user-specific DataFrames
    return pd.concat(predict_base, ignore_index=True)

import torch
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
import pandas as pd

def evaluate(args,model, predict_base,data,top_k=10):
    
    """
    Generate predictions for all users using a trained model.

    Args:
        model: The trained PyTorch model for predictions.
        predict_base (DataFrame): DataFrame containing the prediction base for users and items.
        data_total (DataFrame): Dataset with all user-item interactions.
        device: PyTorch device (e.g., 'cuda' or 'cpu').
        n_user, n_item, n_genre, n_writer, n_director, n_year: Number of unique users, items, and features.
        batch_size (int): Batch size for DataLoader.
        top_k (int): Number of top items to recommend for each user.

    Returns:
        DataFrame: DataFrame containing predictions with columns ['user', 'item', 'score'].
    """

    n_user=len(data['label2idx']['user'])
    n_item=len(data['label2idx']['item'])
    n_genre=len(data['label2idx']['genre'])+1
    # n_writer = len(data['label2idx']['writer'])
    # n_director = len(data['label2idx']['director'])
    # n_year=10
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():
        for user_id in tqdm(data['total']['user'].unique(), desc="Evaluating users"):
            # Filter the prediction base for the current user
            predict_df = predict_base[predict_base['user'] == user_id]

            # Prepare the dataset and DataLoader
            predict_data = InteractionDataset(
                dataframe=predict_df,
                n_user=n_user,
                n_item=n_item,
                n_genre=n_genre,
                n_writer=n_writer,
                n_director=n_director,
                n_year=n_year
            )

            padding_idx = n_user + n_item
            custom_collate_fn = partial(collate_fn, padding_idx=padding_idx)
            predict_dataloader = DataLoader(
                predict_data,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=custom_collate_fn
            )

            # Store predictions for the current user
            user_predictions = []

            for batch in predict_dataloader:
                # Move batch data to device
                user_ids, item_ids, genres, writers, directors, years, interaction = [
                    tensor.to(device) for tensor in batch
                ]

                # Generate predictions
                outputs = model(user_ids, item_ids, genres, writers, directors, years).squeeze()

                # Handle scalar outputs (exception case)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)

                # Collect predictions (item_id, score)
                user_predictions.extend(zip(item_ids.cpu().numpy(), outputs.cpu().numpy()))

            # Sort predictions and select the top_k items
            user_predictions = sorted(user_predictions, key=lambda x: x[1], reverse=True)[:top_k]

            # Add the predictions for the current user
            predictions.extend([(user_id, item_id, score) for item_id, score in user_predictions])

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['user', 'item', 'score'])
    return predictions_df






def main(args):

    # seed
    seed_everything(args)
    # data load
    data = data_loader(args)
    # data_ side
    data = data_sideinfo(args,data)
    params= args.model
    if not arg.predict:
        # train
        data= train_valid_split(data, args)
        data['train'] = sample_negative_items(data['train'], args.seed, args.negative_samples)
        data=side_merge(args,data)

        #train
        model = train_model(args,data)
        print('---end---')
    else:
        # train
        data= train_valid_split(data, args)
        data['train'] = sample_negative_items(data['train'], args.seed, args.negative_samples)
        data=side_merge(args,data)

        #train
        model = train_model(args,data)
        # Inference
        predict_base=generate_prediction_base(args,data)
        result = evaluate(args,model,predict_base,data)
        print(result)
        

    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
   # arg('--config', '-c', '--c', type=str, 
    #    help='Configuration 파일을 설정합니다.', required=True)
    arg('--datapath',type=str,
        help='datapath를 지정하시오')
    arg('--model', '-m', '--m', type=str, 
        choices=['DeepFM','MF','NeuralMF'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed','-s','--s',type=int,
        help= '시드설정')
    arg('--device', '-d', '--d', type=str, 
        , help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--negative',type=int,help='negative sampling갯수를 지정합니다')
    #arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    #arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    #arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    #arg('--loss', '-l', '--l', type=str)
    #arg('--metrics', '-met', '--met', type=ast.literal_eval)
    #arg('--train', '-t', '--t', type=ast.literal_eval)
    
    args = parser.parse_args()
    args.datapath=''
    args.seed=42
    args.device='cuda'