import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

def context_data_load(args):
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
    
    #user 와 item categori화 (deepfm) 필요.
    return data

# 최종 train_test_split 도출
def context_data_split(args,data):
    """
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
            'interaction': [0] * len(sampled_items)  # 부정 샘플은 rating=0
        })

        total.append(negative_samples)

    # 원본 데이터와 부정 샘플을 합침
    data_total = pd.concat([data] + total, ignore_index=True)

    return data_total

def context_data_sideinfo (args,data):
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

    print('----wrtier----')
    writer_df = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')

    # Create a mapping from writer to a unique index
    writer2idx = {}
    for idx, writer in enumerate(set(writer_df['writer'])):
        writer2idx[writer] = idx

    # Replace writer names with their corresponding indices
    writer_df['writer'] = writer_df['writer'].apply(lambda x: [writer2idx[x]])

    # Group by items to aggregate writers
    item_lst = []
    group_lst = []
    for item, group in writer_df.groupby('item', sort=False):
        item_lst.append(item)
        group_lst.append(group['writer'].sum(axis=0))  # Combine writer indices into a list

    # Create the final DataFrame
    A = pd.DataFrame(item_lst, columns=['item'])
    B = pd.DataFrame({'writer': group_lst})
    writer_df = pd.concat([A, B], axis=1)
    writer_df['item'] = writer_df['item'].apply(lambda x: item2idx[x])
    
    print('----director----')
    director_df = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

    # Create a mapping from director to a unique index
    director2idx = {}
    for idx, director in enumerate(set(director_df['director'])):
        director2idx[director] = idx

    # Replace director names with their corresponding indices
    director_df['director'] = director_df['director'].apply(lambda x: [director2idx[x]])

    # Group by items to aggregate directors
    item_lst = []
    group_lst = []
    for item, group in director_df.groupby('item', sort=False):
        item_lst.append(item)
        group_lst.append(group['director'].sum(axis=0))  # Combine director indices into a list

    # Create the final DataFrame
    A = pd.DataFrame(item_lst, columns=['item'])
    B = pd.DataFrame({'director': group_lst})
    director_df = pd.concat([A, B], axis=1)

    print('----year----')
    year_df = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    year_df['year']=year_df['year'].apply(lambda x: year2decade(x))
    year_df['item'] = year_df['item'].apply(lambda x: item2idx[x])

    data['label2idx']['genre']=genre2idx
    data['label2idx']['writer']=writer2idx
    data['label2idx']['director']=director2idx

    data['genre']=genre_df
    data['writer']=writer_df
    data['director']=director_df
    data['year']=year_df

    return data


def year2decade(x):
    if 1920 <= x < 1930:
        return int(0)
    elif 1930 <= x < 1940:
        return int(1)
    elif 1940 <= x < 1950:
        return int(2)
    elif 1950 <= x < 1960:
        return int(3)
    elif 1960 <= x < 1970:
        return int(4)
    elif 1970 <= x < 1980:
        return int(5)
    elif 1980 <= x < 1990:
        return int(6)
    elif 1990 <= x < 2000:
        return int(7)
    elif 2000 <= x < 2010:
        return int(8)
    elif 2010 <= x < 2020:
        return int(9)
    else:
        return -1  # Return -1 for years outside the range

def context_data_side_merge(args,data):

    genre_df = data['genre']
    writer_df = data['writer']
    director_df = data['director']
    year_df = data['year']

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


class InteractionDataset(Dataset):
    def __init__(self, dataframe, n_user, n_item, n_genre, n_writer, n_director, n_year):
        self.data = dataframe
        self.n_user = n_user
        self.n_item = n_item
        self.n_genre = n_genre
        self.n_writer = n_writer
        self.n_director = n_director
        self.n_year = n_year

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user = row['user']
        item = row['item']
        genre = row['genre']
        writer = row['writer']
        director = row['director']
        year = row['year']
        interaction = row['interaction']  # Add interaction/label
        
        user += 0
        item += self.n_user
        genre = [g + self.n_user + self.n_item for g in genre]
        if writer:
            writer = [
                w + self.n_user + self.n_item + self.n_genre for w in writer
                ]
        else:
            writer=[self.n_user + self.n_item]
        if director:
            director = [
                d + self.n_user + self.n_item + self.n_genre + self.n_writer for d in director   
                ]
        else:
            director=[self.n_user + self.n_item]
        if year == 'unknown':
            year = self.n_user + self.n_item  # Padding index for unknown
        else:
            year = year + self.n_user + self.n_item + self.n_genre + self.n_writer + self.n_director


        user_tensor = torch.tensor(user, dtype=torch.long)
        item_tensor = torch.tensor(item, dtype=torch.long)
        year_tensor = torch.tensor(year, dtype=torch.long)

        interaction_tensor = torch.tensor(interaction, dtype=torch.float)  # Ensure it’s a float for BCELoss

        return user_tensor, item_tensor, genre, writer, director, year_tensor, interaction_tensor

def collate_fn(batch, padding_idx):
    # Unpack batch
    users, items, genres, writers, directors, years, interactions = zip(*batch)

    # Stack fixed-length tensors
    user_tensor = torch.stack(users)
    item_tensor = torch.stack(items)
    year_tensor = torch.stack(years)
    interaction_tensor = torch.stack(interactions)

    # Pad variable-length sequences
    max_genre_length = max(len(genre) for genre in genres)
    padded_genres = [
        genre + [padding_idx] * (max_genre_length - len(genre)) for genre in genres
    ]
    genre_tensor = torch.tensor(padded_genres, dtype=torch.long)

    max_writer_length = max(len(writer) for writer in writers)
    padded_writers = [
        writer + [padding_idx] * (max_writer_length - len(writer)) for writer in writers
    ]
    writer_tensor = torch.tensor(padded_writers, dtype=torch.long)

    max_director_length = max(len(director) for director in directors)
    padded_directors = [
        director + [padding_idx] * (max_director_length - len(director)) for director in directors
    ]
    director_tensor = torch.tensor(padded_directors, dtype=torch.long)

    return user_tensor, item_tensor, genre_tensor, writer_tensor, director_tensor, year_tensor, interaction_tensor