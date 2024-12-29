import pandas as pd
import numpy as np
import os
from .basic_data import basic_data_load
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

def context_data_load(args):
    return basic_data_load(args)

# 최종 train_test_split 도출
def context_data_split(args,data):
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

def context_data_sideinfo(args,data):

    # 데이터에서 아이템(영화) 딕셔너리형태로 가져옴
    data_path= args.datapath
    item2idx=data['label2idx']['item']

    # 장르의 번호를 1부터 대응되게만듬
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

    A= pd.DataFrame(item_lst,columns=['item'])
    B= pd.DataFrame({'genre': group_lst})
    
    genre_df=pd.concat([A,B],axis=1)
    genre_df['item'] = genre_df['item'].apply(lambda x: item2idx[x])

    # 작가의 번호를 0부터 대응되게만듬
    writer_df = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')

    writer2idx = {}
    for idx, writer in enumerate(set(writer_df['writer'])):
        writer2idx[writer] = idx
    writer_df['writer'] = writer_df['writer'].apply(lambda x: [writer2idx[x]])

    item_lst = []
    group_lst = []
    
    for item, group in writer_df.groupby('item', sort=False):
        item_lst.append(item)
        group_lst.append(group['writer'].sum(axis=0))  

    A= pd.DataFrame(item_lst, columns=['item'])
    B= pd.DataFrame({'writer': group_lst})
    
    writer_df = pd.concat([A, B], axis=1)
    writer_df['item'] = writer_df['item'].apply(lambda x: item2idx[x])
    
    # 감독의 번호를 0 부터 대응
    director_df = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')

    director2idx = {}
    for idx, director in enumerate(set(director_df['director'])):
        director2idx[director] = idx

    director_df['director'] = director_df['director'].apply(lambda x: [director2idx[x]])

    item_lst = []
    group_lst = []
    
    for item, group in director_df.groupby('item', sort=False):
        item_lst.append(item)
        group_lst.append(group['director'].sum(axis=0)) 

    A = pd.DataFrame(item_lst, columns=['item'])
    B = pd.DataFrame({'director': group_lst})
    
    director_df = pd.concat([A, B], axis=1)
    director_df['item'] = director_df['item'].apply(lambda x: item2idx[x])

    # year2 decade를 통해 고유값으로전달됨
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
def merge_and_fillna(df,merge_cols):
    '''
        각 사이드 인포메이션을 df 에 합치고 결측값을 처리하는 함수입니다
    '''
    for column in merge_cols:
        df=pd.merge(df,column,how='left',on='item')
    df=df.fillna('unknown')

    df['year'] = df['year'].apply(lambda row : int(row) if row!='unknown' else 'unknown')
    df['director'] = df['director'].apply(lambda x: [] if isinstance(x, str) and x == 'unknown' else x)
    df['writer'] = df['writer'].apply(lambda x: [] if isinstance(x, str) and x == 'unknown' else x)

    return df

def context_data_side_merge(args,data):
    '''
        훈련,추론에따라 데이터 통합을 구분지어주는 함수입니다.
    '''
    genre_df = data['genre']
    writer_df = data['writer']
    director_df = data['director']
    year_df = data['year']
    merge_cols=[genre_df,writer_df,director_df,year_df]

    if not args.predict:
        data['train'] = merge_and_fillna(data['train'], merge_cols)
        data['valid'] = merge_and_fillna(data['valid'], merge_cols)
    else:
        data['total'] = merge_and_fillna(data['total'], merge_cols)
        
    return data


class DeepFMDataset(Dataset):
    '''
        Side information 을 활용하는 데이터셋을 만듭니다
        참고할사항은 DeepFm은 임베딩테이블을 공유하므로 사이드인포메이션의 총 유니크한 카테고리 값을
        cumulative 하게 쌓아 임베딩 테이블 의 인덱스값에 상응되게합니다.
    '''
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
    '''
        각 아이템에 multiple 장르,작가 및 감독을 표현애햐가때문에 배치당 길이가 다를수있습니다
        그러한 이유때문에 매 배치마다 길이를 맞추며 패딩을 넣어주는 작업을 여기서 진행합니다.
    '''
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