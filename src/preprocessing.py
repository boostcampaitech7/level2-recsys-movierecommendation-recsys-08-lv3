import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# ---------------------- 데이터 로드 ----------------------
def basic_data_load(args):
    data_path = args.datapath
    rating_data_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    rating_data_df.drop(['time'], axis=1, inplace=True)
    rating_data_df['interaction'] = 1

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

    total, interaction = data['total'], data['total']['interaction'].to_numpy()
    users, items = total['user'].to_numpy(), total['item'].to_numpy()
    num_users, num_items = len(total['user'].unique()), len(total['item'].unique())

    X = create_csr_matrix(users, items, interaction, num_users, num_items)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_torch = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

    data['basic'] = X_torch
    return data

def create_csr_matrix(users, items, values, num_users, num_items):
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))

# ---------------------- 데이터 분할 ----------------------
def context_data_split(args, data):
    train_list = []
    valid_list = []

    for user, group in data['total'].groupby('user'):
        train, valid = train_test_split(group, test_size=args.split_size, random_state=args.seed)
        train_list.append(train)
        valid_list.append(valid)

    train_data = pd.concat(train_list, axis=0, ignore_index=True)
    valid_data = pd.concat(valid_list, axis=0, ignore_index=True)

    data['train'] = train_data
    data['valid'] = valid_data

    return data

# ---------------------- 부정 샘플 생성 ----------------------
def sample_negative_items(data, seed, num_negative):
    rng = np.random.default_rng(seed)
    items = set(data['item'].unique())
    total = []

    for user, group in data.groupby('user'):
        interacted_items = set(group['item'])
        non_interacted_items = list(items - interacted_items)

        sampled_items = rng.choice(non_interacted_items, size=min(num_negative, len(non_interacted_items)), replace=False)
        negative_samples = pd.DataFrame({
            'user': [user] * len(sampled_items),
            'item': sampled_items,
            'interaction': [0] * len(sampled_items)
        })
        total.append(negative_samples)

    data_total = pd.concat([data] + total, ignore_index=True)
    return data_total

# ---------------------- 추가 정보 처리 ----------------------
def context_data_sideinfo(args, data):
    data_path = args.datapath
    item2idx = data['label2idx']['item']
    
    print('---genre---')
    genres_df = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
    genre2idx = {}
    for idx, genre in enumerate(set(genres_df['genre'])):
        genre2idx[genre] = idx + 1
    genres_df['genre'] = genres_df['genre'].apply(lambda x: [genre2idx[x]])

    item_lst = []
    group_lst = []
    for item, group in genres_df.groupby('item', sort=False):
        item_lst.append(item)
        group_lst.append(group['genre'].sum(axis=0))
    A = pd.DataFrame(item_lst, columns=['item'])
    B = pd.DataFrame({'genre': group_lst})
    genre_df = pd.concat([A, B], axis=1)
    genre_df['item'] = genre_df['item'].apply(lambda x: item2idx[x])

    print('----writer----')
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
    A = pd.DataFrame(item_lst, columns=['item'])
    B = pd.DataFrame({'writer': group_lst})
    writer_df = pd.concat([A, B], axis=1)
    writer_df['item'] = writer_df['item'].apply(lambda x: item2idx[x])

    print('----director----')
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

    print('----year----')
    year_df = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')
    year_df['year'] = year_df['year'].apply(lambda x: year2decade(x))
    year_df['item'] = year_df['item'].apply(lambda x: item2idx[x])

    data['label2idx']['genre'] = genre2idx
    data['label2idx']['writer'] = writer2idx
    data['label2idx']['director'] = director2idx

    data['genre'] = genre_df
    data['writer'] = writer_df
    data['director'] = director_df
    data['year'] = year_df
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
        return -1

# ---------------------- 데이터 병합 ----------------------
def context_data_side_merge(args, data):
    genre_df = data['genre']
    writer_df = data['writer']
    director_df = data['director']
    year_df = data['year']

    if not args.predict:
        data['train'] = pd.merge(data['train'], genre_df, how='left', on='item')
        data['valid'] = pd.merge(data['valid'], genre_df, how='left', on='item')

        data['train'] = pd.merge(data['train'], writer_df, how='left', on='item')
        data['valid'] = pd.merge(data['valid'], writer_df, how='left', on='item')

        data['train'] = pd.merge(data['train'], director_df, how='left', on='item')
        data['valid'] = pd.merge(data['valid'], director_df, how='left', on='item')

        data['train'] = pd.merge(data['train'], year_df, how='left', on='item')
        data['valid'] = pd.merge(data['valid'], year_df, how='left', on='item')

        train_df, valid_df = data['train'], data['valid']
        train_df = train_df.fillna('unknown')
        train_df['year'] = train_df['year'].apply(lambda row: int(row) if row != 'unknown' else 'unknown')
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
        data['train'], data['valid'] = train_df, valid_df
        return data
    else:
        data['total'] = pd.merge(data['total'], genre_df, how='left', on='item')
        data['total'] = pd.merge(data['total'], writer_df, how='left', on='item')
        data['total'] = pd.merge(data['total'], director_df, how='left', on='item')
        data['total'] = pd.merge(data['total'], year_df, how='left', on='item')

        train_df = data['total']
        train_df = train_df.fillna('unknown')
        train_df['year'] = train_df['year'].apply(lambda row: int(row) if row != 'unknown' else 'unknown')
        train_df['director'] = train_df['director'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        train_df['writer'] = train_df['writer'].apply(
            lambda x: [] if isinstance(x, str) and x == 'unknown' else x
        )
        data['total'] = train_df
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
        interaction = row['interaction']

        user += 0
        item += self.n_user
        genre = [g + self.n_user + self.n_item for g in genre]
        if writer:
            writer = [
                w + self.n_user + self.n_item + self.n_genre for w in writer
            ]
        else:
            writer = [self.n_user + self.n_item]
        if director:
            director = [
                d + self.n_user + self.n_item + self.n_genre + self.n_writer for d in director
            ]
        else:
            director = [self.n_user + self.n_item]
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
