import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------- 데이터 전처리 ----------------------

def data_pre_for_ease(data_path):
    """
    train_ratings.csv 파일을 로드하고, 시간 기준으로 정렬합니다.
    모든 상호작용을 이진화하여 'rating' 컬럼을 생성합니다.
    """
    data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    # 시점 기준으로 정렬 (가정: 'time' 컬럼이 존재)
    if 'time' in data.columns:
        data = data.sort_values('time')
        data.drop(columns=["time"], inplace=True)
    else:
        print("Warning: 'time' 컬럼이 존재하지 않습니다. 데이터를 시간 순으로 정렬하지 않습니다.")
    data["rating"] = 1.0  # 모든 상호작용을 이진화
    return data

def load_metadata(data_path):
    """
    별도의 메타데이터 파일들(titles.tsv, years.tsv, writers.tsv, directors.tsv, genres.tsv)을 로드하고 통합합니다.
    다중 값(예: 여러 장르, 감독, 작가)을 리스트 형태로 그룹화합니다.
    """
    # titles.tsv
    titles = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep='\t')
    
    # years.tsv
    years = pd.read_csv(os.path.join(data_path, "years.tsv"), sep='\t')
    
    # directors.tsv
    directors = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep='\t')
    
    # genres.tsv
    genres = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep='\t')
    
    # writers.tsv
    writers = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep='\t')
    
    # Merge metadata
    # Since directors, genres, writers can have multiple entries per item, we need to group them as lists
    directors_grouped = directors.groupby('item')['director'].apply(list).reset_index()
    genres_grouped = genres.groupby('item')['genre'].apply(list).reset_index()
    writers_grouped = writers.groupby('item')['writer'].apply(list).reset_index()
    
    # Merge all metadata into a single DataFrame
    metadata = pd.merge(titles, years, on='item', how='left')
    metadata = pd.merge(metadata, directors_grouped, on='item', how='left')
    metadata = pd.merge(metadata, genres_grouped, on='item', how='left')
    metadata = pd.merge(metadata, writers_grouped, on='item', how='left')
    
    # Fill NaN with empty lists for multi-label features
    metadata['director'] = metadata['director'].apply(lambda x: x if isinstance(x, list) else [])
    metadata['genre'] = metadata['genre'].apply(lambda x: x if isinstance(x, list) else [])
    metadata['writer'] = metadata['writer'].apply(lambda x: x if isinstance(x, list) else [])
    
    return metadata

def encode_metadata(metadata):
    """
    메타데이터의 각 유형(writers, directors, genres, titles, years)을 Label Encoding하여 인코딩합니다.
    다중 값(리스트)을 처리하기 위해 각 메타데이터를 리스트의 평균 임베딩으로 집계합니다.
    """
    # Initialize LabelEncoders for each metadata type
    le_writers = LabelEncoder()
    le_directors = LabelEncoder()
    le_genres = LabelEncoder()
    le_titles = LabelEncoder()
    le_years = LabelEncoder()
    
    # For multi-label features, fit on all possible values
    all_writers = [writer for writers in metadata['writer'] for writer in writers]
    all_directors = [director for directors in metadata['director'] for director in directors]
    all_genres = [genre for genres in metadata['genre'] for genre in genres]
    all_titles = metadata['title'].tolist()
    all_years = metadata['year'].tolist()
    
    # Fit LabelEncoders
    le_writers.fit(all_writers)
    le_directors.fit(all_directors)
    le_genres.fit(all_genres)
    le_titles.fit(all_titles)
    le_years.fit(all_years)
    
    # Transform and encode as lists of indices
    metadata['writers_encoded'] = metadata['writer'].apply(lambda x: le_writers.transform(x).tolist() if len(x) > 0 else [0])
    metadata['directors_encoded'] = metadata['director'].apply(lambda x: le_directors.transform(x).tolist() if len(x) > 0 else [0])
    metadata['genres_encoded'] = metadata['genre'].apply(lambda x: le_genres.transform(x).tolist() if len(x) > 0 else [0])
    metadata['titles_encoded'] = metadata['title'].apply(lambda x: le_titles.transform([x])[0]).tolist()
    metadata['years_encoded'] = metadata['year'].apply(lambda x: le_years.transform([x]).tolist()).tolist()  # 리스트로 변환
    
    return metadata, le_writers, le_directors, le_genres, le_titles, le_years

def encode_users_items(data, metadata):
    """
    사용자와 아이템 ID를 Label Encoding하여 인코딩합니다.
    메타데이터에도 인코딩된 아이템 ID를 적용합니다.
    """
    encode_user = LabelEncoder()
    encode_item = LabelEncoder()
    # 원본 ID를 보존하기 위해 새로운 컬럼에 인코딩된 값을 저장
    data['user_encoded'] = encode_user.fit_transform(data["user"])
    data['item_encoded'] = encode_item.fit_transform(data["item"])
    # 메타데이터에도 인코딩된 item 적용
    metadata['item_encoded'] = encode_item.transform(metadata["item"])
    return encode_user, encode_item

def create_csr_matrix(users, items, values, num_users, num_items):
    """
    사용자-아이템 상호작용을 CSR 행렬로 변환합니다.
    """
    if not (len(users) == len(items) == len(values)):
        raise ValueError(f"Users length: {len(users)}, Items length: {len(items)}, Values length: {len(values)}")
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

# ---------------------- RecVAE Hybrid 모델 정의 ----------------------

class RecVaeDataset_NegativeSampling(Dataset):
    """
    Negative Sampling을 적용한 RecVAE Hybrid 모델을 위한 데이터셋 클래스.
    각 사용자당 긍정 예시 1개와 부정 예시 4개를 샘플링하여 반환합니다.
    메타데이터는 고정된 길이로 패딩하여 제공합니다.
    """
    def __init__(self, interactions, metadata, num_items, seq_length=10, negative_ratio=4,
                 writers_max=4, directors_max=2, genres_max=5, years_max=1):
        self.num_items = num_items
        self.seq_length = seq_length
        self.negative_ratio = negative_ratio
        self.users = interactions['user_encoded'].unique()
        self.user_to_items = interactions.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
        self.all_items = set(range(num_items))
        self.metadata = metadata.set_index('item_encoded')
        
        # 최대 길이 설정
        self.writers_max = writers_max
        self.directors_max = directors_max
        self.genres_max = genres_max
        self.years_max = years_max

    def __len__(self):
        return len(self.users)

    def pad_list(self, lst, max_len):
        """
        리스트를 최대 길이로 패딩합니다. 부족한 부분은 0으로 채웁니다.
        """
        if len(lst) >= max_len:
            return lst[:max_len]
        else:
            return lst + [0]*(max_len - len(lst))

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_items = self.user_to_items.get(user, set())

        if len(pos_items) == 0:
            pos_item = 0  # 패딩 아이템
        else:
            pos_item = np.random.choice(list(pos_items))

        # Negative 예시 샘플링
        neg_items = self.all_items - pos_items
        if len(neg_items) >= self.negative_ratio:
            neg_sample = np.random.choice(list(neg_items), size=self.negative_ratio, replace=False)
        else:
            neg_sample = np.random.choice(list(neg_items), size=self.negative_ratio, replace=True)

        # 시퀀스 데이터 준비 (최근 seq_length 아이템)
        items = list(pos_items)
        if len(items) >= self.seq_length:
            item_seq = items[-self.seq_length:]
        else:
            item_seq = [0]*(self.seq_length - len(items)) + items

        item_seq = torch.tensor(item_seq, dtype=torch.long)

        # Positive 예시의 레이블: 1
        item_vector_pos = np.zeros(self.num_items, dtype=np.float32)
        item_vector_pos[list(pos_items)] = 1.0
        item_vector_pos = torch.tensor(item_vector_pos, dtype=torch.float32)

        # Negative 예시의 레이블: 1 (for BCE with targets)
        item_vector_neg = np.zeros(self.num_items, dtype=np.float32)
        item_vector_neg[list(neg_sample)] = 1.0
        item_vector_neg = torch.tensor(item_vector_neg, dtype=torch.float32)

        # 메타데이터 로드
        if pos_item in self.metadata.index:
            metadata = self.metadata.loc[pos_item]
            writers = metadata['writers_encoded']
            directors = metadata['directors_encoded']
            genres = metadata['genres_encoded']
            years = metadata['years_encoded']
        else:
            # 패딩 아이템 또는 메타데이터가 없는 경우
            writers = [0]*self.writers_max
            directors = [0]*self.directors_max
            genres = [0]*self.genres_max
            years = [0]*self.years_max

        # 패딩
        writers_padded = pad_list(writers, self.writers_max) if isinstance(writers, list) else [writers] + [0]*(self.writers_max - 1)
        directors_padded = pad_list(directors, self.directors_max) if isinstance(directors, list) else [directors] + [0]*(self.directors_max - 1)
        genres_padded = pad_list(genres, self.genres_max) if isinstance(genres, list) else [genres] + [0]*(self.genres_max - 1)
        years_padded = pad_list(years, self.years_max) if isinstance(years, list) else [years] + [0]*(self.years_max - 1)

        # Convert lists to tensors
        writers = torch.tensor(writers_padded, dtype=torch.long)
        directors = torch.tensor(directors_padded, dtype=torch.long)
        genres = torch.tensor(genres_padded, dtype=torch.long)
        years = torch.tensor(years_padded, dtype=torch.long)

        return user, item_seq, item_vector_pos, item_vector_neg, writers, directors, genres, years

class RecVAE_Hybrid(nn.Module):
    """
    RecVAE Hybrid 모델 정의.
    사용자 임베딩, 시퀀스 데이터, 그리고 메타데이터 임베딩을 활용하여 잠재 공간을 학습합니다.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, seq_length=10,
                 num_writers=100, num_directors=100, num_genres=50, num_years=30):
        super(RecVAE_Hybrid, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        # 사용자 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 시퀀스 임베딩 (예: 최근 seq_length 아이템)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Item Metadata 임베딩
        self.writers_embedding = nn.Embedding(num_writers, embedding_dim)
        self.directors_embedding = nn.Embedding(num_directors, embedding_dim)
        self.genres_embedding = nn.Embedding(num_genres, embedding_dim)
        self.years_embedding = nn.Embedding(num_years, embedding_dim)

        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim + embedding_dim*4, hidden_dim),  # 64 + 128 + 256 = 448
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, embedding_dim)
        self.logvar_layer = nn.Linear(hidden_dim, embedding_dim)

        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        재매개변수화 트릭을 사용하여 잠재 변수 z를 샘플링합니다.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, user, item_seq, writers, directors, genres, years):
        user_emb = self.user_embedding(user)  # [batch_size, 64]
        print(f"user_emb shape: {user_emb.shape}")  # for Debugging

        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_length, 64]
        lstm_out, _ = self.lstm(item_emb)  # [batch_size, seq_length, 128]
        lstm_last = lstm_out[:, -1, :]  # [batch_size, 128]
        print(f"lstm_last shape: {lstm_last.shape}")  # for Debugging

        # Item Metadata Embeddings
        writers_emb = self.writers_embedding(writers)  # [batch_size, writers_max, 64]
        directors_emb = self.directors_embedding(directors)  # [batch_size, directors_max, 64]
        genres_emb = self.genres_embedding(genres)  # [batch_size, genres_max, 64]
        years_emb = self.years_embedding(years)  # [batch_size, years_max, 64]

        # 평균을 통해 메타데이터 임베딩 집계
        writers_emb_avg = writers_emb.mean(dim=1)  # [batch_size, 64]
        directors_emb_avg = directors_emb.mean(dim=1)  # [batch_size, 64]
        genres_emb_avg = genres_emb.mean(dim=1)  # [batch_size, 64]
        years_emb_avg = years_emb.mean(dim=1)  # [batch_size, 64]
        print(f"writers_emb_avg shape: {writers_emb_avg.shape}")  # for Debugging
        print(f"directors_emb_avg shape: {directors_emb_avg.shape}")  # for Debugging
        print(f"genres_emb_avg shape: {genres_emb_avg.shape}")  # for Debugging
        print(f"years_emb_avg shape: {years_emb_avg.shape}")  # for Debugging

        # 모든 메타데이터 임베딩을 합산 대신 연결하여 [batch_size, 256]로 만듦
        metadata_emb = torch.cat([writers_emb_avg, directors_emb_avg, genres_emb_avg, years_emb_avg], dim=1)  # [batch_size, 256]
        print(f"metadata_emb shape: {metadata_emb.shape}")  # for Debugging

        # 사용자 임베딩과 LSTM 출력, 메타데이터 임베딩 결합
        combined = torch.cat([user_emb, lstm_last, metadata_emb], dim=1)  # [batch_size, 64 + 128 + 256] = [batch_size, 448]
        print(f"combined shape: {combined.shape}")  # for Debugging

        hidden = self.encoder(combined)  # [batch_size, 128]
        mu = self.mu_layer(hidden)  # [batch_size, 64]
        logvar = self.logvar_layer(hidden)  # [batch_size, 64]
        z = self.reparameterize(mu, logvar)  # [batch_size, 64]
        output = self.decoder(z)  # [batch_size, num_items]
        return output, mu, logvar

# ---------------------- GRU4Rec 모델 정의 ----------------------

class GRU4RecDataset(Dataset):
    """
    GRU4Rec 모델을 위한 데이터셋 클래스.
    각 사용자에 대한 시퀀스 데이터를 반환합니다.
    """
    def __init__(self, interactions, num_items, seq_length=10):
        self.num_items = num_items
        self.seq_length = seq_length
        self.users = interactions['user_encoded'].unique()
        self.user_to_items = interactions.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        items = self.user_to_items.get(user, [])
        # 최근 seq_length 아이템을 시퀀스로 사용
        if len(items) >= self.seq_length:
            item_seq = items[-self.seq_length:]
        else:
            # 패딩: 시퀀스 길이에 맞게 0으로 채움
            item_seq = [0]*(self.seq_length - len(items)) + items
        item_seq = torch.tensor(item_seq, dtype=torch.long)
        return user, item_seq

class GRU4Rec(nn.Module):
    """
    GRU4Rec 모델 정의.
    시계열 데이터를 학습하는 GRU 기반 모델입니다.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, num_layers=1, seq_length=10):
        super(GRU4Rec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length

        # 사용자 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 아이템 임베딩
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # GRU 레이어
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)

        # 출력 레이어
        self.fc = nn.Linear(hidden_dim, num_items)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, user, item_seq):
        user_emb = self.user_embedding(user)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_length, embedding_dim]
        gru_out, _ = self.gru(item_emb)  # [batch_size, seq_length, hidden_dim]
        gru_last = gru_out[:, -1, :]  # [batch_size, hidden_dim]
        output = self.fc(gru_last)  # [batch_size, num_items]
        output = self.softmax(output)  # [batch_size, num_items]
        return output

# ---------------------- 손실 함수 정의 ----------------------

def loss_function(recon_x, x_pos, x_neg, mu, logvar):
    """
    RecVAE Hybrid 모델을 위한 손실 함수.
    긍정 예시와 부정 예시에 대한 Binary Cross Entropy 손실과 KL Divergence를 합산합니다.
    """
    BCE_pos = nn.functional.binary_cross_entropy(recon_x, x_pos, reduction='sum')
    BCE_neg = nn.functional.binary_cross_entropy(recon_x, x_neg, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE_pos + BCE_neg + KLD

# ---------------------- Recall@10 계산 함수 ----------------------

def calculate_recall_at_k_val_recvvae(model, val_loader, train_data, val_data, k=10):
    """
    RecVAE Hybrid 모델의 Recall@10을 계산합니다.
    """
    model.eval()
    # Ground truth items per user from val_data
    user_val_items = val_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
    # 사용자별 훈련 아이템 집합
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    total_recall = 0.0
    num_users = len(user_val_items)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating Recall@10 for RecVAE"):
            user, item_seq, item_vector_pos, item_vector_neg, writers, directors, genres, years = batch
            user = user.to(device)
            item_seq = item_seq.to(device)
            item_vector_pos = item_vector_pos.to(device)
            item_vector_neg = item_vector_neg.to(device)
            writers = writers.to(device)
            directors = directors.to(device)
            genres = genres.to(device)
            years = years.to(device)

            recon_batch, mu, logvar = model(user, item_seq, writers, directors, genres, years)
            # scores는 예측된 확률
            scores = recon_batch.cpu().numpy()

            for idx, user_id in enumerate(user.cpu().numpy()):
                val_items = user_val_items.get(user_id, set())
                if not val_items:
                    continue  # If no ground truth, skip

                seen_items = user_train_items.get(user_id, set())
                scores_batch = scores[idx]
                if seen_items:
                    scores_batch[list(seen_items)] = -np.inf  # Mask seen items

                top_k_indices = np.argsort(scores_batch)[-k:][::-1]
                recommended = set(top_k_indices)

                # Ground truth는 이미 인코딩된 상태
                num_correct = len(recommended & val_items)
                recall = num_correct / min(k, len(val_items))
                total_recall += recall

    avg_recall = total_recall / num_users if num_users > 0 else 0.0
    return avg_recall

def calculate_recall_at_k_val_gru4rec(model, val_loader, train_data, val_data, k=10):
    """
    GRU4Rec 모델의 Recall@10을 계산합니다.
    """
    model.eval()
    # Ground truth items per user from val_data
    user_val_items = val_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
    # 사용자별 훈련 아이템 집합
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    total_recall = 0.0
    num_users = len(user_val_items)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating Recall@10 for GRU4Rec"):
            user, item_seq = batch
            user = user.to(device)
            item_seq = item_seq.to(device)
            scores = model(user, item_seq).cpu().numpy()

            for idx, user_id in enumerate(user.cpu().numpy()):
                val_items = user_val_items.get(user_id, set())
                if not val_items:
                    continue  # If no ground truth, skip

                seen_items = user_train_items.get(user_id, set())
                scores_batch = scores[idx]
                if seen_items:
                    scores_batch[list(seen_items)] = -np.inf  # Mask seen items

                top_k_indices = np.argsort(scores_batch)[-k:][::-1]
                recommended = set(top_k_indices)

                # Ground truth는 이미 인코딩된 상태
                num_correct = len(recommended & val_items)
                recall = num_correct / min(k, len(val_items))
                total_recall += recall

    avg_recall = total_recall / num_users if num_users > 0 else 0.0
    return avg_recall

# ---------------------- 앙상블을 위한 함수 정의 ----------------------

def ensemble_recommendations(scores_recvae, scores_gru4rec, k=10, alpha=0.5):
    """
    RecVAE와 GRU4Rec의 점수를 앙상블하여 최종 점수를 계산합니다.
    alpha: RecVAE의 가중치. (1 - alpha): GRU4Rec의 가중치.
    """
    final_scores = alpha * scores_recvae + (1 - alpha) * scores_gru4rec
    topk_items = np.argsort(final_scores, axis=1)[:, -k:][:, ::-1]  # [num_users, top_k]
    return topk_items

# ---------------------- 메인 함수 ----------------------

def pad_list(lst, max_len):
        """
        리스트를 최대 길이로 패딩합니다. 부족한 부분은 0으로 채웁니다.
        """
        if len(lst) >= max_len:
            return lst[:max_len]
        else:
            return lst + [0]*(max_len - len(lst))


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"  # 실제 데이터 경로에 맞게 수정

    # 데이터 로드 및 전처리
    print("Loading training data...")
    data = data_pre_for_ease(data_path)

    print("Loading and merging metadata...")
    metadata = load_metadata(data_path)
    metadata, le_writers, le_directors, le_genres, le_titles, le_years = encode_metadata(metadata)

    print("Encoding users and items...")
    encode_user, encode_item = encode_users_items(data, metadata)

    # 데이터 무결성 확인
    print("\n데이터 무결성 확인:")
    print(data.isnull().sum())
    print("\n데이터의 고유 사용자 수:", data['user'].nunique())
    print("데이터의 고유 아이템 수:", data['item'].nunique())

    # 데이터 분할 (훈련: 90%, 검증: 10%) - 시간 기준 분할
    # 데이터가 시간 순으로 정렬되었으므로, 상위 90%를 훈련 세트, 하위 10%를 검증 세트로 사용
    split_idx = int(0.9 * len(data))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]

    # 검증 세트의 모든 아이템이 LabelEncoder에 포함되었는지 확인
    # 이미 모든 데이터를 기반으로 인코딩했으므로, 모든 아이템이 포함되어야 합니다.
    # 하지만 혹시 모를 문제를 대비하여 확인합니다.
    unseen_items = set(val_data['item_encoded']) - set(encode_item.transform(encode_item.classes_))
    if unseen_items:
        print(f'\nUnseen items in validation set: {unseen_items}')
    else:
        print('\nAll validation items are encoded.')

    # CSR matrix 생성 (필요에 따라 사용)
    X_train = create_csr_matrix(
        train_data['user_encoded'], train_data['item_encoded'], train_data['rating'],
        len(encode_user.classes_), len(encode_item.classes_)
    )
    X_val = create_csr_matrix(
        val_data['user_encoded'], val_data['item_encoded'], val_data['rating'],  # 수정된 부분
        len(encode_user.classes_), len(encode_item.classes_)
    )

    # RecVAE Hybrid 데이터셋 및 데이터로더
    seq_length = 10  # 시퀀스 길이 설정
    negative_ratio = 4  # Negative Sampling Ratio 설정

    print("Creating RecVAE Hybrid datasets and dataloaders...")
    # 설정한 최대 메타데이터 길이 (데이터 분석을 통해 결정하거나 적절히 설정)
    writers_max = metadata['writers_encoded'].apply(len).max()
    directors_max = metadata['directors_encoded'].apply(len).max()
    genres_max = metadata['genres_encoded'].apply(len).max()
    years_max = metadata['years_encoded'].apply(len).max()

    train_dataset_recvae = RecVaeDataset_NegativeSampling(
        train_data, metadata, len(encode_item.classes_),
        seq_length, negative_ratio,
        writers_max, directors_max, genres_max, years_max
    )
    train_loader_recvae = DataLoader(train_dataset_recvae, batch_size=1024, shuffle=True)

    val_dataset_recvae = RecVaeDataset_NegativeSampling(
        val_data, metadata, len(encode_item.classes_),
        seq_length, negative_ratio,
        writers_max, directors_max, genres_max, years_max
    )
    val_loader_recvae = DataLoader(val_dataset_recvae, batch_size=1024, shuffle=False)

    # GRU4Rec 데이터셋 및 데이터로더
    print("Creating GRU4Rec datasets and dataloaders...")
    train_dataset_gru = GRU4RecDataset(train_data, len(encode_item.classes_), seq_length)
    train_loader_gru = DataLoader(train_dataset_gru, batch_size=1024, shuffle=True)

    val_dataset_gru = GRU4RecDataset(val_data, len(encode_item.classes_), seq_length)
    val_loader_gru = DataLoader(val_dataset_gru, batch_size=1024, shuffle=False)

    # 모델 초기화
    num_writers = len(le_writers.classes_)
    num_directors = len(le_directors.classes_)
    num_genres = len(le_genres.classes_)
    num_years = len(le_years.classes_)

    print("Initializing models...")
    model_recvae = RecVAE_Hybrid(
        num_users=len(encode_user.classes_),
        num_items=len(encode_item.classes_),
        embedding_dim=64,
        hidden_dim=128,
        seq_length=seq_length,
        num_writers=num_writers,
        num_directors=num_directors,
        num_genres=num_genres,
        num_years=num_years
    )

    model_gru4rec = GRU4Rec(
        num_users=len(encode_user.classes_),
        num_items=len(encode_item.classes_),
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        seq_length=seq_length
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_recvae = model_recvae.to(device)
    model_gru4rec = model_gru4rec.to(device)

    # 옵티마이저 정의
    optimizer_recvae = optim.Adam(model_recvae.parameters(), lr=1e-3)
    optimizer_gru4rec = optim.Adam(model_gru4rec.parameters(), lr=1e-3)

    # 훈련 설정
    num_epochs = 20

    # 훈련 루프
    for epoch in range(num_epochs):
        ############################
        # RecVAE Hybrid 모델 훈련
        ############################
        print(f"\nEpoch {epoch+1}/{num_epochs} - RecVAE Hybrid Training")
        model_recvae.train()
        train_loss_recvae = 0
        for batch in tqdm(train_loader_recvae, desc=f'Epoch {epoch+1}/{num_epochs} - RecVAE Training'):
            user, item_seq, item_vector_pos, item_vector_neg, writers, directors, genres, years = batch
            user = user.to(device)
            item_seq = item_seq.to(device)
            item_vector_pos = item_vector_pos.to(device)
            item_vector_neg = item_vector_neg.to(device)
            writers = writers.to(device)
            directors = directors.to(device)
            genres = genres.to(device)
            years = years.to(device)

            optimizer_recvae.zero_grad()
            recon_batch, mu, logvar = model_recvae(user, item_seq, writers, directors, genres, years)
            loss = loss_function(recon_batch, item_vector_pos, item_vector_neg, mu, logvar)
            loss.backward()
            train_loss_recvae += loss.item()
            optimizer_recvae.step()

        avg_train_loss_recvae = train_loss_recvae / len(train_loader_recvae.dataset)
        print(f'====> Epoch: {epoch+1} RecVAE Average loss: {avg_train_loss_recvae:.4f}')

        # RecVAE 검증 단계 및 Recall@10 계산
        print(f"Epoch {epoch+1} - RecVAE Validation")
        recall_recvae = calculate_recall_at_k_val_recvvae(model_recvae, val_loader_recvae, train_data, val_data, k=10)
        print(f'====> Epoch: {epoch+1} RecVAE Validation Recall@10: {recall_recvae:.4f}')

        ############################
        # GRU4Rec 모델 훈련
        ############################
        print(f"\nEpoch {epoch+1}/{num_epochs} - GRU4Rec Training")
        model_gru4rec.train()
        train_loss_gru4rec = 0
        for batch in tqdm(train_loader_gru, desc=f'Epoch {epoch+1}/{num_epochs} - GRU4Rec Training'):
            user, item_seq = batch
            user = user.to(device)
            item_seq = item_seq.to(device)

            optimizer_gru4rec.zero_grad()
            output = model_gru4rec(user, item_seq)
            # CrossEntropyLoss는 클래스가 0부터 num_items-1까지일 때 사용
            # 여기서는 시퀀스의 마지막 아이템을 타겟으로 설정
            target = item_seq[:, -1]
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, target)
            loss.backward()
            train_loss_gru4rec += loss.item()
            optimizer_gru4rec.step()

        avg_train_loss_gru4rec = train_loss_gru4rec / len(train_loader_gru.dataset)
        print(f'====> Epoch: {epoch+1} GRU4Rec Average loss: {avg_train_loss_gru4rec:.4f}')

        # GRU4Rec 검증 단계 및 Recall@10 계산
        print(f"Epoch {epoch+1} - GRU4Rec Validation")
        recall_gru4rec = calculate_recall_at_k_val_gru4rec(model_gru4rec, val_loader_gru, train_data, val_data, k=10)
        print(f'====> Epoch: {epoch+1} GRU4Rec Validation Recall@10: {recall_gru4rec:.4f}')

    # ---------------------- 앙상블 및 추천 생성 ----------------------

    print("\nGenerating recommendations for submission (Ensemble of RecVAE Hybrid and GRU4Rec)...")

    model_recvae.eval()
    model_gru4rec.eval()

    # 모든 사용자에 대한 시퀀스 데이터 생성 (훈련 세트 기준)
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(list).to_dict()
    user_sequences = []
    writers_list = []
    directors_list = []
    genres_list = []
    years_list = []

    # 메타데이터 인덱스 설정
    metadata_indexed = metadata.set_index('item_encoded')

    # 루프 최적화: pad_list 함수 외부에서 정의됨
    for user in tqdm(range(len(encode_user.classes_)), desc="Preparing user sequences and metadata"):
        items = user_train_items.get(user, [])
        if len(items) >= seq_length:
            seq = items[-seq_length:]
        else:
            seq = [0]*(seq_length - len(items)) + items
        user_sequences.append(seq)

        # 마지막 아이템의 메타데이터를 사용 (없을 경우 패딩)
        if len(items) > 0:
            last_item = items[-1]
            if last_item in metadata_indexed.index:
                meta = metadata_indexed.loc[last_item]
                writers = meta['writers_encoded']
                directors = meta['directors_encoded']
                genres = meta['genres_encoded']
                years = meta['years_encoded']
            else:
                writers = [0]*train_dataset_recvae.writers_max
                directors = [0]*train_dataset_recvae.directors_max
                genres = [0]*train_dataset_recvae.genres_max
                years = [0]*train_dataset_recvae.years_max
        else:
            writers = [0]*train_dataset_recvae.writers_max
            directors = [0]*train_dataset_recvae.directors_max
            genres = [0]*train_dataset_recvae.genres_max
            years = [0]*train_dataset_recvae.years_max

        # 패딩
        writers_padded = pad_list(writers, train_dataset_recvae.writers_max) if isinstance(writers, list) else [writers] + [0]*(train_dataset_recvae.writers_max - 1)
        directors_padded = pad_list(directors, train_dataset_recvae.directors_max) if isinstance(directors, list) else [directors] + [0]*(train_dataset_recvae.directors_max - 1)
        genres_padded = pad_list(genres, train_dataset_recvae.genres_max) if isinstance(genres, list) else [genres] + [0]*(train_dataset_recvae.genres_max - 1)
        years_padded = pad_list(years, train_dataset_recvae.years_max) if isinstance(years, list) else [years] + [0]*(train_dataset_recvae.years_max - 1)

        writers_list.append(writers_padded)
        directors_list.append(directors_padded)
        genres_list.append(genres_padded)
        years_list.append(years_padded)

    # 텐서 변환 및 디바이스 이동
    user_sequences = torch.tensor(user_sequences, dtype=torch.long).to(device)
    writers_tensor = torch.tensor(writers_list, dtype=torch.long).to(device)
    directors_tensor = torch.tensor(directors_list, dtype=torch.long).to(device)
    genres_tensor = torch.tensor(genres_list, dtype=torch.long).to(device)
    years_tensor = torch.tensor(years_list, dtype=torch.long).to(device)

    with torch.no_grad():
        # RecVAE Hybrid 예측
        scores_recvae, _, _ = model_recvae(
            torch.arange(len(encode_user.classes_)).to(device),
            user_sequences,
            writers_tensor,
            directors_tensor,
            genres_tensor,
            years_tensor
        )
        scores_recvae = scores_recvae.cpu().numpy()

        # GRU4Rec 예측
        scores_gru4rec = model_gru4rec(
            torch.arange(len(encode_user.classes_)).to(device),
            user_sequences
        ).cpu().numpy()

    # 사용자별 훈련 아이템 마스킹
    for user_idx in range(len(encode_user.classes_)):
        seen_items = set(user_train_items.get(user_idx, []))
        if seen_items:
            scores_recvae[user_idx, list(seen_items)] = -np.inf  # RecVAE 마스킹
            scores_gru4rec[user_idx, list(seen_items)] = -np.inf  # GRU4Rec 마스킹

    # 앙상블: RecVAE와 GRU4Rec의 점수를 가중 평균하여 최종 점수 계산
    alpha = 0.5  # RecVAE의 가중치
    top_k = 10
    topk_items = ensemble_recommendations(scores_recvae, scores_gru4rec, k=top_k, alpha=alpha)

    # 사용자 원본 ID 및 아이템 원본 ID 매핑
    user_original_ids = encode_user.inverse_transform(np.arange(len(encode_user.classes_)))
    item_original_ids = encode_item.inverse_transform(np.arange(len(encode_item.classes_)))

    # 제출 파일 작성
    submission = []
    for user_idx, items in enumerate(topk_items):
        user_id = user_original_ids[user_idx]
        recommended_items = item_original_ids[items]
        for item_id in recommended_items:
            submission.append([user_id, item_id])

    submission_df = pd.DataFrame(submission, columns=['user', 'item'])  # 헤더 추가
    submission_df.to_csv('Ensemble_RecVAE_GRU4Rec_submission_20.csv', index=False, header=True)  # 헤더 포함
    print('Submission file saved as Ensemble_RecVAE_GRU4Rec_submission.csv')

    # ---------------------- 최종 Recall@10 계산 ----------------------

    # RecVAE Final Recall
    print("\nCalculating Final Recall@10 on validation set (RecVAE)...")
    recall_val_recvae = calculate_recall_at_k_val_recvvae(model_recvae, val_loader_recvae, train_data, val_data, k=10)
    print(f'Final Recall@10 on validation set (RecVAE): {recall_val_recvae:.4f}')

    # GRU4Rec Final Recall
    print("Calculating Final Recall@10 on validation set (GRU4Rec)...")
    recall_val_gru4rec = calculate_recall_at_k_val_gru4rec(model_gru4rec, val_loader_gru, train_data, val_data, k=10)
    print(f'Final Recall@10 on validation set (GRU4Rec): {recall_val_gru4rec:.4f}')

    # 앙상블 Recall@10 (가중 평균을 사용)
    print("Calculating Final Recall@10 on validation set (Ensemble)...")
    recall_val_ensemble = 0.5 * recall_val_recvae + 0.5 * recall_val_gru4rec
    print(f'Final Recall@10 on validation set (Ensemble): {recall_val_ensemble:.4f}')
