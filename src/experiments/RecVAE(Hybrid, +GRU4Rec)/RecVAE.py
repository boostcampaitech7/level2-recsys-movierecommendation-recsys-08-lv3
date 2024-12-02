import os
import json  # 2. import json 추가
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 데이터 경로 설정
DATA_PATH = '/data/ephemeral/home/KJPark/data/train/'  # 실제 데이터 경로에 맞게 수정

# 데이터 로드
train_ratings = pd.read_csv(os.path.join(DATA_PATH, "train_ratings.csv"))
with open(os.path.join(DATA_PATH, "Ml_item2attributes.json"), 'r') as f:
    item2attributes = json.load(f)

titles = pd.read_csv(os.path.join(DATA_PATH, "titles.tsv"), sep='\t')
years = pd.read_csv(os.path.join(DATA_PATH, "years.tsv"), sep='\t')
directors = pd.read_csv(os.path.join(DATA_PATH, "directors.tsv"), sep='\t')
genres = pd.read_csv(os.path.join(DATA_PATH, "genres.tsv"), sep='\t')
writers = pd.read_csv(os.path.join(DATA_PATH, "writers.tsv"), sep='\t')

# 사용자와 아이템 인덱싱
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

train_ratings['user'] = user_encoder.fit_transform(train_ratings['user'])
train_ratings['item'] = item_encoder.fit_transform(train_ratings['item'])

num_users = train_ratings['user'].nunique()
num_items = train_ratings['item'].nunique()

print(f'Number of users: {num_users}, Number of items: {num_items}')

# 장르 인코딩: MultiLabelBinarizer 사용
mlb_genre = MultiLabelBinarizer()

# 아이템별 장르 리스트 생성
item_genres = genres.groupby('item')['genre'].apply(list).to_dict()

# '1'과 같은 예기치 않은 장르 값 처리
for item, genre_list in item_genres.items():
    cleaned_genres = [genre if isinstance(genre, str) else 'Unknown' for genre in genre_list]
    item_genres[item] = cleaned_genres

# 아이템 인코더가 가진 클래스와 일치하도록 아이템 필터링
filtered_item_genres = {item: genres for item, genres in item_genres.items() if item in item_encoder.classes_}

# 아이템별 장르 리스트 준비
list_of_genres = list(filtered_item_genres.values())

# MultiLabelBinarizer 학습
mlb_genre.fit(list_of_genres)

# 이제 mlb_genre.classes_ 사용 가능
num_genres = len(mlb_genre.classes_)
print(f'Number of genres after encoding: {num_genres}')

# 모든 아이템에 대해 장르 매트릭스 초기화 (초기값은 0)
genre_matrix = np.zeros((num_items, num_genres), dtype=np.float32)

# 아이템별 장르 매트릭스 채우기
for item, genre_list in filtered_item_genres.items():
    item_idx = item_encoder.transform([item])[0]
    genre_encoded = mlb_genre.transform([genre_list])[0]
    genre_matrix[item_idx] = genre_encoded

# 감독 인코딩: MultiLabelBinarizer 사용
mlb_director = MultiLabelBinarizer()

# 아이템별 감독 리스트 생성
item_directors = directors.groupby('item')['director'].apply(list).to_dict()

# 아이템 인코더가 가진 클래스와 일치하도록 아이템 필터링
filtered_item_directors = {item: directors for item, directors in item_directors.items() if item in item_encoder.classes_}

# 아이템별 감독 리스트 준비
list_of_directors = list(filtered_item_directors.values())

# MultiLabelBinarizer 학습
mlb_director.fit(list_of_directors)

# 이제 mlb_director.classes_ 사용 가능
num_directors = len(mlb_director.classes_)
print(f'Number of directors after encoding: {num_directors}')

# 모든 아이템에 대해 감독 매트릭스 초기화 (초기값은 0)
director_matrix = np.zeros((num_items, num_directors), dtype=np.float32)

# 아이템별 감독 매트릭스 채우기
for item, director_list in filtered_item_directors.items():
    item_idx = item_encoder.transform([item])[0]
    director_encoded = mlb_director.transform([director_list])[0]
    director_matrix[item_idx] = director_encoded

# 데이터 분할 (훈련: 90%, 검증: 10%) - 사용자별 분할
def split_user_interactions(interactions, test_size=0.1, random_state=42):
    train_list = []
    val_list = []
    for user, group in interactions.groupby('user'):
        if len(group) >= 2:
            train_group, val_group = train_test_split(group, test_size=test_size, random_state=random_state)
            train_list.append(train_group)
            val_list.append(val_group)
        else:
            train_list.append(group)
    train_split = pd.concat(train_list).reset_index(drop=True)
    val_split = pd.concat(val_list).reset_index(drop=True)
    return train_split, val_split

train_data, val_data = split_user_interactions(train_ratings, test_size=0.1, random_state=42)

print(f'Train data shape: {train_data.shape}, Validation data shape: {val_data.shape}')

# 사용자별 장르 및 감독 벡터 계산
user_genre_matrix = np.zeros((num_users, num_genres), dtype=np.float32)
user_director_matrix = np.zeros((num_users, num_directors), dtype=np.float32)

for user in range(num_users):
    user_items = train_data[train_data['user'] == user]['item'].tolist()
    if user_items:
        user_genre_matrix[user] = genre_matrix[user_items].mean(axis=0)
        user_director_matrix[user] = director_matrix[user_items].mean(axis=0)

# 데이터셋 정의: 사용자별 아이템 상호작용 벡터로 변경
class RecVaeDataset(Dataset):
    def __init__(self, interactions, num_items):
        self.num_items = num_items
        self.users = interactions['user'].unique()
        self.user_to_items = interactions.groupby('user')['item'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        items = self.user_to_items.get(user, set())
        item_vector = np.zeros(self.num_items, dtype=np.float32)
        item_vector[list(items)] = 1.0
        return user, item_vector

train_dataset = RecVaeDataset(train_data, num_items)
val_dataset = RecVaeDataset(val_data, num_items)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# RecVAE 모델 정의
class RecVAE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, genre_dim=64, director_dim=64):
        super(RecVAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 사용자 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 장르 및 감독 임베딩 (선형 변환)
        self.genre_embedding = nn.Linear(num_genres, genre_dim)
        self.director_embedding = nn.Linear(num_directors, director_dim)

        # 인코더: 사용자 임베딩 + 장르 임베딩 + 감독 임베딩
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim + genre_dim + director_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, embedding_dim)
        self.logvar_layer = nn.Linear(hidden_dim, embedding_dim)

        # 디코더: 잠재 벡터 + 장르 임베딩 + 감독 임베딩
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + genre_dim + director_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, user, item_vector, genre, director):
        user_emb = self.user_embedding(user)             # [batch_size, embedding_dim]
        genre_emb = self.genre_embedding(genre)          # [batch_size, genre_dim]
        director_emb = self.director_embedding(director)  # [batch_size, director_dim]

        # 인코더 입력: user_emb + genre_emb + director_emb
        encoder_input = torch.cat([user_emb, genre_emb, director_emb], dim=1)  # [batch_size, embedding_dim + genre_dim + director_dim]
        hidden = self.encoder(encoder_input)  # [batch_size, hidden_dim]
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        z = self.reparameterize(mu, logvar)

        # 디코더 입력: z + genre_emb + director_emb
        decoder_input = torch.cat([z, genre_emb, director_emb], dim=1)  # [batch_size, embedding_dim + genre_dim + director_dim]
        output = self.decoder(decoder_input)  # [batch_size, num_items]

        return output, mu, logvar

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Recall@10 계산 함수 정의 (1. 수정 사항 반영)
def calculate_recall_at_k(model, data_loader, user_genre_matrix, user_director_matrix, train_data, val_data, k=10):
    model.eval()
    total_recall = 0.0
    num_users_with_val = 0

    # 사용자별 훈련 아이템 집합 생성
    user_train_items = train_data.groupby('user')['item'].apply(set).to_dict()

    # 사용자별 검증 아이템 집합 생성
    user_val_items = val_data.groupby('user')['item'].apply(set).to_dict()

    # 모든 사용자를 포함하여 Recall@10 계산
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Recall@10"):
            user, item_vector = batch
            user = user.to(device)
            item_vector = item_vector.to(device)

            genre = torch.tensor(user_genre_matrix[user.cpu().numpy()]).to(device)
            director = torch.tensor(user_director_matrix[user.cpu().numpy()]).to(device)

            recon_batch, mu, logvar = model(user, item_vector, genre, director)
            scores = recon_batch.cpu().numpy()

            for idx, user_id in enumerate(user.cpu().numpy()):
                val_items = user_val_items.get(user_id, set())
                # 1. 모든 사용자를 포함하되, val_items가 없으면 Recall@10 = 0
                if len(val_items) == 0:
                    recall = 0.0
                else:
                    # 상위 k개 아이템 인덱스
                    topk_indices = np.argsort(scores[idx])[::-1][:k]
                    recommended = set(topk_indices)

                    # 훈련 데이터에 있는 아이템 제외
                    train_items = user_train_items.get(user_id, set())
                    recommended = recommended - train_items

                    # 상위 k개 아이템 재선정 (훈련 아이템 제외)
                    if len(recommended) < k:
                        additional = k - len(recommended)
                        scores_copy = scores[idx].copy()
                        scores_copy[list(train_items)] = -np.inf
                        scores_copy[list(recommended)] = -np.inf
                        additional_items = np.argsort(scores_copy)[::-1][:additional]
                        recommended.update(additional_items)

                    recommended = list(recommended)[:k]

                    # Recall@10 계산
                    num_correct = len(set(recommended) & val_items)
                    recall = num_correct / min(k, len(val_items))

                total_recall += recall
                num_users_with_val += 1

    avg_recall = total_recall / num_users_with_val if num_users_with_val > 0 else 0.0
    return avg_recall

# 모델 초기화 및 옵티마이저 설정
model = RecVAE(num_users, num_items, embedding_dim=64, hidden_dim=128, genre_dim=64, director_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 모델 학습
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        user, item_vector = batch
        user = user.to(device)
        item_vector = item_vector.to(device)

        # 메타데이터 로드 (장르, 감독)
        genre = torch.tensor(user_genre_matrix[user.cpu().numpy()]).to(device)
        director = torch.tensor(user_director_matrix[user.cpu().numpy()]).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(user, item_vector, genre, director)
        # 타겟은 전체 아이템 벡터 (multi-label)
        loss = loss_function(recon_batch, item_vector, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_train_loss:.4f}')

    # 검증 단계 및 Recall@10 계산 (1. 수정 사항 반영)
    recall = calculate_recall_at_k(model, val_loader, user_genre_matrix, user_director_matrix, train_data, val_data, k=10)
    print(f'====> Epoch: {epoch+1} Validation Recall@10: {recall:.4f}')

# ---------------------- 최종 모델 학습 및 제출 ----------------------

# 최종 모델을 전체 데이터로 재학습
print("Retraining the model on the entire dataset...")

# Combine train and validation data
full_train_data = pd.concat([train_data, val_data]).reset_index(drop=True)

# Recompute user_genre_matrix and user_director_matrix for full data
user_genre_matrix_full = np.zeros((num_users, num_genres), dtype=np.float32)
user_director_matrix_full = np.zeros((num_users, num_directors), dtype=np.float32)

for user in range(num_users):
    user_items = full_train_data[full_train_data['user'] == user]['item'].tolist()
    if user_items:
        user_genre_matrix_full[user] = genre_matrix[user_items].mean(axis=0)
        user_director_matrix_full[user] = director_matrix[user_items].mean(axis=0)

# Update dataset and dataloader
full_train_dataset = RecVaeDataset(full_train_data, num_items)
full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

# Reinitialize the model
model = RecVAE(num_users, num_items, embedding_dim=64, hidden_dim=128, genre_dim=64, director_dim=64)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Retrain the model
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(full_train_loader, desc=f'Final Training Epoch {epoch+1}/{num_epochs}'):
        user, item_vector = batch
        user = user.to(device)
        item_vector = item_vector.to(device)

        # 메타데이터 로드 (장르, 감독)
        genre = torch.tensor(user_genre_matrix_full[user.cpu().numpy()]).to(device)
        director = torch.tensor(user_director_matrix_full[user.cpu().numpy()]).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(user, item_vector, genre, director)
        # 타겟은 전체 아이템 벡터 (multi-label)
        loss = loss_function(recon_batch, item_vector, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(full_train_loader.dataset)
    print(f'====> Final Training Epoch: {epoch+1} Average loss: {avg_train_loss:.4f}')

# 추천 생성 및 제출 파일 작성 (3. 출력 파일 헤더 추가)
print("Generating recommendations for submission...")

model.eval()
user_ids = np.arange(num_users)
all_users = torch.tensor(user_ids).to(device)

with torch.no_grad():
    # 모든 사용자에 대해 아이템 점수 예측
    genre_full = torch.tensor(user_genre_matrix_full).to(device)
    director_full = torch.tensor(user_director_matrix_full).to(device)
    # 모델의 출력은 [num_users, num_items]
    scores, _, _ = model(all_users, torch.zeros(num_users, num_items).to(device), genre_full, director_full)
    scores = scores.cpu().numpy()

    # 사용자별 훈련 아이템 마스킹
    user_train_items_full = full_train_data.groupby('user')['item'].apply(set).to_dict()
    for user_idx in range(num_users):
        train_items = user_train_items_full.get(user_idx, set())
        if train_items:
            scores[user_idx, list(train_items)] = -np.inf  # 훈련 아이템의 점수를 -inf로 설정

    # 상위 k개 아이템 선택
    top_k = 10
    topk_items = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]  # [num_users, top_k]

# 사용자 원본 ID 및 아이템 원본 ID 매핑
user_original_ids = user_encoder.inverse_transform(user_ids)
item_original_ids = item_encoder.inverse_transform(np.arange(num_items))

# 제출 파일 작성 (3. 출력 파일 헤더 추가)
submission = []
for user_idx, items in enumerate(topk_items):
    user_id = user_original_ids[user_idx]
    recommended_items = item_original_ids[items]
    for item_id in recommended_items:
        submission.append([user_id, item_id])

submission_df = pd.DataFrame(submission, columns=['user', 'item'])  # 헤더 추가
submission_df.to_csv('RecVAE_submission.csv', index=False, header=True)  # 헤더 포함
print('Submission file saved as RecVAE_submission.csv')

# ---------------------- 최종 Recall@10 계산 ----------------------

print("Calculating Final Recall@10...")

# 사용자별 검증 아이템 집합 생성 (이 경우, 이미 전체 데이터를 사용했으므로 val_items는 없음)
# 대회에서는 테스트 데이터가 별도로 주어지므로, 여기서는 예시로만 계산합니다.
# 실제 대회에서는 별도의 테스트 데이터를 사용하여 Recall@10을 계산해야 합니다.

# 예시로 전체 데이터에 대해 Recall@10 계산 (실제 대회 환경과 다를 수 있음)
user_val_items_full = {}  # 실제 테스트 데이터가 있다면 해당 데이터를 사용

# Recall@10 초기화
total_recall = 0.0
num_users_with_val = 0

for user_idx in range(num_users):
    val_items = user_val_items_full.get(user_idx, set())
    if len(val_items) == 0:
        recall = 0.0  # Recall@10 = 0
    else:
        recommended = set(topk_items[user_idx])
        # 훈련 데이터에 있는 아이템 제외
        recommended = recommended - user_train_items_full.get(user_idx, set())
        num_correct = len(recommended & val_items)
        recall = num_correct / min(10, len(val_items))
    total_recall += recall
    num_users_with_val += 1

# 평균 Recall@10 계산
avg_recall = total_recall / num_users_with_val if num_users_with_val > 0 else 0.0
print(f'Final Recall@10: {avg_recall:.4f}')
