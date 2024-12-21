import pandas as pd
import numpy as np
import json
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
train_ratings = pd.read_csv(f'{DATA_PATH}train_ratings.csv')
with open(f'{DATA_PATH}Ml_item2attributes.json', 'r') as f:
    item2attributes = json.load(f)

titles = pd.read_csv(f'{DATA_PATH}titles.tsv', sep='\t')
years = pd.read_csv(f'{DATA_PATH}years.tsv', sep='\t')
directors = pd.read_csv(f'{DATA_PATH}directors.tsv', sep='\t')
genres = pd.read_csv(f'{DATA_PATH}genres.tsv', sep='\t')
writers = pd.read_csv(f'{DATA_PATH}writers.tsv', sep='\t')

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

# 데이터셋 정의
class RecVaeDataset(Dataset):
    def __init__(self, interactions):
        self.user = interactions['user'].values
        self.item = interactions['item'].values
        self.time = interactions['time'].values

        # 시간순으로 정렬
        sorted_indices = np.argsort(self.time)
        self.user = self.user[sorted_indices]
        self.item = self.item[sorted_indices]
        self.time = self.time[sorted_indices]

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.item[idx]

# 데이터 분할 (훈련: 90%, 검증: 10%)
train_data, val_data = train_test_split(train_ratings, test_size=0.1, random_state=42)

train_dataset = RecVaeDataset(train_data)
val_dataset = RecVaeDataset(val_data)

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

        # 사용자 및 아이템 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 장르 및 감독 임베딩 (선형 변환)
        self.genre_embedding = nn.Linear(num_genres, genre_dim)
        self.director_embedding = nn.Linear(num_directors, director_dim)

        # 인코더: 입력 크기를 256으로 수정 (user_emb + item_emb + genre_emb + director_emb)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + genre_dim + director_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, embedding_dim)
        self.logvar_layer = nn.Linear(hidden_dim, embedding_dim)

        # 디코더
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

    def forward(self, user, item, genre, director):
        user_emb = self.user_embedding(user)         # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item)         # [batch_size, embedding_dim]
        genre_emb = self.genre_embedding(genre)      # [batch_size, genre_dim]
        director_emb = self.director_embedding(director)  # [batch_size, director_dim]

        # 인코더 입력: user_emb + item_emb + genre_emb + director_emb
        encoder_input = torch.cat([user_emb, item_emb, genre_emb, director_emb], dim=1)  # [batch_size, 256]
        hidden = self.encoder(encoder_input)  # [batch_size, hidden_dim]
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        z = self.reparameterize(mu, logvar)

        # 디코더 입력: z + genre_emb + director_emb
        decoder_input = torch.cat([z, genre_emb, director_emb], dim=1)  # [batch_size, 192]
        output = self.decoder(decoder_input)  # [batch_size, num_items]

        return output, mu, logvar

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 모델 초기화 및 옵티마이저 설정
model = RecVAE(num_users, num_items, embedding_dim=64, hidden_dim=128, genre_dim=64, director_dim=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 모델 학습
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        user, item = batch
        user = user.to(device)
        item = item.to(device)

        # 메타데이터 로드 (장르, 감독)
        # 아이템 인덱스를 이용하여 genre_matrix와 director_matrix에서 해당하는 행을 추출
        genre = torch.tensor(genre_matrix[item.cpu().numpy()]).to(device)
        director = torch.tensor(director_matrix[item.cpu().numpy()]).to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(user, item, genre, director)
        # 타겟은 전체 아이템 벡터 (one-hot)
        target = torch.zeros_like(recon_batch).to(device)
        target.scatter_(1, item.unsqueeze(1), 1.0)
        loss = loss_function(recon_batch, target, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_train_loss:.4f}')

    # 검증 단계
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            user, item = batch
            user = user.to(device)
            item = item.to(device)

            genre = torch.tensor(genre_matrix[item.cpu().numpy()]).to(device)
            director = torch.tensor(director_matrix[item.cpu().numpy()]).to(device)

            recon_batch, mu, logvar = model(user, item, genre, director)
            target = torch.zeros_like(recon_batch).to(device)
            target.scatter_(1, item.unsqueeze(1), 1.0)
            loss = loss_function(recon_batch, target, mu, logvar)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f'====> Validation loss: {avg_val_loss:.4f}')

# 추천 생성 및 제출 파일 작성
model.eval()
user_ids = np.arange(num_users)
all_users = torch.tensor(user_ids).to(device)

with torch.no_grad():
    user_emb = model.user_embedding(all_users)
    item_emb = model.item_embedding.weight
    scores = torch.matmul(user_emb, item_emb.t())
    top_k = 10
    _, topk_items = torch.topk(scores, top_k, dim=1)

topk_items = topk_items.cpu().numpy()

# 사용자 원본 ID 및 아이템 원본 ID 매핑
user_original_ids = user_encoder.inverse_transform(user_ids)
item_original_ids = item_encoder.inverse_transform(np.arange(num_items))

# 제출 파일 작성
submission = []
for user_idx, items in enumerate(topk_items):
    user_id = user_original_ids[user_idx]
    recommended_items = item_original_ids[items]
    for item_id in recommended_items:
        submission.append([user_id, item_id])

submission_df = pd.DataFrame(submission, columns=['user', 'item']).drop_duplicates()
submission_df.to_csv('submission.csv', index=False, header=False)
print('Submission file saved as submission.csv')

# ---------------------- Recall@10 계산 추가 ----------------------

# 검증 데이터 (val_data)를 ground-truth으로 사용
# 각 사용자에 대한 실제 선호 아이템 집합 생성
user_val_items = val_data.groupby('user')['item'].apply(set).to_dict()

# Recall@10 초기화
total_recall = 0.0
num_users_with_val = 0

for user_idx in range(num_users):
    val_items = user_val_items.get(user_idx, set())
    if len(val_items) == 0:
        continue  # 해당 사용자의 검증 데이터가 없으면 건너뜁니다.

    recommended = set(topk_items[user_idx])
    num_correct = len(recommended & val_items)
    recall = num_correct / min(10, len(val_items))
    total_recall += recall
    num_users_with_val += 1

# 평균 Recall@10 계산
avg_recall = total_recall / num_users_with_val if num_users_with_val > 0 else 0.0
print(f'Recall@10: {avg_recall:.4f}')
