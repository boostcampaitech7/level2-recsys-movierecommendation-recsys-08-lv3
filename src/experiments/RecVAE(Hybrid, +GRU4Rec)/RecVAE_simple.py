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

# 데이터 전처리 함수들
def data_pre_for_ease(data_path):
    data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    data.drop(columns=["time"], inplace=True)
    data["rating"] = 1.0  # 모든 상호작용을 이진화
    interaction = data["rating"].to_numpy()
    return data, interaction

def encode_users_items(data):
    encode_user = LabelEncoder()
    encode_item = LabelEncoder()
    users = encode_user.fit_transform(data["user"])
    items = encode_item.fit_transform(data["item"])
    return users, items, encode_user, encode_item

def create_csr_matrix(users, items, values):
    return csr_matrix((values, (users, items)))

# RecVAE 모델 정의
class RecVAE(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128):
        super(RecVAE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 사용자 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, user):
        user_emb = self.user_embedding(user)  # [batch_size, embedding_dim]
        hidden = self.encoder(user_emb)      # [batch_size, hidden_dim]
        mu = self.mu_layer(hidden)           # [batch_size, embedding_dim]
        logvar = self.logvar_layer(hidden)   # [batch_size, embedding_dim]
        z = self.reparameterize(mu, logvar) # [batch_size, embedding_dim]
        output = self.decoder(z)             # [batch_size, num_items]
        return output, mu, logvar

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 데이터셋 정의
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

# Recall@10 계산 함수
def calculate_recall_at_k(model, data_loader, X, k=10):
    model.eval()
    num_users = X.shape[0]
    user_train_items = {user: set(X[user].indices) for user in range(num_users)}

    total_recall = 0.0
    for batch in tqdm(data_loader, desc="Calculating Recall@10"):
        user, _ = batch
        user = user.to(device)
        scores, _, _ = model(user)
        scores = scores.cpu().numpy()

        for idx, user_id in enumerate(user.cpu().numpy()):
            seen_items = user_train_items[user_id]
            scores[user_id, list(seen_items)] = -np.inf  # 이미 본 아이템 마스킹
            top_k_items = np.argsort(scores[user_id])[::-1][:k]
            recommended = set(top_k_items)

            # 실제 검증 데이터가 없으므로, Recall@10을 0으로 설정
            # 실제 대회에서는 별도의 테스트 데이터 필요
            recall = 0.0  # 실제 Recall 계산은 테스트 데이터가 필요
            total_recall += recall

    avg_recall = total_recall / num_users
    return avg_recall

# 메인 함수
if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"

    # 데이터 로드 및 전처리
    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)

    # 인코딩된 값을 data DataFrame에 다시 할당
    data['user'] = users
    data['item'] = items

    # CSR matrix 생성
    X = create_csr_matrix(users, items, interaction)

    num_users = X.shape[0]
    num_items = X.shape[1]
    print(f'Number of users: {num_users}, Number of items: {num_items}')

    # 데이터셋 및 데이터로더
    train_dataset = RecVaeDataset(data, num_items)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # 모델 초기화
    model = RecVAE(num_users, num_items, embedding_dim=64, hidden_dim=128)
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

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(user)
            loss = loss_function(recon_batch, item_vector, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / num_users
        print(f'====> Epoch: {epoch+1} Average loss: {avg_train_loss:.4f}')

    # 추천 생성 및 제출 파일 작성
    print("Generating recommendations for submission...")

    model.eval()
    user_ids = np.arange(num_users)
    all_users = torch.tensor(user_ids).to(device)

    with torch.no_grad():
        # 모든 사용자에 대해 아이템 점수 예측
        scores, _, _ = model(all_users)
        scores = scores.cpu().numpy()

    for user_idx in range(num_users):
        seen_items = X[user_idx].indices
        scores[user_idx, seen_items] = -np.inf  # 이미 본 아이템 마스킹

    top_k = 10
    topk_items = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]

    # 사용자 원본 ID 및 아이템 원본 ID 매핑
    user_original_ids = encode_user.inverse_transform(user_ids)
    item_original_ids = encode_item.inverse_transform(np.arange(num_items))

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
