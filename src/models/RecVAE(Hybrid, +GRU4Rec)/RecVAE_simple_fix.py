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
    # 시점 기준으로 정렬 (가정: 'time' 컬럼이 존재)
    if 'time' in data.columns:
        data = data.sort_values('time')
        data.drop(columns=["time"], inplace=True)
    else:
        print("Warning: 'time' 컬럼이 존재하지 않습니다. 데이터를 시간 순으로 정렬하지 않습니다.")
    data["rating"] = 1.0  # 모든 상호작용을 이진화
    return data

def encode_users_items(data):
    encode_user = LabelEncoder()
    encode_item = LabelEncoder()
    # 원본 ID를 보존하기 위해 새로운 컬럼에 인코딩된 값을 저장
    data['user_encoded'] = encode_user.fit_transform(data["user"])
    data['item_encoded'] = encode_item.fit_transform(data["item"])
    return encode_user, encode_item

def create_csr_matrix(users, items, values, num_users, num_items):
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

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
        self.users = interactions['user_encoded'].unique()
        self.user_to_items = interactions.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        items = self.user_to_items.get(user, set())
        # 유효한 아이템 인덱스만 선택
        valid_items = [item for item in items if item < self.num_items]
        if len(valid_items) != len(items):
            print(f"Warning: Some items for user {user} are out of bounds and have been removed.")
        item_vector = np.zeros(self.num_items, dtype=np.float32)
        item_vector[valid_items] = 1.0
        return user, item_vector

# Recall@10 계산 함수 (검증 세트 기준)
def calculate_recall_at_k_val(model, val_loader, train_data, val_data, k=10):
    model.eval()
    # Ground truth items per user from val_data
    user_val_items = val_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
    # 사용자별 훈련 아이템 집합
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    total_recall = 0.0
    num_users = len(user_val_items)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating Recall@10"):
            user, _ = batch
            user = user.to(device)
            scores, _, _ = model(user)
            scores = scores.cpu().numpy()

            for idx, user_id in enumerate(user.cpu().numpy()):
                val_items = user_val_items.get(user_id, set())
                if not val_items:
                    continue  # If no ground truth, skip

                seen_items = user_train_items.get(user_id, set())
                scores_batch = scores[idx]
                if seen_items:
                    scores_batch[list(seen_items)] = -np.inf  # Mask seen items

                top_k_indices = np.argsort(scores_batch)[::-1][:k]
                recommended = set(top_k_indices)

                # Ground truth are already encoded
                num_correct = len(recommended & val_items)
                recall = num_correct / min(k, len(val_items))
                total_recall += recall

    avg_recall = total_recall / num_users if num_users > 0 else 0.0
    return avg_recall

# 메인 함수
if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"  # 실제 데이터 경로에 맞게 수정

    # 데이터 로드 및 전처리
    data = data_pre_for_ease(data_path)

    # 데이터 무결성 확인
    print("데이터 무결성 확인:")
    print(data.isnull().sum())
    print("\n데이터의 고유 사용자 수:", data['user'].nunique())
    print("데이터의 고유 아이템 수:", data['item'].nunique())

    # 전체 데이터에 대해 LabelEncoder 피팅
    encode_user, encode_item = encode_users_items(data)

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
        train_data['user_encoded'], train_data['item_encoded'], train_data['rating'], len(encode_user.classes_), len(encode_item.classes_)
    )
    X_val = create_csr_matrix(
        val_data['user_encoded'], val_data['item_encoded'], val_data['rating'], len(encode_user.classes_), len(encode_item.classes_)
    )

    # 데이터셋 및 데이터로더
    train_dataset = RecVaeDataset(train_data, len(encode_item.classes_))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    val_dataset = RecVaeDataset(val_data, len(encode_item.classes_))
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # 모델 초기화
    model = RecVAE(len(encode_user.classes_), len(encode_item.classes_), embedding_dim=64, hidden_dim=128)
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

        avg_train_loss = train_loss / len(encode_user.classes_)
        print(f'====> Epoch: {epoch+1} Average loss: {avg_train_loss:.4f}')

        # 검증 단계 및 Recall@10 계산
        recall = calculate_recall_at_k_val(model, val_loader, train_data, val_data, k=10)
        print(f'====> Epoch: {epoch+1} Validation Recall@10: {recall:.4f}')

    # 추천 생성 및 제출 파일 작성 (RecVAE)
    print("Generating recommendations for submission (RecVAE)...")

    model.eval()
    all_users = torch.arange(len(encode_user.classes_)).to(device)

    with torch.no_grad():
        # 모든 사용자에 대해 아이템 점수 예측
        scores, _, _ = model(all_users)
        scores = scores.cpu().numpy()

    # 사용자별 훈련 아이템 마스킹
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
    for user_idx in range(len(encode_user.classes_)):
        seen_items = user_train_items.get(user_idx, set())
        if seen_items:
            scores[user_idx, list(seen_items)] = -np.inf  # 이미 본 아이템 마스킹

    # 상위 k개 아이템 선택
    top_k = 10
    topk_items = np.argsort(scores, axis=1)[:, -top_k:][:, ::-1]  # [num_users, top_k]

    # 사용자 원본 ID 및 아이템 원본 ID 매핑
    user_original_ids = encode_user.inverse_transform(np.arange(len(encode_user.classes_)))
    item_original_ids = encode_item.inverse_transform(np.arange(len(encode_item.classes_)))

    # 제출 파일 작성 (출력 파일 헤더 추가)
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

    print("Calculating Final Recall@10 on validation set (RecVAE)...")

    # Recall@10 계산 함수 호출 (RecVAE)
    recall_val = calculate_recall_at_k_val(model, val_loader, train_data, val_data, k=10)
    print(f'Final Recall@10 on validation set (RecVAE): {recall_val:.4f}')
