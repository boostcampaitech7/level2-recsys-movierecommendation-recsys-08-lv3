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

# ---------------------- RecVAE Hybrid 모델 ----------------------

class RecVAE_Hybrid(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=128, seq_length=10):
        super(RecVAE_Hybrid, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        # 사용자 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # 시퀀스 임베딩 (예: 최근 seq_length 아이템)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
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

    def forward(self, user, item_seq):
        user_emb = self.user_embedding(user)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_length, embedding_dim]
        lstm_out, _ = self.lstm(item_emb)  # [batch_size, seq_length, hidden_dim]
        lstm_last = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # 사용자 임베딩과 LSTM 출력 결합
        combined = torch.cat([user_emb, lstm_last], dim=1)  # [batch_size, embedding_dim + hidden_dim]
        hidden = self.encoder(combined)  # [batch_size, hidden_dim]
        mu = self.mu_layer(hidden)  # [batch_size, embedding_dim]
        logvar = self.logvar_layer(hidden)  # [batch_size, embedding_dim]
        z = self.reparameterize(mu, logvar)  # [batch_size, embedding_dim]
        output = self.decoder(z)  # [batch_size, num_items]
        return output, mu, logvar

# ---------------------- GRU4Rec 모델 ----------------------

class GRU4Rec(nn.Module):
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

# ---------------------- 훈련 루프 ----------------------

# RecVAE Hybrid 훈련을 위한 데이터셋 정의
class RecVaeDataset_Hybrid(Dataset):
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
        # 정답 벡터 (이전에 본 아이템은 1, 나머지는 0)
        seen_items = set(items)
        item_vector = np.zeros(self.num_items, dtype=np.float32)
        item_vector[list(seen_items)] = 1.0
        item_vector = torch.tensor(item_vector, dtype=torch.float32)
        return user, item_seq, item_vector

# GRU4Rec 훈련을 위한 데이터셋 정의
class GRU4RecDataset(Dataset):
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

# 손실 함수 정의
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Recall@10 계산 함수 for RecVAE
def calculate_recall_at_k_val_recvvae(model, val_loader, train_data, val_data, k=10):
    model.eval()
    # Ground truth items per user from val_data
    user_val_items = val_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()
    # 사용자별 훈련 아이템 집합
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(set).to_dict()

    total_recall = 0.0
    num_users = len(user_val_items)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating Recall@10 for RecVAE"):
            user, item_seq, _ = batch
            user = user.to(device)
            item_seq = item_seq.to(device)
            recon_batch, _, _ = model(user, item_seq)
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

# Recall@10 계산 함수 for GRU4Rec
def calculate_recall_at_k_val_gru4rec(model, val_loader, train_data, val_data, k=10):
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

# 앙상블을 위한 함수 정의
def ensemble_recommendations(scores_recvae, scores_gru4rec, k=10, alpha=0.5):
    """
    RecVAE와 GRU4Rec의 점수를 앙상블하여 최종 점수를 계산.
    alpha: RecVAE의 가중치. (1 - alpha): GRU4Rec의 가중치.
    """
    final_scores = alpha * scores_recvae + (1 - alpha) * scores_gru4rec
    topk_items = np.argsort(final_scores, axis=1)[:, -k:][:, ::-1]  # [num_users, top_k]
    return topk_items

# ---------------------- 메인 함수 ----------------------

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

    # LabelEncoder 클래스 수 확인
    print(f'\nLabelEncoder has {len(encode_item.classes_)} classes.')
    print(f'Total unique items in data: {data["item"].nunique()}')

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

    # RecVAE Hybrid 데이터셋 및 데이터로더
    seq_length = 10  # 시퀀스 길이 설정
    train_dataset_recvae = RecVaeDataset_Hybrid(train_data, len(encode_item.classes_), seq_length)
    train_loader_recvae = DataLoader(train_dataset_recvae, batch_size=1024, shuffle=True)

    val_dataset_recvae = RecVaeDataset_Hybrid(val_data, len(encode_item.classes_), seq_length)
    val_loader_recvae = DataLoader(val_dataset_recvae, batch_size=1024, shuffle=False)

    # GRU4Rec 데이터셋 및 데이터로더
    train_dataset_gru = GRU4RecDataset(train_data, len(encode_item.classes_), seq_length)
    train_loader_gru = DataLoader(train_dataset_gru, batch_size=1024, shuffle=True)

    val_dataset_gru = GRU4RecDataset(val_data, len(encode_item.classes_), seq_length)
    val_loader_gru = DataLoader(val_dataset_gru, batch_size=1024, shuffle=False)

    # 모델 초기화
    model_recvae = RecVAE_Hybrid(len(encode_user.classes_), len(encode_item.classes_), embedding_dim=64, hidden_dim=128, seq_length=seq_length)
    model_gru4rec = GRU4Rec(len(encode_user.classes_), len(encode_item.classes_), embedding_dim=64, hidden_dim=128, num_layers=1, seq_length=seq_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_recvae = model_recvae.to(device)
    model_gru4rec = model_gru4rec.to(device)

    # 옵티마이저 정의
    optimizer_recvae = optim.Adam(model_recvae.parameters(), lr=1e-3)
    optimizer_gru4rec = optim.Adam(model_gru4rec.parameters(), lr=1e-3)

    # 훈련 설정
    num_epochs = 10

    # 훈련 루프
    for epoch in range(num_epochs):
        ############################
        # RecVAE Hybrid 모델 훈련
        ############################
        model_recvae.train()
        train_loss_recvae = 0
        for batch in tqdm(train_loader_recvae, desc=f'Epoch {epoch+1}/{num_epochs} - RecVAE Training'):
            user, item_seq, item_vector = batch
            user = user.to(device)
            item_seq = item_seq.to(device)
            item_vector = item_vector.to(device)

            optimizer_recvae.zero_grad()
            recon_batch, mu, logvar = model_recvae(user, item_seq)
            loss = loss_function(recon_batch, item_vector, mu, logvar)
            loss.backward()
            train_loss_recvae += loss.item()
            optimizer_recvae.step()

        avg_train_loss_recvae = train_loss_recvae / len(train_loader_recvae.dataset)
        print(f'====> Epoch: {epoch+1} RecVAE Average loss: {avg_train_loss_recvae:.4f}')

        # RecVAE 검증 단계 및 Recall@10 계산
        recall_recvae = calculate_recall_at_k_val_recvvae(model_recvae, val_loader_recvae, train_data, val_data, k=10)
        print(f'====> Epoch: {epoch+1} RecVAE Validation Recall@10: {recall_recvae:.4f}')

        ############################
        # GRU4Rec 모델 훈련
        ############################
        model_gru4rec.train()
        train_loss_gru4rec = 0
        for batch in tqdm(train_loader_gru, desc=f'Epoch {epoch+1}/{num_epochs} - GRU4Rec Training'):
            user, item_seq = batch
            user = user.to(device)
            item_seq = item_seq.to(device)

            optimizer_gru4rec.zero_grad()
            output = model_gru4rec(user, item_seq)
            # CrossEntropyLoss는 클래스가 0부터 num_items-1까지일 때 사용
            # 여기서는 BCE를 사용하지 않고 CrossEntropy를 사용
            # 정답 벡터 대신 마지막 아이템을 정답으로 설정
            # 또는 더 적합한 손실 함수를 선택할 수 있음
            # 여기서는 간단히 CrossEntropyLoss를 사용
            # 실제 구현에서는 다른 손실 함수 고려 필요
            loss_fn = nn.CrossEntropyLoss()
            # 타겟: 시퀀스의 다음 아이템 (예: 예측할 아이템)
            # 여기서는 시퀀스의 마지막 아이템을 타겟으로 설정 (간단한 예)
            target = item_seq[:, -1]
            loss = loss_fn(output, target)
            loss.backward()
            train_loss_gru4rec += loss.item()
            optimizer_gru4rec.step()

        avg_train_loss_gru4rec = train_loss_gru4rec / len(train_loader_gru.dataset)
        print(f'====> Epoch: {epoch+1} GRU4Rec Average loss: {avg_train_loss_gru4rec:.4f}')

        # GRU4Rec 검증 단계 및 Recall@10 계산
        recall_gru4rec = calculate_recall_at_k_val_gru4rec(model_gru4rec, val_loader_gru, train_data, val_data, k=10)
        print(f'====> Epoch: {epoch+1} GRU4Rec Validation Recall@10: {recall_gru4rec:.4f}')

    # ---------------------- 앙상블 및 추천 생성 ----------------------

    # 추천 생성 및 제출 파일 작성 (앙상블)
    print("Generating recommendations for submission (Ensemble of RecVAE Hybrid and GRU4Rec)...")

    model_recvae.eval()
    model_gru4rec.eval()
    all_users = torch.arange(len(encode_user.classes_)).to(device)

    # RecVAE Hybrid 모델을 위한 시퀀스 생성
    user_train_items = train_data.groupby('user_encoded')['item_encoded'].apply(list).to_dict()

    # 모든 사용자에 대해 시퀀스 데이터 생성 (훈련 세트 기준)
    user_sequences = []
    for user in range(len(encode_user.classes_)):
        items = user_train_items.get(user, [])
        if len(items) >= seq_length:
            seq = items[-seq_length:]
        else:
            seq = [0]*(seq_length - len(items)) + items
        user_sequences.append(seq)
    user_sequences = torch.tensor(user_sequences, dtype=torch.long).to(device)

    with torch.no_grad():
        # RecVAE Hybrid 예측
        scores_recvae, _, _ = model_recvae(all_users, user_sequences)
        scores_recvae = scores_recvae.cpu().numpy()

        # GRU4Rec 예측
        scores_gru4rec = model_gru4rec(all_users, user_sequences).cpu().numpy()

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
    submission_df.to_csv('Ensemble_RecVAE_GRU4Rec_submission_pos.csv', index=False, header=True)  # 헤더 포함
    print('Submission file saved as Ensemble_RecVAE_GRU4Rec_submission_pos.csv')

    # ---------------------- 최종 Recall@10 계산 ----------------------

    # RecVAE Final Recall
    print("Calculating Final Recall@10 on validation set (RecVAE)...")
    recall_val_recvae = calculate_recall_at_k_val_recvvae(model_recvae, val_loader_recvae, train_data, val_data, k=10)
    print(f'Final Recall@10 on validation set (RecVAE): {recall_val_recvae:.4f}')

    # GRU4Rec Final Recall
    print("Calculating Final Recall@10 on validation set (GRU4Rec)...")
    recall_val_gru4rec = calculate_recall_at_k_val_gru4rec(model_gru4rec, val_loader_gru, train_data, val_data, k=10)
    print(f'Final Recall@10 on validation set (GRU4Rec): {recall_val_gru4rec:.4f}')

    # 앙상블 Recall@10 (간단히 RecVAE와 GRU4Rec의 평균을 사용)
    print("Calculating Final Recall@10 on validation set (Ensemble)...")
    recall_val_ensemble = (recall_val_recvae + recall_val_gru4rec) / 2
    print(f'Final Recall@10 on validation set (Ensemble): {recall_val_ensemble:.4f}')
