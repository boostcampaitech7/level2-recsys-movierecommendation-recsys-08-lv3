import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------- EASE 모델 정의 ----------------------
class EASE:
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda

    def train(self, X):
        G = np.dot(X.T, X).toarray()  # G = X^T X
        diag_idx = np.arange(G.shape[0])  # diag index 구하기
        G[diag_idx, diag_idx] += self._lambda  # (X^T)X + (lambda)I
        P = np.linalg.inv(G)  # (X^T)X + (lambda)I 역행렬

        self.B = P / -np.diag(P)  # 최종 B
        self.B[diag_idx, diag_idx] = 0  # 대각 값 0

    def predict(self, X):
        return np.dot(X.toarray(), self.B)  # 예측 점수 계산

# ---------------------- 데이터 전처리 함수 ----------------------
def data_pre_for_ease(data_path):
    data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    data.drop(columns=["time"], inplace=True)
    data["rating"] = 1.0
    interaction = data["rating"].to_numpy()
    return data, interaction

def encode_users_items(data):
    encode_user = LabelEncoder()  # user encoder
    encode_item = LabelEncoder()  # item encoder
    users = encode_user.fit_transform(data["user"])
    items = encode_item.fit_transform(data["item"])
    return users, items, encode_user, encode_item

def create_csr_matrix(users, items, values, num_users, num_items):
    return csr_matrix((values, (users, items)), shape=(num_users, num_items))

# ---------------------- SASRec 모델 정의 ----------------------
class SASRecDataset(Dataset):
    def __init__(self, user_train_dict, num_items, max_seq_len):
        self.user_train_dict = user_train_dict
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.users = list(user_train_dict.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        items = self.user_train_dict[user]
        seq = items[-self.max_seq_len:]
        seq_len = len(seq)
        if seq_len < self.max_seq_len:
            padding_length = self.max_seq_len - seq_len
            seq = [0] * padding_length + list(seq)  # 패딩을 리스트로 처리
        else:
            seq = list(seq)
        return user, torch.tensor(seq, dtype=torch.long)

class SASRecModel(nn.Module):
    def __init__(self, num_users, num_items, max_seq_len, embed_dim, num_heads, num_layers, dropout_rate):
        super(SASRecModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)  # 아이템 인덱스는 1부터 시작, 0은 패딩
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.embed_dim = embed_dim

    def forward(self, input_seq):
        seq_embeddings = self.item_embedding(input_seq)  # [batch_size, max_seq_len, embed_dim]
        positions = torch.arange(self.max_seq_len, dtype=torch.long, device=input_seq.device).unsqueeze(0)
        pos_embeddings = self.position_embedding(positions)  # [1, max_seq_len, embed_dim]
        
        # Debugging: Print shapes to ensure they match
        # Uncomment the following lines if you need to debug
        # print(f"seq_embeddings shape: {seq_embeddings.shape}")
        # print(f"pos_embeddings shape: {pos_embeddings.shape}")
        
        seq_embeddings += pos_embeddings
        seq_embeddings = self.dropout(seq_embeddings.transpose(0, 1))  # [max_seq_len, batch_size, embed_dim]

        # attention_mask should be [batch_size, max_seq_len]
        attention_mask = (input_seq == 0)  # [batch_size, max_seq_len]
        for layer in self.layers:
            seq_embeddings = layer(seq_embeddings, src_key_padding_mask=attention_mask)

        output = seq_embeddings[-1]  # [batch_size, embed_dim]
        logits = torch.matmul(output, self.item_embedding.weight[1:].transpose(0, 1))  # [batch_size, num_items]
        return logits

# ---------------------- 앙상블 및 제출 파일 생성 함수 ----------------------
def ensemble_and_generate_submission(scores_ease, scores_sasrec, X, encode_user, encode_item, alpha=0.5):
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

        # 두 모델의 점수 가져오기
        scores_ease_user = scores_ease[user_idx]
        scores_sasrec_user = scores_sasrec[user_idx]

        # 앙상블 점수 계산 (가중 평균)
        final_scores = alpha * scores_ease_user + (1 - alpha) * scores_sasrec_user
        final_scores[seen_items] = -np.inf  # 이미 본 아이템 마스킹

        # 상위 10개 아이템 선택
        top_items_idx = np.argsort(final_scores)[-10:][::-1]
        top_items = encode_item.inverse_transform(top_items_idx)  # 원래 item id로
        user_id = encode_user.inverse_transform([user_idx])[0]  # 원래 user id로

        for item in top_items:
            result.append([user_id, item])

    recommendations_df = pd.DataFrame(result, columns=["user", "item"])
    return recommendations_df

# ---------------------- 메인 함수 ----------------------
if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"
    
    # EASE에 맞는 데이터 처리
    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)
    num_users = len(encode_user.classes_)
    num_items = len(encode_item.classes_)

    # CSR matrix 생성
    X = create_csr_matrix(users, items, interaction, num_users, num_items)

    # EASE 모델 학습
    _lambda = 450
    ease_model = EASE(_lambda)
    print("Training EASE model...")
    ease_model.train(X)
    print("Predicting with EASE model...")
    predict_result_ease = ease_model.predict(X)

    # 사용자별 시퀀스 데이터 생성 (SASRec용)
    # DeprecationWarning 해결: group_keys=False 추가 및 특정 컬럼만 선택
    user_train_dict = data.groupby('user')['item'].apply(list).to_dict()
    # EASE와 SASRec을 앙상블하기 위해 아이템 인덱스를 +1로 이동 (0은 패딩)
    user_train_dict_encoded = {
        encode_user.transform([k])[0]: (encode_item.transform(v) + 1).tolist() 
        for k, v in user_train_dict.items()
    }

    # SASRec 데이터셋 및 데이터로더 생성
    max_seq_len_dataset = 49  # 시퀀스 길이를 49로 설정하여 input_seq와 position_embedding의 길이를 일치시킴
    sasrec_dataset = SASRecDataset(user_train_dict_encoded, num_items, max_seq_len_dataset)
    sasrec_dataloader = DataLoader(sasrec_dataset, batch_size=256, shuffle=True)

    # SASRec 모델 초기화
    max_seq_len_model = max_seq_len_dataset - 1  # 모델의 max_seq_len을 input_seq의 길이와 일치시킴 (49 - 1 =48)
    embed_dim = 64
    num_heads = 2
    num_layers = 2
    dropout_rate = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sasrec_model = SASRecModel(num_users, num_items, max_seq_len_model, embed_dim, num_heads, num_layers, dropout_rate)
    sasrec_model = sasrec_model.to(device)
    optimizer = torch.optim.Adam(sasrec_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # SASRec 모델 학습
    num_epochs = 10
    print("Training SASRec model...")
    for epoch in range(num_epochs):
        sasrec_model.train()
        total_loss = 0
        for batch in tqdm(sasrec_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user_ids, sequences = batch
            user_ids = user_ids.to(device)
            sequences = sequences.to(device)
            optimizer.zero_grad()

            # 다음 아이템 예측을 위한 타겟 생성
            input_seq = sequences[:, :-1]  # [batch_size, max_seq_len_model] = [batch_size, 48]
            target_seq = sequences[:, -1]   # [batch_size]

            logits = sasrec_model(input_seq)
            loss = criterion(logits, target_seq - 1)  # 아이템 인덱스가 1부터 시작하므로 -1

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(sasrec_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # SASRec 예측
    print("Predicting with SASRec model...")
    sasrec_model.eval()
    all_user_scores_sasrec = np.zeros((num_users, num_items))
    with torch.no_grad():
        sasrec_dataset_full = SASRecDataset(user_train_dict_encoded, num_items, max_seq_len_dataset)
        sasrec_dataloader_full = DataLoader(sasrec_dataset_full, batch_size=256, shuffle=False)
        for batch in tqdm(sasrec_dataloader_full, desc="Generating SASRec predictions"):
            user_ids, sequences = batch
            user_ids = user_ids.to(device)
            sequences = sequences.to(device)

            input_seq = sequences[:, :-1]  # [batch_size, max_seq_len_model] = [batch_size,48]
            logits = sasrec_model(input_seq)  # [batch_size, num_items]
            logits = logits.cpu().numpy()
            for i, user_id in enumerate(user_ids.cpu().numpy()):
                all_user_scores_sasrec[user_id] = logits[i]

    # 두 모델의 예측 결과 앙상블 및 제출 파일 생성
    alpha = 0.5  # EASE 모델의 가중치
    print("Ensembling predictions and generating submission file...")
    recommendations_df = ensemble_and_generate_submission(
        predict_result_ease, all_user_scores_sasrec, X, encode_user, encode_item, alpha=alpha
    )

    # 제출 파일 생성
    recommendations_df.to_csv("ease_sasrec_ensemble_epoch10.csv", index=False)
    print('Submission file saved as ease_sasrec_ensemble_epoch10.csv')
