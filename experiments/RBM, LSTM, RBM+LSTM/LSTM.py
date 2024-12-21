import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random

class RatingsDataset(Dataset):
    def __init__(self, ratings_file, sequence_length, item2idx=None, idx2item=None, is_train=True):
        self.sequence_length = sequence_length
        self.user_sequences = []
        self.is_train = is_train
        self.user_test_items = {}

        # CSV 파일 로드
        ratings = pd.read_csv(ratings_file)

        # 사용자별로 데이터 그룹화
        user_group = ratings.groupby('user')

        # 아이템 ID 매핑
        if item2idx is None or idx2item is None:
            self.all_items = ratings['item'].unique()
            self.num_items = len(self.all_items)
            # 아이템 ID를 인덱스로 매핑
            self.item2idx = {item: idx for idx, item in enumerate(self.all_items)}
            self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        else:
            self.item2idx = item2idx
            self.idx2item = idx2item
            self.all_items = list(self.item2idx.keys())
            self.num_items = len(self.all_items)

        # 각 사용자에 대해 아이템 시퀀스 생성
        self.user_histories = {}  # 각 사용자의 아이템 시퀀스 저장
        for user_id, group in user_group:
            # 타임스탬프 순으로 정렬
            group = group.sort_values('time')
            item_seq = group['item'].values
            self.user_histories[user_id] = [self.item2idx[item] for item in item_seq if item in self.item2idx]

            if len(item_seq) < sequence_length + 1:
                continue  # 시퀀스 길이가 부족한 경우 제외

            if self.is_train:
                # 슬라이딩 윈도우 적용하여 학습용 시퀀스 생성
                for i in range(len(item_seq) - sequence_length):
                    seq = item_seq[i:i+sequence_length]
                    self.user_sequences.append((user_id, seq))
            else:
                # 검증용 시퀀스 생성 (마지막 sequence_length 개의 아이템 사용)
                seq = item_seq[-sequence_length:]
                self.user_sequences.append((user_id, seq))
                self.user_test_items[user_id] = self.item2idx[item_seq[-1]]

        # 모든 아이템의 인덱스 리스트
        self.all_item_indices = list(range(self.num_items))

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        user_id, seq_items = self.user_sequences[idx]
        # 아이템 ID를 인덱스로 변환
        seq_idx = [self.item2idx[item] for item in seq_items if item in self.item2idx]
        # 시퀀스 데이터 (마지막 아이템은 Positive Item으로 사용)
        sequence_data = torch.tensor(seq_idx[:-1], dtype=torch.long)  # 마지막 아이템 제외
        pos_item = torch.tensor(seq_idx[-1], dtype=torch.long)  # Positive Item

        return user_id, sequence_data, pos_item

def collate_fn(batch):
    user_ids, sequences, pos_items = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    batch_size = len(pos_items)
    num_negatives = batch_size // 4  # Positive:Negative = 4:1

    # Negative Items 샘플링
    neg_items = []
    all_item_indices = set(range(sequences.size(1)))  # 모든 아이템 인덱스
    for _ in range(num_negatives):
        neg_item = random.choice(list(all_item_indices))
        neg_items.append(neg_item)
    neg_items = torch.tensor(neg_items, dtype=torch.long)

    return user_ids, sequences, pos_items, neg_items

def collate_fn_valid(batch):
    user_ids, sequences, pos_items = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    return user_ids, sequences, pos_items

class LSTM(nn.Module):
    def __init__(self, num_items, embedding_size, lstm_hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers

        # 아이템 임베딩 레이어
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # LSTM 모듈
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # 예측을 위한 선형 레이어
        self.output_layer = nn.Linear(lstm_hidden_size, embedding_size)

    def forward(self, sequence):
        # 시퀀스의 아이템 임베딩
        seq_embedded = self.item_embedding(sequence)  # [batch_size, seq_len, embedding_size]

        # LSTM 순전파
        lstm_out, _ = self.lstm(seq_embedded)  # lstm_out: [batch_size, seq_len, lstm_hidden_size]

        # 마지막 타임스텝의 출력 사용
        lstm_output = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]

        # 선형 레이어를 통해 아이템 임베딩 공간으로 매핑
        user_vector = self.output_layer(lstm_output)  # [batch_size, embedding_size]

        return user_vector  # 사용자 벡터 반환

    def predict(self, sequences):
        self.eval()
        device = next(self.parameters()).device
        sequences = sequences.to(device)
        with torch.no_grad():
            user_vector = self.forward(sequences)  # [batch_size, embedding_size]
            # 모든 아이템 임베딩 가져오기
            item_vectors = self.item_embedding.weight  # [num_items, embedding_size]
            # 사용자 벡터와 아이템 벡터 간의 점수 계산 (내적)
            scores = torch.matmul(user_vector, item_vectors.t())  # [batch_size, num_items]
        return scores  # 각 사용자에 대한 모든 아이템의 점수

    def train_model(self, train_loader, num_epochs, learning_rate):
        device = next(self.parameters()).device  # 모델의 디바이스 가져오기
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (user_ids, sequences, pos_items, neg_items) in enumerate(train_loader):
                    sequences = sequences.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = neg_items.to(device)

                    optimizer.zero_grad()
                    # 순전파
                    user_vector = self.forward(sequences)  # [batch_size, embedding_size]

                    # Positive Items 임베딩
                    pos_item_vector = self.item_embedding(pos_items)  # [batch_size, embedding_size]
                    # Negative Items 임베딩 (Negative Sample 수에 맞게 선택)
                    neg_item_vector = self.item_embedding(neg_items)  # [num_negatives, embedding_size]

                    # Positive Scores
                    pos_scores = torch.sum(user_vector[:neg_item_vector.size(0), :] * pos_item_vector[:neg_item_vector.size(0), :], dim=1)  # [num_negatives]

                    # Negative Scores
                    neg_scores = torch.sum(user_vector[:neg_item_vector.size(0), :] * neg_item_vector, dim=1)  # [num_negatives]

                    # BPR Loss 계산
                    loss = torch.mean(F.softplus(neg_scores - pos_scores))

                    # 역전파 및 가중치 업데이트
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    pbar.set_postfix({'Batch Loss': batch_loss})
                    pbar.update(1)
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

# 사용 예시

# 하이퍼파라미터 설정
data_path = "/data/ephemeral/home/KJPark/data/train/"
ratings_file = os.path.join(data_path, "train_ratings.csv")
sequence_length = 10    # 시퀀스 길이
num_epochs = 10
learning_rate = 1e-3
batch_size = 128

# 모델 하이퍼파라미터
embedding_size = 64
lstm_hidden_size = 128
num_layers = 1

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 학습 데이터셋 및 데이터로더 생성
train_dataset = RatingsDataset(ratings_file, sequence_length, is_train=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collate_fn
)

# 모델 초기화
model = LSTM(
    num_items=train_dataset.num_items,
    embedding_size=embedding_size,
    lstm_hidden_size=lstm_hidden_size,
    num_layers=num_layers
).to(device)

# 모델 학습
model.train_model(train_loader, num_epochs, learning_rate)

# 검증 데이터셋 및 데이터로더 생성
valid_dataset = RatingsDataset(
    ratings_file,
    sequence_length,
    item2idx=train_dataset.item2idx,
    idx2item=train_dataset.idx2item,
    is_train=False
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn_valid
)

# 예측 및 결과 저장
model.eval()
predictions = []
hit = 0  # Recall@10 계산을 위한 변수
total = 0

with torch.no_grad():
    with tqdm(total=len(valid_loader), desc="Predicting") as pbar:
        for user_ids, sequences, pos_items in valid_loader:
            sequences = sequences.to(device)
            user_ids = user_ids.numpy()
            pos_items = pos_items.numpy()
            # 모든 아이템에 대한 예측 점수 계산
            scores = model.predict(sequences)  # [batch_size, num_items]
            # 이미 본 아이템은 제외
            for idx, seq in enumerate(sequences):
                scores[idx, seq] = float('-inf')
            # 상위 K개의 아이템 추천
            topk = 10
            _, topk_indices = torch.topk(scores, topk, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            for idx, user_id in enumerate(user_ids):
                test_item = pos_items[idx]
                recommended_items = topk_indices[idx]
                if test_item in recommended_items:
                    hit += 1
                total += 1
                # 예측 결과 저장
                for item_idx in recommended_items:
                    item_id = valid_dataset.idx2item[item_idx]
                    predictions.append({
                        'user': user_id,
                        'item': item_id
                    })
            pbar.update(1)

# Recall@10 계산
recall_at_10 = hit / total if total > 0 else 0
print(f"Recall@10: {recall_at_10:.4f}")

# 결과를 DataFrame으로 변환
predictions_df = pd.DataFrame(predictions)

# CSV 파일로 저장
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "LSTM.csv")

try:
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
except Exception as e:
    print(f"Error saving predictions: {e}")
