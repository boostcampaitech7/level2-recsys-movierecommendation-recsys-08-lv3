import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import random
from torch.cuda.amp import GradScaler, autocast
from functools import partial

# 데이터셋 클래스
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

# Collate function for training (Standard Negative Sampling)
def collate_fn(batch, num_items):
    user_ids, sequences, pos_items = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    batch_size = len(pos_items)
    num_negatives = batch_size  # Positive:Negative = 1:1

    # Negative Items 샘플링 (무작위)
    neg_items = torch.randint(0, num_items, (num_negatives,), dtype=torch.long)

    return user_ids, sequences, pos_items, neg_items

# Collate function for validation (No Negative Sampling)
def collate_fn_valid(batch):
    user_ids, sequences, pos_items = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    return user_ids, sequences, pos_items

# NARM 모델 클래스
class NARM(nn.Module):
    def __init__(self, num_items, embedding_size, lstm_hidden_size, num_layers=1, dropout=0.2):
        super(NARM, self).__init__()
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers

        # 아이템 임베딩 레이어
        self.item_embedding = nn.Embedding(num_items, embedding_size, padding_idx=0)

        # LSTM 모듈
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)

        # 어텐션 레이어
        self.attention = nn.Linear(lstm_hidden_size, 1)

        # 최종 사용자 벡터 생성
        self.output_layer = nn.Linear(lstm_hidden_size, embedding_size)

    def forward(self, sequence):
        # 시퀀스의 아이템 임베딩
        seq_embedded = self.item_embedding(sequence)  # [batch_size, seq_len, embedding_size]

        # LSTM 순전파
        lstm_out, _ = self.lstm(seq_embedded)  # lstm_out: [batch_size, seq_len, lstm_hidden_size]

        # 어텐션 점수 계산
        attn_scores = self.attention(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # 어텐션을 적용한 사용자 벡터 계산
        user_vector = torch.sum(lstm_out * attn_weights, dim=1)  # [batch_size, lstm_hidden_size]

        # 최종 사용자 벡터 매핑
        user_vector = self.output_layer(user_vector)  # [batch_size, embedding_size]

        return user_vector  # 사용자 벡터 반환

    def predict(self, sequences):
        self.eval()
        device = next(self.parameters()).device
        sequences = sequences.to(device)
        with torch.no_grad():
            user_vector = self.forward(sequences)  # [batch_size, embedding_size]
            # 모든 아이템 임베딩 가져오기
            item_vectors = self.item_embedding.weight  # [num_items, embedding_size]
            # 사용자 벡터와 아이템 임베딩 간의 점수 계산 (내적)
            scores = torch.matmul(user_vector, item_vectors.t())  # [batch_size, num_items]
        return scores  # 각 사용자에 대한 모든 아이템의 점수

    def hard_negative_sampling_batch(self, user_vector, pos_items, top_k=100):
        # 모든 아이템과의 점수 계산
        scores = torch.matmul(user_vector, self.item_embedding.weight.t())  # [batch_size, num_items]

        # Top-K 아이템 선택
        _, top_indices = torch.topk(scores, top_k, dim=1)  # [batch_size, top_k]

        # 각 사용자의 Positive Item과 Top-K 중 다른 아이템 선택
        pos_items_expanded = pos_items.unsqueeze(1).expand(-1, top_k)  # [batch_size, top_k]
        mask = top_indices != pos_items_expanded  # [batch_size, top_k]
        filtered_top_indices = top_indices.masked_fill(~mask, -1)  # [batch_size, top_k]

        # 무작위 아이템으로 대체 (-1인 곳에)
        rand_neg = torch.randint(0, self.num_items, (top_indices.size(0), 1), device=top_indices.device)
        filtered_top_indices = torch.where(filtered_top_indices != -1, filtered_top_indices, rand_neg)

        # 첫 번째 유효한 Hard Negative 선택
        hard_neg_items = filtered_top_indices[:, 0]  # [batch_size]

        return hard_neg_items  # [batch_size]

    def train_model(self, train_loader, num_epochs, learning_rate, device, hard_neg_ratio=0.2):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)  # Weight Decay 추가
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)  # 학습률 스케줄러 추가
        scaler = GradScaler()
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (user_ids, sequences, pos_items, neg_items) in enumerate(train_loader):
                    sequences = sequences.to(device, non_blocking=True)
                    pos_items = pos_items.to(device, non_blocking=True)
                    neg_items = neg_items.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with autocast():
                        # 순전파
                        user_vector = self.forward(sequences)  # [batch_size, embedding_size]

                        # Positive Items 임베딩
                        pos_item_vector = self.item_embedding(pos_items)  # [batch_size, embedding_size]

                        # Standard Negative Items 임베딩
                        neg_item_vector = self.item_embedding(neg_items)  # [batch_size, embedding_size]

                        # BPR Loss 계산 (Standard Negative Sampling)
                        pos_scores = torch.sum(user_vector * pos_item_vector, dim=1)  # [batch_size]
                        neg_scores = torch.sum(user_vector * neg_item_vector, dim=1)  # [batch_size]
                        loss = torch.mean(F.softplus(neg_scores - pos_scores))

                        # Hard Negative Sampling
                        if hard_neg_ratio > 0:
                            hard_neg_items = self.hard_negative_sampling_batch(user_vector, pos_items, top_k=100)  # [batch_size]
                            hard_neg_item_vector = self.item_embedding(hard_neg_items)  # [batch_size, embedding_size]
                            hard_neg_scores = torch.sum(user_vector * hard_neg_item_vector, dim=1)  # [batch_size]
                            hard_loss = torch.mean(F.softplus(hard_neg_scores - pos_scores))
                            loss += hard_loss  # 총 Loss에 추가

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})
                    pbar.update(1)
            scheduler.step()  # 학습률 스케줄러 업데이트
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

# 사용 예시

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    data_path = "/data/ephemeral/home/KJPark/data/train/"
    ratings_file = os.path.join(data_path, "train_ratings.csv")
    sequence_length = 10    # 시퀀스 길이
    num_epochs = 1
    learning_rate = 4e-3
    batch_size = 2048  # 배치 크기 대폭 증가 (2048에서 4096으로 증가)

    # 디바이스 설정 및 CuDNN 최적화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    # 학습 데이터셋 및 데이터로더 생성
    train_dataset = RatingsDataset(ratings_file, sequence_length, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # CPU 코어 수에 맞게 조절 (예: 8)
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # 미리 로드할 배치 수
        collate_fn=partial(collate_fn, num_items=train_dataset.num_items)
    )

    # 모델 초기화
    model = NARM(
        num_items=train_dataset.num_items,
        embedding_size=256,        # 임베딩 크기 설정
        lstm_hidden_size=512,      # LSTM 히든 사이즈 설정
        num_layers=2,              # LSTM 레이어 수 설정
        dropout=0.2                # 드롭아웃 비율 설정
    ).to(device)

    # 모델 학습
    model.train_model(train_loader, num_epochs, learning_rate, device, hard_neg_ratio=0.2)

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
        num_workers=4,  # 검증 시에는 Workers 수를 줄임 (예: 4)
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
    output_file = os.path.join(output_dir, "NARM_predictions.csv")

    try:
        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
