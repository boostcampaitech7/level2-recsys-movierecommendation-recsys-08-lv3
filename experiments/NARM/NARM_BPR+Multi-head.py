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
from torch.nn.utils.rnn import pad_sequence
import copy

# 데이터셋 클래스 (변경 없음)
class RatingsDataset(Dataset):
    def __init__(self, ratings, sequence_length, item2idx=None, idx2item=None, is_train=True, item_metadata=None):
        self.sequence_length = sequence_length
        self.user_sequences = []
        self.is_train = is_train
        self.user_histories = {}
        self.user_test_items = {}
        self.item_metadata = item_metadata  # 아이템 메타데이터
        
        # DataFrame 복사본 생성
        ratings = ratings.copy()
        
        # 아이템 ID 매핑
        if item2idx is None or idx2item is None:
            self.all_items = ratings['item'].unique()
            self.num_items = len(self.all_items)
            self.item2idx = {item: idx for idx, item in enumerate(self.all_items)}
            self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        else:
            self.item2idx = item2idx
            self.idx2item = idx2item
            self.all_items = list(self.item2idx.keys())
            self.num_items = len(self.all_items)
        
        # 아이템 인덱스로 변환
        ratings['item'] = ratings['item'].map(self.item2idx)
        
        # 사용자별로 데이터 그룹화
        user_group = ratings.groupby('user')
        
        for user_id, group in user_group:
            group = group.sort_values('time')
            item_seq = group['item'].values
            
            if user_id not in self.user_histories:
                self.user_histories[user_id] = []
            
            if self.is_train:
                # 훈련 데이터
                items = item_seq[:-2] if len(item_seq) >= 3 else item_seq[:-1]
                self.user_histories[user_id].extend(items.tolist())
                
                if len(items) < self.sequence_length:
                    continue  # 시퀀스 길이가 부족한 경우 제외
                
                # 슬라이딩 윈도우 적용하여 학습용 시퀀스 생성
                for i in range(len(items) - self.sequence_length + 1):
                    seq = items[i:i + self.sequence_length]
                    self.user_sequences.append((user_id, seq))
            else:
                # 검증 및 테스트 데이터
                # 마지막 두 개의 상호작용을 검증과 테스트로 사용
                if len(item_seq) >= 2:
                    train_items = item_seq[:-2]
                    valid_item = item_seq[-2]
                    test_item = item_seq[-1]
                    
                    # 훈련 히스토리에 추가
                    self.user_histories[user_id].extend(train_items.tolist())
                    
                    # 검증 및 테스트 시퀀스 추가
                    self.user_sequences.append((user_id, item_seq[:-1]))
                    
                    # Ground Truth 아이템 저장
                    if user_id not in self.user_test_items:
                        self.user_test_items[user_id] = []
                    self.user_test_items[user_id].append(test_item)
                elif len(item_seq) == 1:
                    # 상호작용이 하나뿐인 경우, 훈련에만 포함
                    self.user_histories[user_id].extend(item_seq.tolist())
    
    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, idx):
        user_id, seq_items = self.user_sequences[idx]
        # 시퀀스 데이터 (마지막 아이템은 Positive Item으로 사용)
        if len(seq_items) < 2:
            # 시퀀스 길이가 1인 경우, 패딩
            sequence = [0] * (self.sequence_length - len(seq_items)) + seq_items.tolist()
            sequence_data = torch.tensor(sequence[:-1], dtype=torch.long)
            pos_item = torch.tensor(sequence[-1], dtype=torch.long)
        else:
            # 시퀀스가 충분한 길이인지 확인하고 패딩
            if len(seq_items) >= self.sequence_length:
                sequence = seq_items[-self.sequence_length:].tolist()
            else:
                # 패딩 길이 계산
                padding_length = self.sequence_length - len(seq_items)
                # 앞쪽에 0을 추가하여 패딩
                sequence = [0] * padding_length + seq_items.tolist()
            sequence_data = torch.tensor(sequence[:-1], dtype=torch.long)  # 마지막 아이템 제외
            pos_item = torch.tensor(sequence[-1], dtype=torch.long)  # Positive Item

        # 아이템 메타데이터 가져오기
        if self.item_metadata is not None:
            meta_features = {}
            # 메타데이터 키와 최대 길이를 정의합니다.
            meta_keys = list(self.item_metadata[next(iter(self.item_metadata))].keys())
            max_meta_length = {
                'genres': 5,
                'writers': 5,
                'directors': 5,
                'years': 1  # 연도는 보통 하나만 있으므로 1로 설정
            }
            for key in meta_keys:
                meta_features[key] = []
                for item_id in sequence[:-1]:  # 시퀀스의 각 아이템에 대해 메타데이터 가져오기
                    features = self.item_metadata.get(item_id, {}).get(key, [])
                    # 최대 길이로 자르고, 부족하면 패딩
                    if len(features) > max_meta_length[key]:
                        features = features[:max_meta_length[key]]
                    else:
                        features = [0] * (max_meta_length[key] - len(features)) + features
                    meta_features[key].append(features)
            # Convert lists to tensors
            for key in meta_features:
                meta_features[key] = torch.tensor(meta_features[key], dtype=torch.long)
        else:
            meta_features = None

        return user_id, sequence_data, pos_item, meta_features

# Collate function for training (Standard Negative Sampling with multiple negatives)
def collate_fn(batch, num_items, num_negatives=3):
    user_ids, sequences, pos_items, meta_features = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    # 메타데이터 처리
    if meta_features[0] is not None:
        batch_meta = {}
        for key in meta_features[0].keys():
            metas = [meta[key] for meta in meta_features]
            # 패딩이 이미 되어 있으므로 torch.stack 사용
            metas_padded = torch.stack(metas, dim=0)
            batch_meta[key] = metas_padded
    else:
        batch_meta = None

    batch_size = len(pos_items)
    neg_items = []
    for _ in range(num_negatives):
        neg = torch.randint(0, num_items, (batch_size,), dtype=torch.long)
        neg_items.append(neg)
    neg_items = torch.stack(neg_items, dim=1)  # [batch_size, num_negatives]

    return user_ids, sequences, pos_items, neg_items, batch_meta

# Collate function for validation and test (No Negative Sampling)
def collate_fn_valid(batch):
    user_ids, sequences, pos_items, meta_features = zip(*batch)
    user_ids = torch.tensor(user_ids, dtype=torch.long)
    sequences = torch.stack(sequences)
    pos_items = torch.stack(pos_items)

    # 메타데이터 처리
    if meta_features[0] is not None:
        batch_meta = {}
        for key in meta_features[0].keys():
            metas = [meta[key] for meta in meta_features]
            # 패딩이 이미 되어 있으므로 torch.stack 사용
            metas_padded = torch.stack(metas, dim=0)
            batch_meta[key] = metas_padded
    else:
        batch_meta = None

    return user_ids, sequences, pos_items, batch_meta

# 예측 단계에서 모든 사용자 상호작용 아이템 제외 함수 (벡터화)
def exclude_user_history(scores, user_ids, user_history, num_items):
    # scores: [batch_size, num_items]
    # user_ids: [batch_size]
    # user_history: dict {user_id: [item1, item2, ...], ...}
    # num_items: int

    device = scores.device
    batch_size = scores.size(0)

    # 모든 사용자 아이템 히스토리 가져오기
    max_hist_len = max(len(user_history.get(user_id, [])) for user_id in user_ids)
    hist_matrix = torch.zeros((batch_size, max_hist_len), dtype=torch.long, device=device)

    for idx, user_id in enumerate(user_ids):
        items = user_history.get(user_id, [])
        if items:
            hist_len = len(items)
            hist_matrix[idx, :hist_len] = torch.tensor(items, dtype=torch.long, device=device)

    # 마스킹할 인덱스 생성
    mask = torch.zeros_like(scores, dtype=torch.bool, device=device)
    hist_mask = hist_matrix != 0  # 0이 아닌 곳이 히스토리 아이템
    hist_indices = hist_matrix[hist_mask]
    batch_indices = hist_mask.nonzero(as_tuple=True)[0]
    mask[batch_indices, hist_indices] = True

    # 점수 마스킹
    scores.masked_fill_(mask, float('-inf'))

    return scores

# BPR Loss 함수 정의
def bpr_loss(pos_scores, neg_scores):
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
    return loss

# NARM 모델 클래스 (Attention Mechanism을 Multi-head Self-Attention으로 변경)
class NARM(nn.Module):
    def __init__(self, num_items, embedding_size, lstm_hidden_size, num_layers=2, dropout=0.3,
                 num_genres=0, num_writers=0, num_directors=0, num_years=0, num_heads=8):
        super(NARM, self).__init__()
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads  # 추가된 부분

        # 아이템 임베딩 레이어
        self.item_embedding = nn.Embedding(num_items, embedding_size, padding_idx=0)

        # 메타데이터 임베딩 레이어
        self.genre_embedding = nn.Embedding(num_genres + 1, embedding_size, padding_idx=0) if num_genres > 0 else None
        self.writer_embedding = nn.Embedding(num_writers + 1, embedding_size, padding_idx=0) if num_writers > 0 else None
        self.director_embedding = nn.Embedding(num_directors + 1, embedding_size, padding_idx=0) if num_directors > 0 else None
        self.year_embedding = nn.Embedding(num_years + 1, embedding_size, padding_idx=0) if num_years > 0 else None

        # Bidirectional LSTM 모듈
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True,
                            bidirectional=True)

        # Multi-head Self-Attention 레이어로 변경
        self.multihead_attn = nn.MultiheadAttention(embed_dim=lstm_hidden_size * 2, num_heads=num_heads, dropout=dropout, batch_first=True)

        # 최종 사용자 벡터 생성
        self.output_layer = nn.Linear(lstm_hidden_size * 2, embedding_size)

    def forward(self, sequence, meta_features=None):
        # 시퀀스의 아이템 임베딩
        seq_embedded = self.item_embedding(sequence)  # [batch_size, seq_len, embedding_size]

        # 메타데이터 임베딩 처리
        if meta_features is not None:
            meta_embeddings = []
            if self.genre_embedding is not None and 'genres' in meta_features:
                genres = meta_features['genres']  # [batch_size, seq_len, max_genres]
                # 평균 임베딩
                genres_embedded = self.genre_embedding(genres)  # [batch_size, seq_len, max_genres, embedding_size]
                genres_embedded = genres_embedded.mean(dim=2)  # [batch_size, seq_len, embedding_size]
                meta_embeddings.append(genres_embedded)
            if self.writer_embedding is not None and 'writers' in meta_features:
                writers = meta_features['writers']  # [batch_size, seq_len, max_writers]
                writers_embedded = self.writer_embedding(writers)  # [batch_size, seq_len, max_writers, embedding_size]
                writers_embedded = writers_embedded.mean(dim=2)  # [batch_size, seq_len, embedding_size]
                meta_embeddings.append(writers_embedded)
            if self.director_embedding is not None and 'directors' in meta_features:
                directors = meta_features['directors']  # [batch_size, seq_len, max_directors]
                directors_embedded = self.director_embedding(directors)  # [batch_size, seq_len, max_directors, embedding_size]
                directors_embedded = directors_embedded.mean(dim=2)  # [batch_size, seq_len, embedding_size]
                meta_embeddings.append(directors_embedded)
            if self.year_embedding is not None and 'years' in meta_features:
                years = meta_features['years']  # [batch_size, seq_len, max_years]
                years_embedded = self.year_embedding(years)  # [batch_size, seq_len, max_years, embedding_size]
                years_embedded = years_embedded.mean(dim=2)  # [batch_size, seq_len, embedding_size]
                meta_embeddings.append(years_embedded)
            # 메타데이터 임베딩을 아이템 임베딩에 더함
            if meta_embeddings:
                meta_embeddings = torch.stack(meta_embeddings, dim=0).mean(dim=0)  # [batch_size, seq_len, embedding_size]
                seq_embedded += meta_embeddings

        # LSTM 순전파
        lstm_out, _ = self.lstm(seq_embedded)  # lstm_out: [batch_size, seq_len, lstm_hidden_size * 2]

        # Multi-head Self-Attention 적용
        attn_output, attn_output_weights = self.multihead_attn(
            lstm_out, lstm_out, lstm_out
        )  # 모두 lstm_out을 사용하여 Self-Attention 수행

        # 어텐션 출력을 평균하여 사용자 벡터 생성
        user_vector = attn_output.mean(dim=1)  # [batch_size, lstm_hidden_size * 2]

        # 최종 사용자 벡터 매핑
        user_vector = self.output_layer(user_vector)  # [batch_size, embedding_size]

        return user_vector  # 사용자 벡터 반환

    def predict(self, sequences, meta_features=None):
        self.eval()
        device = next(self.parameters()).device
        sequences = sequences.to(device)
        if meta_features is not None:
            for key in meta_features:
                meta_features[key] = meta_features[key].to(device)
        with torch.no_grad():
            user_vector = self.forward(sequences, meta_features)  # [batch_size, embedding_size]
            # 모든 아이템 임베딩 가져오기
            item_vectors = self.item_embedding.weight  # [num_items, embedding_size]
            # 사용자 벡터와 아이템 임베딩 간의 점수 계산 (내적)
            scores = torch.matmul(user_vector, item_vectors.t())  # [batch_size, num_items]
        return scores  # 각 사용자에 대한 모든 아이템의 점수

    def hard_negative_sampling_batch(self, user_vector, pos_items, top_k=100):
        # 모든 아이템과의 점수 계산
        scores = torch.matmul(user_vector, self.item_embedding.weight.t())  # [batch_size, num_items]

        # Top-K+1 아이템 선택 (Positive Item이 포함될 수 있으므로)
        _, top_indices = torch.topk(scores, top_k + 1, dim=1)  # [batch_size, top_k+1]

        # Positive Item을 제외한 마스킹 생성
        pos_items_expanded = pos_items.unsqueeze(1)  # [batch_size, 1]
        mask = top_indices != pos_items_expanded  # [batch_size, top_k+1]

        # Positive Item 위치를 num_items으로 대체 (유효하지 않은 인덱스)
        top_indices_filtered = top_indices.masked_fill(~mask, self.num_items)  # [batch_size, top_k+1]

        # 각 배치에서 첫 번째 유효한 하드 네거티브 아이템 선택
        hard_neg_items, _ = torch.min(top_indices_filtered, dim=1)  # [batch_size]

        # 하드 네거티브 아이템이 유효한지 확인하고, 유효하지 않은 경우 무작위 네거티브 샘플링
        invalid_mask = hard_neg_items == self.num_items
        if invalid_mask.any():
            random_negatives = torch.randint(0, self.num_items, (invalid_mask.sum().item(),), device=pos_items.device)
            hard_neg_items[invalid_mask] = random_negatives

        return hard_neg_items  # [batch_size]

    def train_model(self, train_loader, valid_loader, total_user_histories, user_test_items, num_epochs, learning_rate, device, hard_neg_ratio=0.3, early_stopping_patience=5):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        scaler = GradScaler()
        self.train()

        best_recall = 0
        patience_counter = 0
        best_model_state = copy.deepcopy(self.state_dict())

        for epoch in range(num_epochs):
            epoch_loss = 0
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (user_ids, sequences, pos_items, neg_items, meta_features) in enumerate(train_loader):
                    sequences = sequences.to(device, non_blocking=True)
                    pos_items = pos_items.to(device, non_blocking=True)
                    neg_items = neg_items.to(device, non_blocking=True)

                    # 메타데이터를 GPU로 이동
                    if meta_features is not None:
                        for key in meta_features:
                            meta_features[key] = meta_features[key].to(device, non_blocking=True)

                    optimizer.zero_grad()
                    with autocast():
                        # 순전파
                        user_vector = self.forward(sequences, meta_features)  # [batch_size, embedding_size]

                        # Positive Items 임베딩
                        pos_item_vector = self.item_embedding(pos_items)  # [batch_size, embedding_size]

                        # Standard Negative Items 임베딩
                        # neg_items: [batch_size, num_negatives]
                        neg_item_vectors = self.item_embedding(neg_items)  # [batch_size, num_negatives, embedding_size]

                        # BPR Loss 계산 (Standard Negative Sampling)
                        pos_scores = torch.sum(user_vector * pos_item_vector, dim=1, keepdim=True)  # [batch_size, 1]
                        neg_scores = torch.bmm(neg_item_vectors, user_vector.unsqueeze(2)).squeeze(2)  # [batch_size, num_negatives]
                        loss = bpr_loss(pos_scores, neg_scores)

                        # Hard Negative Sampling
                        if hard_neg_ratio > 0:
                            hard_neg_items = self.hard_negative_sampling_batch(user_vector, pos_items, top_k=100)  # [batch_size]
                            hard_neg_item_vector = self.item_embedding(hard_neg_items)  # [batch_size, embedding_size]
                            hard_neg_scores = torch.sum(user_vector * hard_neg_item_vector, dim=1)  # [batch_size]
                            hard_loss = bpr_loss(pos_scores.squeeze(1), hard_neg_scores)
                            loss += hard_loss * hard_neg_ratio  # 총 Loss에 추가

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})
                    pbar.update(1)

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

            # 검증 성능 평가
            recall_at_10 = evaluate_model(self, valid_loader, total_user_histories, user_test_items, K=10)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Recall@10: {recall_at_10:.4f}")

            # 모델을 다시 학습 모드로 전환
            self.train()

            # 스케줄러 업데이트
            scheduler.step(recall_at_10)

            # Early Stopping 체크
            if recall_at_10 > best_recall:
                best_recall = recall_at_10
                patience_counter = 0
                # 최상의 모델 저장
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        # 최상의 모델 로드
        self.load_state_dict(best_model_state)
        print(f"Best Validation Recall@10: {best_recall:.4f}")

# 평가 함수 (변경 없음)
def evaluate_model(model, data_loader, user_history, user_test_items, K=10):
    model.eval()
    total_recall = 0.0
    total_users = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating") as pbar:
            for user_ids, sequences, _, meta_features in data_loader:
                sequences = sequences.to(device)
                user_ids = user_ids.numpy()
                if meta_features is not None:
                    for key in meta_features:
                        meta_features[key] = meta_features[key].to(device)
                # 모든 아이템에 대한 예측 점수 계산
                scores = model.predict(sequences, meta_features)  # [batch_size, num_items]
                # 이미 본 아이템은 제외 (훈련 및 검증 데이터의 히스토리 모두)
                user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
                scores = exclude_user_history(scores, user_ids_tensor, user_history, model.num_items)
                # 상위 K*2개의 아이템 추천 (중복 제거)
                _, topk_indices = torch.topk(scores, K * 2, dim=1)
                topk_indices = topk_indices.cpu().numpy()
                for idx, user_id in enumerate(user_ids):
                    recommended_items = []
                    for item_idx in topk_indices[idx]:
                        if item_idx not in recommended_items:
                            recommended_items.append(item_idx)
                        if len(recommended_items) == K:
                            break
                    ground_truth_items = user_test_items.get(user_id, [])
                    num_relevant_items = len(ground_truth_items)
                    if num_relevant_items == 0:
                        continue  # 해당 사용자의 Ground Truth가 없으면 건너뜁니다.
                    num_hits = len(set(recommended_items) & set(ground_truth_items))
                    recall = num_hits / min(K, len(ground_truth_items))
                    total_recall += recall
                    total_users += 1
                pbar.update(1)
    recall_at_k = total_recall / total_users if total_users > 0 else 0
    return recall_at_k

# 추천 목록을 생성하고 저장하는 함수 (변경 없음)
def generate_recommendations(model, data_loader, user_history, all_test_users, K=10):
    model.eval()
    recommendations = {user: [] for user in all_test_users}  # 모든 테스트 사용자를 초기화
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Generating Recommendations") as pbar:
            for user_ids, sequences, _, meta_features in data_loader:
                sequences = sequences.to(device)
                user_ids = user_ids.numpy()
                if meta_features is not None:
                    for key in meta_features:
                        meta_features[key] = meta_features[key].to(device)
                # 모든 아이템에 대한 예측 점수 계산
                scores = model.predict(sequences, meta_features)  # [batch_size, num_items]
                # 이미 본 아이템은 제외 (훈련 및 검증 데이터의 히스토리 모두)
                user_ids_tensor = torch.tensor(user_ids, dtype=torch.long)
                scores = exclude_user_history(scores, user_ids_tensor, user_history, model.num_items)
                # 상위 K*2개의 아이템 추천 (중복 제거)
                _, topk_indices = torch.topk(scores, K * 2, dim=1)
                topk_indices = topk_indices.cpu().numpy()
                for idx, user_id in enumerate(user_ids):
                    recommended_items = []
                    for item_idx in topk_indices[idx]:
                        if item_idx not in recommended_items:
                            recommended_items.append(item_idx)
                        if len(recommended_items) == K:
                            break
                    # 아이템 ID로 변환하여 추가
                    for item_idx in recommended_items:
                        item_id = test_dataset.idx2item[item_idx]
                        recommendations[user_id].append(item_id)
                pbar.update(1)
    return recommendations

# 사용 예시

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    data_path = "/data/ephemeral/home/KJPark/data/train/"
    ratings_file = os.path.join(data_path, "train_ratings.csv")
    sequence_length = 10    # 시퀀스 길이
    num_epochs = 20         # 에포크 수 증가
    learning_rate = 1e-3
    batch_size = 512
    num_negatives = 3       # Negative Sampling 비율 증가
    early_stopping_patience = 5  # Early Stopping 인내심

    # (1) 데이터 로드 및 전처리
    ratings = pd.read_csv(ratings_file)

    # 시간순으로 정렬
    ratings = ratings.sort_values('time')

    # 사용자별로 데이터 분할
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    train_data = []
    valid_data = []
    test_data = []

    user_group = ratings.groupby('user')

    for user_id, group in user_group:
        group = group.sort_values('time')
        item_seq = group['item'].values
        n_interactions = len(item_seq)
        
        if n_interactions >= 3:
            train_end = int(n_interactions * train_ratio)
            valid_end = int(n_interactions * (train_ratio + valid_ratio))
            
            train_items = item_seq[:train_end]
            valid_item = item_seq[train_end:valid_end]
            test_item = item_seq[valid_end:]
            
            train_data.append(pd.DataFrame({'user': [user_id]*len(train_items),
                                            'item': train_items,
                                            'time': group['time'].iloc[:train_end]}))
            
            valid_data.append(pd.DataFrame({'user': [user_id]*len(valid_item),
                                            'item': valid_item,
                                            'time': group['time'].iloc[train_end:valid_end]}))
            
            test_data.append(pd.DataFrame({'user': [user_id]*len(test_item),
                                           'item': test_item,
                                           'time': group['time'].iloc[valid_end:]}))
        elif n_interactions == 2:
            # 상호작용이 두 개인 경우, 하나는 훈련, 하나는 테스트
            train_items = item_seq[:1]
            test_item = item_seq[1:]
            
            train_data.append(pd.DataFrame({'user': [user_id],
                                            'item': train_items,
                                            'time': group['time'].iloc[:1]}))
            
            test_data.append(pd.DataFrame({'user': [user_id],
                                           'item': test_item,
                                           'time': group['time'].iloc[1:]}))
        else:
            # 상호작용이 하나뿐인 경우, 훈련에만 포함
            train_data.append(pd.DataFrame({'user': [user_id],
                                            'item': item_seq,
                                            'time': group['time']}))

    # 데이터프레임 합치기
    train_data = pd.concat(train_data, ignore_index=True)
    valid_data = pd.concat(valid_data, ignore_index=True)
    test_data = pd.concat(test_data, ignore_index=True)

    # 아이템 ID 매핑
    all_items = ratings['item'].unique()
    item2idx = {item: idx for idx, item in enumerate(all_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    num_items = len(all_items)

    # 메타데이터 로드 및 처리
    # genres.tsv 로드
    genres_file = os.path.join(data_path, "genres.tsv")
    genres_df = pd.read_csv(genres_file, sep='\t')
    genres_df['genre_id'], genre_unique = pd.factorize(genres_df['genre'])
    num_genres = len(genre_unique)

    # writers.tsv 로드
    writers_file = os.path.join(data_path, "writers.tsv")
    writers_df = pd.read_csv(writers_file, sep='\t')
    writers_df['writer_id'], writer_unique = pd.factorize(writers_df['writer'])
    num_writers = len(writer_unique)

    # directors.tsv 로드
    directors_file = os.path.join(data_path, "directors.tsv")
    directors_df = pd.read_csv(directors_file, sep='\t')
    directors_df['director_id'], director_unique = pd.factorize(directors_df['director'])
    num_directors = len(director_unique)

    # years.tsv 로드
    years_file = os.path.join(data_path, "years.tsv")
    years_df = pd.read_csv(years_file, sep='\t')
    years_df['year_id'], year_unique = pd.factorize(years_df['year'])
    num_years = len(year_unique)

    # 아이템별 메타데이터 생성
    item_metadata = {}

    for item in all_items:
        item_idx = item2idx[item]
        meta = {}
        # 장르
        genres = genres_df[genres_df['item'] == item]['genre_id'].tolist()
        meta['genres'] = genres if genres else [0]
        # 작가
        writers = writers_df[writers_df['item'] == item]['writer_id'].tolist()
        meta['writers'] = writers if writers else [0]
        # 감독
        directors = directors_df[directors_df['item'] == item]['director_id'].tolist()
        meta['directors'] = directors if directors else [0]
        # 연도
        years = years_df[years_df['item'] == item]['year_id'].tolist()
        meta['years'] = years if years else [0]
        item_metadata[item_idx] = meta

    # (4) 디바이스 설정 및 CuDNN 최적화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    # (5) 학습 및 검증 데이터셋 생성
    train_dataset = RatingsDataset(train_data, sequence_length, item2idx=item2idx, idx2item=idx2item, is_train=True, item_metadata=item_metadata)
    valid_dataset = RatingsDataset(valid_data, sequence_length, item2idx=item2idx, idx2item=idx2item, is_train=False, item_metadata=item_metadata)
    test_dataset = RatingsDataset(test_data, sequence_length, item2idx=item2idx, idx2item=idx2item, is_train=False, item_metadata=item_metadata)

    # 훈련 데이터셋의 아이템 인덱스 유효성 검증
    if train_dataset.user_histories:
        max_index = max([max(items) if items else -1 for items in train_dataset.user_histories.values()])
        print(f"Max item index in user_histories: {max_index}, num_items: {train_dataset.num_items}")
        assert max_index < train_dataset.num_items, "Found item index >= num_items in user_histories"
    else:
        print("No user_histories found in training dataset.")

    # (6) 사용자 히스토리 및 Ground Truth 정의
    # 평가를 위한 사용자 히스토리 생성 (훈련 + 검증 데이터)
    total_user_histories = train_dataset.user_histories.copy()
    for user_id, items in valid_dataset.user_histories.items():
        if user_id in total_user_histories:
            total_user_histories[user_id].extend(items)
        else:
            total_user_histories[user_id] = items

    # 사용자별 Ground Truth 아이템 생성 (검증 데이터)
    user_test_items = valid_dataset.user_test_items

    # (7) 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # CPU 코어 수에 맞게 조절
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=partial(collate_fn, num_items=train_dataset.num_items, num_negatives=num_negatives)
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # 검증 시에도 워커 수를 적절히 설정
        pin_memory=True,
        collate_fn=collate_fn_valid
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,  # 테스트 시에도 워커 수를 적절히 설정
        pin_memory=True,
        collate_fn=collate_fn_valid
    )

    # (8) 모델 초기화
    model = NARM(
        num_items=train_dataset.num_items,
        embedding_size=512,        # 임베딩 크기 증가
        lstm_hidden_size=1024,     # LSTM 히든 사이즈 증가
        num_layers=2,              # LSTM 레이어 수
        dropout=0.3,               # Dropout 비율 증가
        num_genres=num_genres,
        num_writers=num_writers,
        num_directors=num_directors,
        num_years=num_years,
        num_heads=8                # 추가된 num_heads 파라미터
    ).to(device)

    # (9) 모델 학습
    model.train_model(
        train_loader=train_loader,
        valid_loader=valid_loader,
        total_user_histories=total_user_histories,
        user_test_items=user_test_items,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        hard_neg_ratio=0.3,
        early_stopping_patience=early_stopping_patience
    )

    # (10) 테스트 데이터에 대한 추천 목록 생성 및 저장
    # 테스트 데이터의 사용자 히스토리 생성 (훈련 + 검증 + 테스트 데이터)
    for user_id, items in test_dataset.user_histories.items():
        if user_id in total_user_histories:
            total_user_histories[user_id].extend(items)
        else:
            total_user_histories[user_id] = items

    # 모든 테스트 사용자 목록 가져오기
    all_test_users = list(test_dataset.user_histories.keys())

    # 테스트 데이터 로더는 이미 만들어져 있으므로, 추천 목록 생성
    recommendations = generate_recommendations(model, test_loader, total_user_histories, all_test_users, K=10)

    # 추천 생성 후 모든 사용자가 포함되었는지 확인
    missing_users = set(all_test_users) - set(recommendations.keys())
    if missing_users:
        print(f"Missing users in recommendations: {missing_users}")
    else:
        print("All test users have recommendations.")

    # 특정 사용자 (예: 사용자 11번) 확인
    user_id = 11
    if user_id in recommendations:
        print(f"Recommendations for user {user_id}: {recommendations[user_id]}")
    else:
        print(f"User {user_id} is missing in recommendations.")

    # 추천 목록을 DataFrame으로 변환
    # 대회 제출 형식에 맞게 변환 (예: user, item)
    submission = []
    for user_id, items in recommendations.items():
        for item in items[:10]:  # 상위 10개 아이템
            submission.append({
                'user': user_id,
                'item': item
            })

    submission_df = pd.DataFrame(submission)

    # CSV 파일로 저장
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "NARM_predictions.csv")

    try:
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
