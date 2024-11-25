import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 진행률 표시줄 라이브러리
import os

class RatingsDataset(Dataset):
    def __init__(self, ratings_file, sequence_length, item2idx=None, idx2item=None, is_train=True):
        self.sequence_length = sequence_length
        self.user_sequences = []
        self.is_train = is_train

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

        # 학습용 데이터셋과 검증용 데이터셋 생성
        self.user_test_items = {}  # 각 사용자의 검증용 아이템

        for user_id, group in user_group:
            # 타임스탬프 순으로 정렬
            group = group.sort_values('time')
            item_seq = group['item'].values

            if len(item_seq) < sequence_length + 1:
                continue  # 시퀀스 길이가 부족한 경우 제외

            # 마지막 아이템을 검증용으로 사용
            test_item = item_seq[-1]
            train_items = item_seq[:-1]

            if self.is_train:
                # 슬라이딩 윈도우 적용하여 학습용 시퀀스 생성
                for i in range(len(train_items) - sequence_length):
                    seq = train_items[i:i+sequence_length]
                    self.user_sequences.append((user_id, seq))
            else:
                # 검증용 시퀀스 생성 (마지막 sequence_length 개의 아이템 사용)
                seq = train_items[-sequence_length:]
                self.user_sequences.append((user_id, seq))
                self.user_test_items[user_id] = test_item

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        user_id, seq_items = self.user_sequences[idx]
        # 아이템 ID를 인덱스로 변환
        seq_idx = [self.item2idx[item] for item in seq_items if item in self.item2idx]
        # 원-핫 인코딩 벡터로 변환
        seq_vectors = np.zeros((len(seq_idx), self.num_items), dtype=np.float32)
        for i, item_idx in enumerate(seq_idx):
            seq_vectors[i, item_idx] = 1.0
        # 텐서로 변환
        sequence_data = torch.from_numpy(seq_vectors)
        return user_id, sequence_data

class LSTM_RBM(nn.Module):
    def __init__(self, visible_size, hidden_size, lstm_hidden_size, num_layers=1):
        super(LSTM_RBM, self).__init__()
        self.visible_size = visible_size  # 가시층 크기
        self.hidden_size = hidden_size    # 숨겨진 층 크기
        self.lstm_hidden_size = lstm_hidden_size  # LSTM 숨겨진 층 크기
        self.num_layers = num_layers      # LSTM 레이어 수

        # RBM 파라미터
        self.W = nn.Parameter(torch.randn(visible_size, hidden_size) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(visible_size))
        self.h_bias = nn.Parameter(torch.zeros(hidden_size))

        # LSTM 모듈
        self.lstm = nn.LSTM(input_size=visible_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # LSTM 출력에서 RBM 숨겨진 층으로의 연결
        self.lstm_to_hidden = nn.Linear(lstm_hidden_size, hidden_size)

    def sample_from_p(self, p):
        return torch.bernoulli(p)

    def visible_to_hidden(self, v, h_bias_adjusted):
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + h_bias_adjusted)
        h_sample = self.sample_from_p(h_prob)
        return h_prob, h_sample

    def hidden_to_visible(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_sample = self.sample_from_p(v_prob)
        return v_prob, v_sample

    def forward(self, input_sequence):
        # LSTM 순전파
        lstm_out, _ = self.lstm(input_sequence)
        lstm_features = lstm_out[:, -1, :]  # 마지막 타임스텝의 출력 사용

        # LSTM 출력으로 RBM 숨겨진 층 바이어스 조정
        h_bias_adjusted = self.h_bias + self.lstm_to_hidden(lstm_features)

        # 입력 시퀀스의 마지막 타임스텝 사용
        v = input_sequence[:, -1, :]

        # RBM 순전파
        h_prob, h_sample = self.visible_to_hidden(v, h_bias_adjusted)
        v_prob, v_sample = self.hidden_to_visible(h_sample)

        return v_prob

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W) + self.h_bias
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term

    def train_model(self, train_loader, num_epochs, learning_rate):
        device = next(self.parameters()).device  # 모델의 디바이스 가져오기
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            # tqdm을 사용하여 진행률 표시줄 추가
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (user_ids, data) in enumerate(train_loader):
                    data = data.float().to(device)
                    optimizer.zero_grad()
                    # 순전파
                    v_prob = self.forward(data)
                    # 손실 계산
                    loss = criterion(v_prob, data[:, -1, :])
                    # 역전파 및 가중치 업데이트
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item()
                    epoch_loss += batch_loss
                    # 진행률 표시줄 업데이트
                    pbar.set_postfix({'Batch Loss': batch_loss})
                    pbar.update(1)
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")

    def predict(self, input_sequence):
        self.eval()
        device = next(self.parameters()).device
        input_sequence = input_sequence.to(device)
        with torch.no_grad():
            v_prob = self.forward(input_sequence)
        return v_prob

# 사용 예시

# 하이퍼파라미터 설정
data_path = "/data/ephemeral/home/KJPark/data/train/"
ratings_file = os.path.join(data_path, "train_ratings.csv")
sequence_length = 10    # 시퀀스 길이
num_epochs = 5
learning_rate = 1e-3
batch_size = 128

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 학습 데이터셋 및 데이터로더 생성
train_dataset = RatingsDataset(ratings_file, sequence_length, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4)

# 모델 초기화
visible_size = train_dataset.num_items   # 아이템 수가 가시층 크기
hidden_size = 1000
lstm_hidden_size = 512
num_layers = 1

model = LSTM_RBM(visible_size, hidden_size, lstm_hidden_size, num_layers).to(device)

# 모델 학습
model.train_model(train_loader, num_epochs, learning_rate)

# 검증 데이터셋 및 데이터로더 생성
valid_dataset = RatingsDataset(ratings_file, sequence_length,
                               item2idx=train_dataset.item2idx,
                               idx2item=train_dataset.idx2item,
                               is_train=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor = 4)

# 예측 및 결과 저장
model.eval()
predictions = []
hit = 0  # Recall@10 계산을 위한 변수
total = 0

with torch.no_grad():
    with tqdm(total=len(valid_loader), desc="Predicting") as pbar:
        for user_ids, data in valid_loader:
            data = data.float().to(device)
            v_prob = model.predict(data)
            # 예측된 아이템 인덱스 가져오기 (상위 10개 추천)
            topk = 10  # 상위 10개 아이템 추천
            _, topk_indices = torch.topk(v_prob, topk, dim=1)
            topk_indices = topk_indices.cpu().numpy()
            user_ids = user_ids.numpy()
            for user_id, indices in zip(user_ids, topk_indices):
                test_item = valid_dataset.user_test_items[user_id]
                test_item_idx = train_dataset.item2idx.get(test_item, None)
                if test_item_idx is None:
                    continue  # 테스트 아이템이 훈련 데이터에 없을 경우 무시
                total += 1
                recommended_items = [idx for idx in indices if idx != test_item_idx]  # 테스트 아이템 제외
                if test_item_idx in indices:
                    hit += 1
                # 예측 결과 저장
                for idx in indices:
                    item_id = train_dataset.idx2item[idx]
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
output_file = "predictions.csv"
predictions_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
