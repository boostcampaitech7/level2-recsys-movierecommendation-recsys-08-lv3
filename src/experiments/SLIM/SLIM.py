import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

# ---------------------- SLIM 모델 정의 ----------------------
class SLIMModel(nn.Module):
    def __init__(self, num_items, l1_reg=0.91, l2_reg=0.91, alpha=0.0002, max_iter=73, device="cuda"):  # 동일하게 lambda 값으로 l1, l2 튜닝
        """
        l1_reg: L1 regularization term
        l2_reg: L2 regularization term
        alpha: Learning rate
        max_iter: Maximum number of iterations
        """
        super(SLIMModel, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.max_iter = max_iter
        self.device = device

        # 학습할 가중치 행렬
        self.W = nn.Parameter(torch.zeros((num_items, num_items), device=self.device))
        self.num_items = num_items

    def train(self, X_sparse):
        """
        X_sparse: PyTorch sparse matrix (users x items) on GPU
        """
        # sparse matrix를 dense로 변환
        X = X_sparse.to_dense()  # [num_users, num_items]

        optimizer = torch.optim.SGD([self.W], lr=self.alpha)

        for _ in tqdm(range(self.max_iter), desc="Training SLIM on GPU"):
            optimizer.zero_grad()

            # 예측 및 손실 계산
            preds = torch.matmul(X, self.W)
            errors = X - preds
            loss = (
                errors.square().sum() +
                self.l1_reg * self.W.abs().sum() +
                self.l2_reg * self.W.square().sum()
            ) / X.shape[0]

            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

            # Self-loop 제거
            with torch.no_grad():
                self.W.fill_diagonal_(0)

    def predict(self, X_sparse):
        """
        X_sparse: PyTorch sparse matrix (users x items) on GPU
        Returns: Predicted scores (users x items) on GPU
        """
        X = X_sparse.to_dense()  # dense로 변환
        with torch.no_grad():
            return torch.matmul(X, self.W)

# ---------------------- 데이터 전처리 함수 ----------------------
def data_pre_for_slim(data_path):
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

def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    """
    PyTorch sparse tensor로 변환
    """
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))

# ---------------------- 추천 결과 생성 함수 ----------------------
def generate_submission(scores, X, encode_user, encode_item):
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

        # 점수 마스킹
        scores[user_idx, seen_items] = -np.inf

        # 상위 10개 아이템 선택
        top_items_idx = np.argsort(scores[user_idx])[-10:][::-1]
        top_items = encode_item.inverse_transform(top_items_idx)  # 원래 item id로
        user_id = encode_user.inverse_transform([user_idx])[0]  # 원래 user id로

        for item in top_items:
            result.append([user_id, item])

    recommendations_df = pd.DataFrame(result, columns=["user", "item"])
    return recommendations_df

# ---------------------- 메인 함수 ----------------------
if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"
    
    # 데이터 전처리
    data, interaction = data_pre_for_slim(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)
    num_users = len(encode_user.classes_)
    num_items = len(encode_item.classes_)

    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # CSR matrix 생성
    X = create_csr_matrix(users, items, interaction, num_users, num_items)

    # PyTorch sparse matrix로 변환
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_torch = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

    # ---------------------- SLIM 모델 학습 및 예측 ----------------------
    slim_model = SLIMModel(num_items, l1_reg=0.1, l2_reg=0.1, alpha=0.01, max_iter=100, device=device)
    print("Training SLIM model on GPU...")
    slim_model.train(X_torch)
    print("Predicting with SLIM model on GPU...")
    predict_result_slim = slim_model.predict(X_torch).cpu().numpy()

    # ---------------------- 제출 파일 생성 ----------------------
    print("Generating submission file...")
    recommendations_df = generate_submission(predict_result_slim, X, encode_user, encode_item)

    # 제출 파일 생성
    recommendations_df.to_csv("slim_gpu.csv", index=False)
    print('Submission file saved as slim_gpu.csv')
