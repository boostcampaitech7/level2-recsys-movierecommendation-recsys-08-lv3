import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

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

# ---------------------- GPU 기반 SLIM 모델 정의 ----------------------
class SLIMGPU(nn.Module):
    def __init__(self, num_items, _lambda=0.1, alpha=0.01, max_iter=100, device="cuda"):
        """
        _lambda: L2 regularization term
        alpha: Learning rate
        max_iter: Maximum number of iterations
        """
        super(SLIMGPU, self).__init__()
        self._lambda = _lambda
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
            loss = (errors.square().sum() + self._lambda * self.W.square().sum()) / X.shape[0]

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

# ---------------------- 데이터 전처리 함수 ----------------------
def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    """
    PyTorch sparse tensor로 변환
    """
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))

# ---------------------- 앙상블 및 제출 파일 생성 함수 ----------------------
def ensemble_and_generate_submission_topk(scores_ease, scores_slim, X, encode_user, encode_item, alpha=0.7, top_k_each=15, final_k=10):
    """
    alpha: EASE 모델의 가중치 (0 ~ 1)
    (1 - alpha): SLIM 모델의 가중치
    top_k_each: 각 모델에서 뽑을 아이템의 개수
    final_k: 최종 추천 아이템의 개수
    """
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

        # 두 모델의 점수 가져오기
        scores_ease_user = scores_ease[user_idx]
        scores_slim_user = scores_slim[user_idx]

        # 각 모델에서 top_k_each 추출
        top_k_ease_idx = np.argsort(scores_ease_user)[-top_k_each:][::-1]
        top_k_slim_idx = np.argsort(scores_slim_user)[-top_k_each:][::-1]

        # 앙상블 대상 아이템 리스트 생성
        candidate_items = np.unique(np.concatenate([top_k_ease_idx, top_k_slim_idx]))

        # 앙상블 점수 계산
        final_scores = alpha * scores_ease_user[candidate_items] + (1 - alpha) * scores_slim_user[candidate_items]
        final_scores = np.where(np.isin(candidate_items, seen_items), -np.inf, final_scores)

        # 상위 final_k 아이템 선택
        top_items_idx = candidate_items[np.argsort(final_scores)[-final_k:][::-1]]
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
    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)
    num_users = len(encode_user.classes_)
    num_items = len(encode_item.classes_)

    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # CSR matrix 생성
    X = create_csr_matrix(users, items, interaction, num_users, num_items)

    # PyTorch sparse matrix로 변환
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_torch = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

    # ---------------------- EASE 모델 학습 및 예측 ----------------------
    _lambda = 450
    ease_model = EASE(_lambda)

    print("Training EASE model...")
    ease_model.train(X)
    print("Predicting with EASE model...")
    predict_result_ease = ease_model.predict(X)

    # ---------------------- SLIM 모델 학습 및 예측 ----------------------
    slim_model = SLIMGPU(num_items, _lambda=0.1, alpha=0.01, max_iter=100, device=device)
    print("Training SLIM model on GPU...")
    slim_model.train(X_torch)
    print("Predicting with SLIM model on GPU...")
    predict_result_slim = slim_model.predict(X_torch).cpu().numpy()

    # ---------------------- 앙상블 및 제출 파일 생성 ----------------------
    alpha = 0.7  # EASE 모델의 가중치, 필요에 따라 조정 가능
    print("Ensembling predictions and generating submission file...")
    recommendations_df = ensemble_and_generate_submission_topk(
        predict_result_ease, predict_result_slim, X, encode_user, encode_item, alpha=alpha
    )

    # 제출 파일 생성
    recommendations_df.to_csv("ease_slim_ensemble_gpu.csv", index=False)
    print('Submission file saved as ease_slim_ensemble_gpu.csv')
