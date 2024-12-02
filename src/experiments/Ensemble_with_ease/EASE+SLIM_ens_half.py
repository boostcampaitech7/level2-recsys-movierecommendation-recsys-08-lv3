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
    def __init__(self, num_items, _lambda, alpha, max_iter, device="cuda"):
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
def ensemble_and_generate_submission(scores_ease, scores_multivae, X, encode_user, encode_item):
    """
    Ensemble by selecting top 5 recommendations from EASE and top 5 from MultiVAE for each user.
    
    Parameters:
        scores_ease (np.ndarray): EASE prediction scores (users x items)
        scores_multivae (np.ndarray): MultiVAE prediction scores (users x items)
        X (csr_matrix): Interaction matrix (users x items)
        encode_user (LabelEncoder): Encoder for user IDs
        encode_item (LabelEncoder): Encoder for item IDs
    
    Returns:
        pd.DataFrame: Recommendations DataFrame with columns ["user", "item"]
    """
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]  # numpy.ndarray 형태로 변경

        # EASE: Top 5 아이템
        ease_scores = scores_ease[user_idx].copy()  # 원본 점수 보존을 위해 복사
        ease_scores[seen_items] = -np.inf  # 마스킹
        top_ease_idx = np.argsort(ease_scores)[-5:][::-1]
        top_ease_items = set(top_ease_idx)

        # MultiVAE: Top 5 아이템
        multivae_scores = scores_multivae[user_idx].copy()  # 원본 점수 보존을 위해 복사
        multivae_scores[seen_items] = -np.inf  # 마스킹
        top_multivae_idx = np.argsort(multivae_scores)[-5:][::-1]
        top_multivae_items = set(top_multivae_idx)

        # 앙상블: EASE와 MultiVAE의 상위 5개 아이템 결합
        combined_items = list(top_ease_items.union(top_multivae_items))

        # 만약 중복으로 인해 10개 미만일 경우, 추가로 상위 아이템을 선택
        if len(combined_items) < 10:
            # Combine EASE and MultiVAE scores
            combined_scores = ease_scores + multivae_scores
            # Mask seen items and already selected items
            mask = np.full_like(combined_scores, False, dtype=bool)
            mask[seen_items] = True
            mask[combined_items] = True
            combined_scores[mask] = -np.inf
            additional_needed = 10 - len(combined_items)
            if additional_needed > 0:
                additional_idx = np.argsort(combined_scores)[-additional_needed:][::-1]
                combined_items.extend(additional_idx.tolist())

        # 최종적으로 상위 10개 아이템 선택
        final_top_items = combined_items[:10]

        # 인코딩된 아이템 ID를 원래 ID로 변환
        top_items = encode_item.inverse_transform(final_top_items)
        user_id = encode_user.inverse_transform([user_idx])[0]

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
    _lambda = 600
    ease_model = EASE(_lambda)

    print("Training EASE model...")
    ease_model.train(X)
    print("Predicting with EASE model...")
    predict_result_ease = ease_model.predict(X)

    # ---------------------- SLIM 모델 학습 및 예측 ----------------------
    slim_model = SLIMGPU(num_items, _lambda=0.576, alpha=0.03, max_iter=190, device=device)
    print("Training SLIM model on GPU...")
    slim_model.train(X_torch)
    print("Predicting with SLIM model on GPU...")
    predict_result_slim = slim_model.predict(X_torch).cpu().numpy()

    # ---------------------- 앙상블 및 제출 파일 생성 ----------------------
    # Remove the alpha parameter if it's no longer needed
    print("Ensembling predictions and generating submission file...")
    recommendations_df = ensemble_and_generate_submission(
        predict_result_ease, predict_result_slim, X, encode_user, encode_item
    )

    # 제출 파일 생성
    recommendations_df.to_csv("ease_slim_ensemble.csv", index=False)
    print('Submission file saved as ease_slim_ensemble.csv')
