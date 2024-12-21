import os
import numpy as np
import pandas as pd
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm


# ---------------------- 평가 함수 ----------------------
def evaluate_model(recommendations_df, ground_truth_df):
    """
    recommendations_df: 추천 결과 DataFrame (columns: ["user", "item"])
    ground_truth_df: Ground Truth DataFrame (columns: ["user", "item"])
    
    Returns:
        평균 Recall@10 점수
    """
    # 유저별 추천 아이템 리스트
    recommended_items = (
        recommendations_df.groupby("user")["item"].apply(list).to_dict()
    )

    # 유저별 Ground Truth 아이템 리스트
    ground_truth_items = (
        ground_truth_df.groupby("user")["item"].apply(set).to_dict()
    )

    recalls = []

    for user, rec_items in recommended_items.items():
        if user in ground_truth_items:
            gt_items = ground_truth_items[user]  # Ground Truth 아이템
            num_relevant_items = len(gt_items)  # |I_u|
            
            # 분모: min(K, |I_u|)
            denominator = min(10, num_relevant_items)

            # 분자: 상위 10개 추천 아이템 중 관련된 아이템 개수
            numerator = len(set(rec_items[:10]) & gt_items)

            # Recall@10 계산
            if denominator > 0:
                recalls.append(numerator / denominator)

    # 평균 Recall@10 반환
    return np.mean(recalls)

# ---------------------- GPU 기반 SLIM 모델 정의 ----------------------
class SLIMGPU(nn.Module):
    def __init__(self, num_items, _lambda=0.1, alpha=0.01, max_iter=100, device="cuda"):
        """
        _lambda: L2 정규화 파라미터
        alpha: 학습률
        max_iter: 최대 반복 횟수
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

# ---------------------- 데이터셋 정의 ----------------------
class InteractionDataset(Dataset):
    def __init__(self, interaction_matrix):
        # interaction_matrix는 numpy 배열 또는 PyTorch 텐서여야 함
        if isinstance(interaction_matrix, np.ndarray):
            self.data = torch.FloatTensor(interaction_matrix)
        elif isinstance(interaction_matrix, torch.Tensor):
            self.data = interaction_matrix
        else:
            raise ValueError("interaction_matrix must be a numpy array or PyTorch tensor")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

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

if __name__ == "__main__":
    # Optuna 스터디 생성
    def objective(trial):
        # 하이퍼파라미터 샘플링
        slim_lambda = trial.suggest_float('slim_lambda', 0.01, 1.0)
        slim_alpha = trial.suggest_float('slim_alpha', 0.0001, 0.1, log=True)
        slim_max_iter = trial.suggest_int('slim_max_iter', 50, 200)

        # 데이터 전처리
        data_path = "/data/ephemeral/home/KJPark/data/train/"

        data, interaction = data_pre_for_ease(data_path)
        users, items, encode_user, encode_item = encode_users_items(data)
        num_users = len(encode_user.classes_)
        num_items = len(encode_item.classes_)

        # CSR 및 Torch Sparse Matrix 생성
        X_csr = create_csr_matrix(users, items, interaction, num_users, num_items)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_torch_sparse = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

        # SLIM 모델 학습
        slim_model = SLIMGPU(num_items, _lambda=slim_lambda, alpha=slim_alpha, max_iter=slim_max_iter, device=device)
        slim_model.train(X_torch_sparse)

        # 추천 결과 생성
        predictions = slim_model.predict(X_torch_sparse).cpu().numpy()
        recommendations_df = pd.DataFrame({
            "user": np.repeat(np.arange(num_users), 10),
            "item": np.argsort(-predictions, axis=1)[:, :10].flatten()
        })

        # Ground truth 데이터 로드
        ground_truth_df = pd.read_csv('/data/ephemeral/home/ease.csv')

        # 모델 평가
        metric = evaluate_model(recommendations_df, ground_truth_df)

        return -metric  # Recall@10을 최대화하려면 음수로 반환

    # Optuna 실행
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # 모든 트라이얼 결과 출력
    print("\nAll Trials:")
    for trial in study.trials:
        print(f"Trial {trial.number}: Value={trial.value}, Params={trial.params}")

    # 베스트 트라이얼 출력
    print("\nBest Trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
