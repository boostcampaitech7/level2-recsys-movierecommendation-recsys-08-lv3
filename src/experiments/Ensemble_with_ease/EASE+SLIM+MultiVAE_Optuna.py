import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

import optuna

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

# ---------------------- Mult-VAE 모델 정의 ----------------------
class MultVAE(nn.Module):
    def __init__(self, num_items, hidden_dim=600, latent_dim=200, dropout=0.5):
        super(MultVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_items),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization
        return self.decoder(z), mu, logvar

# ---------------------- Mult-VAE 학습 및 예측 함수 ----------------------
def train_multvae(model, data_loader, optimizer, num_epochs=10, anneal_cap=0.2, total_anneal_steps=200000):
    model.train()
    update_count = 0
    for epoch in range(num_epochs):
        for batch in tqdm(data_loader, desc=f"Training Mult-VAE Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(model.device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            # Loss 계산 (Multinomial likelihood + KLD)
            recon_loss = -torch.sum(batch * torch.log(recon_batch + 1e-10), dim=1)
            kld_weight = min(anneal_cap, update_count / total_anneal_steps)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss = torch.mean(recon_loss + kld_weight * kld_loss)
            loss.backward()
            optimizer.step()
            update_count += 1

def predict_multvae(model, data_loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting with Mult-VAE"):
            batch = batch.to(model.device)
            recon_batch, _, _ = model(batch)
            preds.append(recon_batch.cpu())
    return torch.cat(preds, dim=0).numpy()

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

# ---------------------- 앙상블 및 제출 파일 생성 함수 ----------------------
def ensemble_and_generate_submission(scores_ease, scores_slim, scores_multvae, X, encode_user, encode_item, alphas):
    """
    alphas: [alpha_ease, alpha_slim, alpha_multvae]
    """
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]

        # 각 모델의 점수 가져오기
        scores_ease_user = scores_ease[user_idx]
        scores_slim_user = scores_slim[user_idx]
        scores_multvae_user = scores_multvae[user_idx]

        # 앙상블 점수 계산
        final_scores = (
            alphas[0] * scores_ease_user +
            alphas[1] * scores_slim_user +
            alphas[2] * scores_multvae_user
        )
        final_scores[seen_items] = -np.inf  # 이미 본 아이템 마스킹

        # 상위 10개 아이템 선택
        top_items_idx = np.argsort(final_scores)[-10:][::-1]
        top_items = encode_item.inverse_transform(top_items_idx)
        user_id = encode_user.inverse_transform([user_idx])[0]

        for item in top_items:
            result.append([user_id, item])

    recommendations_df = pd.DataFrame(result, columns=["user", "item"])
    return recommendations_df

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

import optuna

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

# ---------------------- Optuna Objective Function ----------------------
def objective(trial):
    # 하이퍼파라미터 샘플링
    # EASE 하이퍼파라미터
    ease_lambda = trial.suggest_float('ease_lambda', 100, 1000)
    # SLIM 하이퍼파라미터
    slim_lambda = trial.suggest_float('slim_lambda', 0.01, 1.0)
    slim_alpha = trial.suggest_float('slim_alpha', 0.0001, 0.1, log=True)
    slim_max_iter = trial.suggest_int('slim_max_iter', 50, 200)
    # Mult-VAE 하이퍼파라미터
    multvae_hidden_dim = trial.suggest_int('multvae_hidden_dim', 200, 1000, step=100)
    multvae_latent_dim = trial.suggest_int('multvae_latent_dim', 50, 500, step=50)
    multvae_dropout = trial.suggest_float('multvae_dropout', 0.0, 0.5)
    multvae_num_epochs = trial.suggest_int('multvae_num_epochs', 5, 20)
    multvae_anneal_cap = trial.suggest_float('multvae_anneal_cap', 0.1, 0.5)
    multvae_total_anneal_steps = trial.suggest_int('multvae_total_anneal_steps', 100000, 500000, step=50000)
    multvae_lr = trial.suggest_float('multvae_lr', 0.0001, 0.01, log=True)
    # 앙상블 가중치 (alphas)
    alpha_ease = trial.suggest_float('alpha_ease', 0.0, 1.0)
    alpha_slim = trial.suggest_float('alpha_slim', 0.0, 1.0 - alpha_ease)
    alpha_multvae = 1.0 - alpha_ease - alpha_slim
    if alpha_multvae < 0:
        alpha_multvae = 0.0
    alphas = [alpha_ease, alpha_slim, alpha_multvae]
    # 가중치 정규화
    alpha_sum = sum(alphas)
    if alpha_sum > 0:
        alphas = [a / alpha_sum for a in alphas]
    else:
        alphas = [1/3, 1/3, 1/3]

    # 데이터 전처리 (캐싱 고려)
    data_path = "/data/ephemeral/home/KJPark/data/train/"

    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)
    num_users = len(encode_user.classes_)
    num_items = len(encode_item.classes_)

    X = create_csr_matrix(users, items, interaction, num_users, num_items)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_torch_sparse = create_torch_sparse_matrix(users, items, interaction, num_users, num_items, device=device)

    # ---------------------- EASE 모델 학습 및 예측 ----------------------
    ease_model = EASE(ease_lambda)
    ease_model.train(X)
    predict_result_ease = ease_model.predict(X)

    # ---------------------- SLIM 모델 학습 및 예측 ----------------------
    slim_model = SLIMGPU(num_items, _lambda=slim_lambda, alpha=slim_alpha, max_iter=slim_max_iter, device=device)
    slim_model.train(X_torch_sparse)
    predict_result_slim = slim_model.predict(X_torch_sparse).cpu().numpy()

    # ---------------------- Mult-VAE 모델 학습 및 예측 ----------------------
    X_dense = X.toarray()  # numpy 배열로 변환
    dataset = InteractionDataset(X_dense)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    multvae_model = MultVAE(num_items, hidden_dim=multvae_hidden_dim, latent_dim=multvae_latent_dim, dropout=multvae_dropout).to(device)
    multvae_model.device = device  # 모델에 디바이스 정보 추가
    optimizer = torch.optim.Adam(multvae_model.parameters(), lr=multvae_lr)
    train_multvae(multvae_model, data_loader, optimizer, num_epochs=multvae_num_epochs, anneal_cap=multvae_anneal_cap, total_anneal_steps=multvae_total_anneal_steps)

    # 예측
    data_loader_eval = DataLoader(dataset, batch_size=512, shuffle=False)
    predict_result_multvae = predict_multvae(multvae_model, data_loader_eval)

    # ---------------------- 앙상블 및 제출 파일 생성 ----------------------
    recommendations_df = ensemble_and_generate_submission(
        predict_result_ease, predict_result_slim, predict_result_multvae, X, encode_user, encode_item, alphas
    )

    # Ground truth 데이터 로드
    ground_truth_df = pd.read_csv('/data/ephemeral/home/ease.csv')

    # 모델 평가
    metric = evaluate_model(recommendations_df, ground_truth_df)

    return -metric  # Recall@10을 최대화하려면 음수로 반환

# ---------------------- 메인 실행부 ----------------------
if __name__ == "__main__":
    # Optuna 스터디 생성
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
