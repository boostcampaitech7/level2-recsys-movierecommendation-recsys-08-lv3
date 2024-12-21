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

# ---------------------- MultiVAE 모델 정의 ----------------------
class MultiVAE(nn.Module):
    def __init__(self, num_items, hidden_dims=[200, 600], dropout=0.5):
        super(MultiVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], num_items),
            nn.Sigmoid()  # 출력 값을 0과 1 사이로
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def train_model(self, train_loader, optimizer, criterion, device, epochs=10):
        self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Training MultiVAE Epoch {epoch+1}/{epochs}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                output = self.forward(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

    def predict(self, X_dense):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_dense).to(next(self.parameters()).device)
            output = self.forward(X_tensor)
            return output.cpu().numpy()

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

# ---------------------- 앙상블 및 제출 파일 생성 함수 ----------------------
def ensemble_and_generate_submission(scores_ease, scores_multivae, X, encode_user, encode_item, alphas):
    """
    Ensemble by selecting top 5 recommendations from EASE and top 5 from MultiVAE for each user.
    
    Parameters:
        scores_ease (np.ndarray): EASE prediction scores (users x items)
        scores_multivae (np.ndarray): MultiVAE prediction scores (users x items)
        X (csr_matrix): Interaction matrix (users x items)
        encode_user (LabelEncoder): Encoder for user IDs
        encode_item (LabelEncoder): Encoder for item IDs
        alphas (list): [alpha_ease, alpha_multivae] (Not used in this ensemble method)
    
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

# ---------------------- 데이터 전처리 함수 ----------------------
def create_torch_sparse_matrix(users, items, values, num_users, num_items, device="cuda"):
    """
    PyTorch sparse tensor로 변환
    """
    users = np.array(users)  # 리스트를 numpy 배열로 변환
    items = np.array(items)
    indices = torch.tensor([users, items], dtype=torch.long, device=device)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=(num_users, num_items))

# ---------------------- Optuna Objective Function ----------------------
def objective(trial):
    # 하이퍼파라미터 샘플링
    # EASE 하이퍼파라미터
    ease_lambda = trial.suggest_float('ease_lambda', 100, 1000)
    
    # MultiVAE 하이퍼파라미터
    multivae_hidden1 = trial.suggest_int('multivae_hidden1', 100, 300)
    multivae_hidden2 = trial.suggest_int('multivae_hidden2', 300, 700)
    multivae_dropout = trial.suggest_float('multivae_dropout', 0.2, 0.6)
    multivae_learning_rate = trial.suggest_float('multivae_learning_rate', 1e-4, 1e-2, log=True)
    multivae_epochs = trial.suggest_int('multivae_epochs', 10, 30)

    # 앙상블 가중치 (alphas)
    alpha_ease = trial.suggest_float('alpha_ease', 0.0, 1.0)
    alpha_multivae = 1.0 - alpha_ease
    alphas = [alpha_ease, alpha_multivae]
    # 가중치 정규화
    alpha_sum = sum(alphas)
    if alpha_sum > 0:
        alphas = [a / alpha_sum for a in alphas]
    else:
        alphas = [0.5, 0.5]  # 기본값

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

    # ---------------------- MultiVAE 모델 학습 및 예측 ----------------------
    # Dense 행렬로 변환 (MultiVAE는 Dense 입력을 받음)
    X_dense = X.toarray()

    # MultiVAE 모델 초기화
    multivae_model = MultiVAE(
        num_items=num_items,
        hidden_dims=[multivae_hidden1, multivae_hidden2],
        dropout=multivae_dropout
    )

    # 데이터 로더 준비
    dataset = InteractionDataset(X_dense)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 옵티마이저 및 손실 함수 정의
    optimizer = torch.optim.Adam(multivae_model.parameters(), lr=multivae_learning_rate)
    criterion = nn.BCELoss()

    # MultiVAE 학습
    multivae_model.train_model(train_loader, optimizer, criterion, device, epochs=multivae_epochs)

    # MultiVAE 예측
    predict_result_multivae = multivae_model.predict(X_dense)

    # ---------------------- 앙상블 및 제출 파일 생성 ----------------------
    recommendations_df = ensemble_and_generate_submission(
        predict_result_ease, predict_result_multivae, X, encode_user, encode_item, alphas
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
    study.optimize(objective, n_trials=50)

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
