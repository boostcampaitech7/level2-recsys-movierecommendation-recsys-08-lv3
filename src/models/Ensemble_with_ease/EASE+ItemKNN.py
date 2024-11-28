import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

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

# ---------------------- ItemKNN 모델 정의 ----------------------
class ItemKNN:
    def __init__(self, n_neighbors=20, similarity='cosine'):
        self.n_neighbors = n_neighbors
        self.similarity = similarity
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.similarity)

    def train(self, X):
        """
        X: CSR matrix (users x items)
        """
        self.model.fit(X.T)  # 아이템 기반으로 학습

    def predict(self, X):
        """
        X: CSR matrix (users x items)
        Returns: numpy array (users x items) with predicted scores
        """
        scores = np.zeros((X.shape[1], X.shape[0]))  # 아이템 x 사용자 형태
        batch_size = 1000  # 메모리 관리를 위해 배치 단위로 처리
        X_t = X.T  # 아이템 기반으로 변환
        for start in tqdm(range(0, X_t.shape[0], batch_size), desc="ItemKNN Predicting"):
            end = min(start + batch_size, X_t.shape[0])
            batch = X_t[start:end]  # [batch_size, 사용자 수]
            distances, indices = self.model.kneighbors(batch, return_distance=True)
            for i, item_idx in enumerate(range(start, end)):
                for neighbor_idx, neighbor_item in enumerate(indices[i]):
                    scores[item_idx, neighbor_item] += 1 / (1 + distances[i][neighbor_idx])  # 가중치 부여
        return scores.T  # 사용자 x 아이템으로 반환

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
def ensemble_and_generate_submission(scores_ease, scores_knn, X, encode_user, encode_item, alpha=0.7):
    """
    alpha: EASE 모델의 가중치 (0 ~ 1)
    (1 - alpha): ItemKNN 모델의 가중치
    """
    n_users = X.shape[0]
    result = []

    for user_idx in range(n_users):
        # 이미 본 아이템 제거
        user_row = X[user_idx].toarray().flatten()
        seen_items = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

        # 두 모델의 점수 가져오기
        scores_ease_user = scores_ease[user_idx]
        scores_knn_user = scores_knn[user_idx]

        # 앙상블 점수 계산 (가중 평균)
        final_scores = alpha * scores_ease_user + (1 - alpha) * scores_knn_user
        final_scores[seen_items] = -np.inf  # 이미 본 아이템 마스킹

        # 상위 10개 아이템 선택
        top_items_idx = np.argsort(final_scores)[-10:][::-1]
        top_items = encode_item.inverse_transform(top_items_idx)  # 원래 item id로
        user_id = encode_user.inverse_transform([user_idx])[0]  # 원래 user id로

        for item in top_items:
            result.append([user_id, item])

    recommendations_df = pd.DataFrame(result, columns=["user", "item"])
    return recommendations_df

# ---------------------- 메인 함수 ----------------------
if __name__ == "__main__":
    data_path = "/data/ephemeral/home/KJPark/data/train/"
    
    # EASE에 맞는 데이터 처리
    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)
    num_users = len(encode_user.classes_)
    num_items = len(encode_item.classes_)

    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # CSR matrix 생성
    X = create_csr_matrix(users, items, interaction, num_users, num_items)
    
    # ---------------------- EASE 모델 학습 및 예측 ----------------------
    _lambda = 450
    ease_model = EASE(_lambda)

    print("Training EASE model...")
    ease_model.train(X)
    print("Predicting with EASE model...")
    predict_result_ease = ease_model.predict(X)

    # ---------------------- ItemKNN 모델 학습 및 예측 ----------------------
    knn_model = ItemKNN(n_neighbors=20, similarity='cosine')
    print("Training ItemKNN model...")
    knn_model.train(X)
    print("Predicting with ItemKNN model...")
    predict_result_knn = knn_model.predict(X)

    # ---------------------- 앙상블 및 제출 파일 생성 ----------------------
    alpha = 0.7  # EASE 모델의 가중치, 필요에 따라 조정 가능
    print("Ensembling predictions and generating submission file...")
    recommendations_df = ensemble_and_generate_submission(
        predict_result_ease, predict_result_knn, X, encode_user, encode_item, alpha=alpha
    )

    # 제출 파일 생성
    recommendations_df.to_csv("ease_knn_ensemble.csv", index=False)
    print('Submission file saved as ease_knn_ensemble.csv')
