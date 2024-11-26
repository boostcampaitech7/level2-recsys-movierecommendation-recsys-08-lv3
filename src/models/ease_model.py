import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


class EASE:
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda

    def train(self, X):
        G = np.dot(X.T, X).toarray()  # G = X^T X
        diag_idx = list(range(G.shape[0]))  # diag index 구하기
        G[diag_idx, diag_idx] += self._lambda  # (X^T)X + (lambda)I
        P = np.linalg.inv(G)  # (X^T)X + (lambda)I 역행렬

        self.B = P / -np.diag(P)  # 최종 B
        self.B[diag_idx, diag_idx] = 0  # 대각 값 0

    def predict(self, X):
        return np.dot(X.toarray(), self.B)  # 예측 점수 계산


def predict_each_user(predict_result, X, encode_user, encode_item):
    n_users, n_items = X.shape[0], X.shape[1]

    result = []

    for user_idx in range(n_users):
        user_row = X[user_idx].toarray().flatten()
        see_user = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

        scores = predict_result[user_idx]
        scores[see_user] = -np.inf  # 시청한 item 제거

        top_items_idx = np.argsort(scores)[-10:][::-1]
        top_items = encode_item.inverse_transform(top_items_idx)  # 원래 item id로
        user_id = encode_user.inverse_transform([user_idx])[0]  # 원래 user id로

        for item in top_items:
            result.append([user_id, item])

    recommendations_df = pd.DataFrame(result, columns=["user", "item"])

    return recommendations_df


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


def create_csr_matrix(users, items, values):
    return csr_matrix((values, (users, items)))


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/lee/data/train/"
    # EASE에 맞는 data 처리
    data, interaction = data_pre_for_ease(data_path)
    users, items, encode_user, encode_item = encode_users_items(data)

    # CSR matrix
    X = create_csr_matrix(users, items, interaction)

    _lambda = 450
    ease_model = EASE(_lambda)

    print("Train")
    ease_model.train(X)
    print("predict score")
    predict_result = ease_model.predict(X)
    print("item 10 for each user")
    recommendations_df = predict_each_user(predict_result, X, encode_user, encode_item)

    # 제출 파일 생성
    recommendations_df.to_csv("ease.csv", index=False)
