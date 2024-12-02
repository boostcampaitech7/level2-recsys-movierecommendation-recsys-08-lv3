import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def sgd(P, Q, b, b_u, b_i, samples, learning_rate, regularization):
    # 확률적 경사하강법을 이용한 업데이트
    for user_id, item_id, rating in samples:
        # 예측 평점 계산
        predicted_rating = b + b_u[user_id] + b_i[item_id] + P[user_id, :].dot(Q[item_id, :].T)
        error = rating - predicted_rating  # 오차 계산

        b_u[user_id] += learning_rate * (error - regularization * b_u[user_id])  # 사용자 편향 업데이트
        b_i[item_id] += learning_rate * (error - regularization * b_i[item_id])  # 아이템 편향 업데이트

        # 사용자 latent factor 업데이트
        P[user_id, :] += learning_rate * (error * Q[item_id, :] - regularization * P[user_id, :])
        # 아이템 latent factor 업데이트
        Q[item_id, :] += learning_rate * (error * P[user_id, :] - regularization * Q[item_id, :])


def binary_cross_entropy(R, predicted_R):
    # Binary Cross-Entropy을 이용한 손실 계산
    delta = 1e-7  # 0이 될 가능성 때문에 추가
    bce_matrix = -(R * np.log(predicted_R + delta) + (1 - R) * np.log(1 - predicted_R + delta))
    return bce_matrix.mean()  # 손실의 평균값 반환


def get_predicted_full_matrix(P, Q, b, b_u, b_i):
    # 전체 예측 행렬 계산
    if b is None:
        return P.dot(Q.T)  # b 없으면 그냥 내적
    else:
        return b + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P.dot(Q.T)


class MatrixFactorization(object):
    def __init__(self, R, K, epochs, learning_rate, regularization, verbose=True):
        self.R = R  # 사용자-아이템 행렬
        self.K = K  # Latent Factor의 차원
        self.num_users = R.shape[0]  # user 수
        self.num_items = R.shape[1]  # item 수
        self.learning_rate = learning_rate  # 학습률
        self.regularization = regularization  # 정규화 계수
        self.epochs = epochs  # epoch 수
        self.verbose = verbose  # 로그 출력 여부
        self.samples = list()  # 학습에 사용할 (user, item, rating) 샘플들

    def train(self):
        # P와 Q 초기화
        self.P = np.random.normal(scale=1.0 / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1.0 / self.K, size=(self.num_items, self.K))

        # b, b_u, b_i 초기화
        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        # 학습 데이터 샘플 과정
        for row in range(self.num_users):
            for column in range(self.num_items):
                if self.R[row, column] > 0:
                    self.samples.append((row, column, self.R[row, column]))
                elif random.random() < 0.2:
                    self.samples.append((row, column, self.R[row, column]))

        # 학습
        for epoch in tqdm(range(1, self.epochs + 1)):
            print("{} start".format(epoch))

            np.random.shuffle(self.samples)
            # sgd로 매개변수 업데이트
            sgd(self.P, self.Q, self.b, self.b_u, self.b_i, self.samples, self.learning_rate, self.regularization)
            # 예측 행렬 계산
            predicted_R = np.clip(self.get_predicted_full_matrix(), 0, 1)
            # bce 계산
            bce = binary_cross_entropy(self.R, predicted_R)

            if self.verbose and (epoch % 10 == 0):
                print("epoch: {} and Binary-Cross entropy: {}".format(epoch, bce))

    def get_predicted_full_matrix(self):
        # 전체 행렬 계산
        return get_predicted_full_matrix(self.P, self.Q, self.b, self.b_u, self.b_i)


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/lee/data/train/"
    data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    data["rating"] = 1.0

    user_item_matrix = data.pivot_table(index="user", columns="item", values="rating", fill_value=0)

    each_user_see_item = data.groupby("user")["item"].apply(list).to_dict()

    R = user_item_matrix.to_numpy()
    K = 50
    learning_rate = 0.002
    regularization = 0.2
    epochs = 10
    verbose = True

    mf_model = MatrixFactorization(R, K, epochs, learning_rate, regularization, verbose)
    print("train start")
    mf_model.train()

    print("predict start")
    predicted_user_item_matrix = pd.DataFrame(
        mf_model.get_predicted_full_matrix(), columns=user_item_matrix.columns, index=user_item_matrix.index
    )

    user_recommendations = {}
    unique_user = data["user"].unique()

    for user in unique_user:
        each_see_item = np.array(each_user_see_item[user])
        user_score = predicted_user_item_matrix.loc[user]
        user_not_see_item = user_score[~user_score.index.isin(each_see_item)]
        top_10_recommendations = user_not_see_item.nlargest(10).index.tolist()
        user_recommendations[user] = top_10_recommendations

    user_item_pairs = []

    for user, items in user_recommendations.items():
        for item in items:
            user_item_pairs.append([user, item])

    recommendations_df = pd.DataFrame(user_item_pairs, columns=["user", "item"])
    recommendations_df.to_csv("user_item.csv", index=False)
