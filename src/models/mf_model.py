import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def sgd(P, Q, b, b_u, b_i, samples, learning_rate, regularization):
    for user_id, item_id, rating in samples:
        predicted_rating = b + b_u[user_id] + b_i[item_id] + P[user_id, :].dot(Q[item_id, :].T)
        error = rating - predicted_rating

        b_u[user_id] += learning_rate * (error - regularization * b_u[user_id])
        b_i[item_id] += learning_rate * (error - regularization * b_i[item_id])

        P[user_id, :] += learning_rate * (error * Q[item_id, :] - regularization * P[user_id, :])
        Q[item_id, :] += learning_rate * (error * P[user_id, :] - regularization * Q[item_id, :])


def binary_cross_entropy(R, predicted_R):
    delta = 1e-7
    bce_matrix = -(R * np.log(predicted_R + delta) + (1 - R) * np.log(1 - predicted_R + delta))
    return bce_matrix.mean()


def get_predicted_full_matrix(P, Q, b, b_u, b_i):
    if b is None:
        return P.dot(Q.T)
    else:
        return b + b_u[:, np.newaxis] + b_i[np.newaxis, :] + P.dot(Q.T)


class MatrixFactorization(object):
    def __init__(self, R, K, epochs, learning_rate, regularization, verbose=True):
        self.R = R
        self.K = K
        self.num_users = R.shape[0]
        self.num_items = R.shape[1]
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.verbose = verbose
        self.samples = list()

    def train(self):
        self.P = np.random.normal(scale=1.0 / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1.0 / self.K, size=(self.num_items, self.K))

        self.b = np.mean(self.R[np.where(self.R != 0)])
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)

        for row in range(self.num_users):
            for column in range(self.num_items):
                if self.R[row, column] > 0:
                    self.samples.append((row, column, self.R[row, column]))
                elif random.random() < 0.2:
                    self.samples.append((row, column, self.R[row, column]))

        for epoch in tqdm(range(1, self.epochs + 1)):
            print("{} start".format(epoch))
            np.random.shuffle(self.samples)
            sgd(self.P, self.Q, self.b, self.b_u, self.b_i, self.samples, self.learning_rate, self.regularization)
            predicted_R = np.clip(self.get_predicted_full_matrix(), 0, 1)
            bce = binary_cross_entropy(self.R, predicted_R)

            if self.verbose and (epoch % 10 == 0):
                print("epoch: {} and Binary-Cross entropy: {}".format(epoch, bce))

    def get_predicted_full_matrix(self):
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
