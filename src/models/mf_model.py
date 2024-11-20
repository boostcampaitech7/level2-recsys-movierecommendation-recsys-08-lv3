import os
import random

import numpy as np
import pandas as pd
import torch.nn as nn
import tqdm


def sgd():
    pass


def binary_croos_entropy():
    pass


def get_predicted_full_matrix():
    pass


class MatrixFactorization(object):
    def __init__(self, R, K, learning_rate, regularization, epochs, verbose=False):
        self.R = R
        self.K = K
        self.num_users = R.shape[0]
        self.num_items = R.shape[1]
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.verbose = verbose
        self.samples = list()
        self.training_process = list()

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
                elif random.random() < 0.2:  # negative sampling 수정하기
                    self.samples.append((row, column, self.R[row, column]))

        for epoch in tqdm(range()):
            np.random.shuffle(self.samples)
            sgd()  # 추가
            predicted_R = np.clip(self.get_predicted_full_matrix(), 0, 1)
            bce = binary_croos_entropy(self.R, predicted_R)
            self.training_process.append((epoch, bce))

            if self.verbose and (epoch % 10 == 0):
                print()  # 추가

    def get_predicted_full_matrix(self):
        return get_predicted_full_matrix()  # 추가


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/lee/data/train/"
    data = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    data["rating"] = 1.0

    user_item_matrix = data.pivot_table(index="user", columns="item", values="rating", fill_value=0)

    R = user_item_matrix.to_numpy()
    K = 50
    learning_rate = 0.002
    regularization = 0.2
    epochs = 100
    verbose = True
