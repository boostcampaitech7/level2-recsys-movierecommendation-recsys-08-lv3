from tqdm import tqdm
from src.models import SLIMModel
import torch
import numpy as np
import pandas as pd

class SLIMTrainer:
    def __init__(self, args,data):
        """
        Handles the training and prediction logic for the SLIM model.
        
        Parameters:
        - model: An instance of SLIMModel.
        """
        self.args=args
        self.data=data
        self.params=args.model_args[args.model]
        self.model = SLIMModel(
            num_items=len(self.data['label2idx']['item'].keys()),  # Replace with actual number of items
            l1_reg=self.params['l1_reg'],
            l2_reg=self.params['l2_reg'],
            alpha=self.params['alpha'],
            max_iter=self.params['max_iter'],
            device=self.args.device,
        )

    def train(self):
        """
        Trains the SLIM model.
        
        Parameters:
        - X_sparse: PyTorch sparse matrix (users x items) on GPU
        """
        # Convert sparse matrix to dense
        X_sparse=self.data['basic']
        X = X_sparse.to_dense()  # [num_users, num_items]

        optimizer = torch.optim.SGD([self.model.W], lr=self.model.alpha)

        for _ in tqdm(range(self.model.max_iter), desc="Training SLIM on GPU"):
            optimizer.zero_grad()

            # Predictions and loss computation
            preds = torch.matmul(X, self.model.W)
            errors = X - preds
            loss = (
                errors.square().sum() +
                self.model.l1_reg * self.model.W.abs().sum() +
                self.model.l2_reg * self.model.W.square().sum()
            ) / X.shape[0]

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()

            # Remove self-loops
            with torch.no_grad():
                self.model.W.fill_diagonal_(0)
        return self.model

    def predict(self):
        """
        Predicts using the trained SLIM model.
        
        Parameters:
        - X_sparse: PyTorch sparse matrix (users x items) on GPU
        
        Returns:
        - Predicted scores (users x items) on GPU
        """
        X_sparse=self.data['basic']
        X = X_sparse.to_dense()  # Convert sparse matrix to dense
        with torch.no_grad():
            return torch.matmul(X, self.model.W)
        
    
    def evaluate(self,scores):
        
        X=self.data['basic']
        n_users = X.shape[0]
        result = []

        for user_idx in range(n_users):
            # 이미 본 아이템 제거
            user_row = X[user_idx].toarray().flatten()
            seen_items = np.where(user_row > 0)[0]  # 시청한 item idx 찾기

            # 점수 마스킹
            scores[user_idx, seen_items] = -np.inf

            # 상위 10개 아이템 선택
            top_items_idx = np.argsort(scores[user_idx])[-10:][::-1]

            for item in top_items_idx:
                result.append([user_idx, item])

        recommendations_df = pd.DataFrame(result, columns=["user", "item"])
        return recommendations_df
    