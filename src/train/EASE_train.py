import numpy as np
from src.models import EASE
import pandas as pd
from sklearn.metrics import mean_squared_error
class EASETrainer:
    def __init__(self, args,data):
        """
        Handles the training and prediction logic for the EASE model.

        Parameters:
        - model: An instance of the EASE class.
        """
        self.args=args
        self.data=data
        self.params=self.args.model_args[self.args.model]
        self.model = EASE(_lambda=self.params['lambda'])


    def train(self):
        """
        Trains the EASE model.

        Parameters:
        - X: Sparse or dense user-item interaction matrix.

        Returns:
        - Trained model with updated weights (B).
        """
        if self.args.predict:
            X= self.data['basic']
        else:
            X=self.data['train']
        G = np.dot(X.T, X).toarray()  # G = X^T X
        diag_idx = list(range(G.shape[0]))  # Diagonal indices
        G[diag_idx, diag_idx] += self.model._lambda  # (X^T)X + (lambda)I
        P = np.linalg.inv(G)  # Inverse of (X^T)X + (lambda)I

        self.model.B = P / -np.diag(P)  # Compute final B matrix
        self.model.B[diag_idx, diag_idx] = 0  # Set diagonal values to 0
        return self.model

    def predict(self):
        """
        Predicts scores for the given interaction matrix.

        Parameters:
        - X: Sparse or dense user-item interaction matrix.

        Returns:
        - Predicted scores matrix.
        """ 
        if self.args.predict:
            X=self.data['basic']
        else:
            X=self.data['valid']
        return np.dot(X.toarray(), self.model.B)
    
    def evaluate(self,scores):
        if self.args.predict:
            X=self.data['basic']
            n_users = self.data['basic'].shape[0]
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
        else:

            X_valid= self.data['valid']
            X_valid_dense = X_valid.to_dense().numpy()
            
            scores_dense = scores
            mask = X_valid_dense > 0
            
            y_true = X_valid_dense[mask]
            y_pred = scores_dense[mask]
            validation_loss = mean_squared_error(y_true, y_pred)
            
            return validation_loss