import numpy as np

class EASE:
    def __init__(self, _lambda):
        """
        Initialize the EASE class with a regularization parameter.

        Parameters:
        - _lambda: Regularization parameter for the diagonal elements.
        """
        self.B = None
        self._lambda = _lambda

    def train(self, X):
        """
        Train the EASE model.

        Parameters:
        - X: User-item interaction matrix (dense).
        """
        G = np.dot(X.T, X)  # G = X^T X
        diag_idx = np.arange(G.shape[0])  # Diagonal indices
        G[diag_idx, diag_idx] += self._lambda  # Regularization

        P = np.linalg.inv(G)  # Inverse of (X^T X + lambda I)
        self.B = -P / np.diag(P)  # Compute weight matrix B
        self.B[diag_idx, diag_idx] = 0  # Set diagonal to 0

    def predict(self, X):
        """
        Predict scores using the trained EASE model.

        Parameters:
        - X: User-item interaction matrix (dense).

        Returns:
        - Predicted scores matrix.
        """
        return np.dot(X, self.B)
