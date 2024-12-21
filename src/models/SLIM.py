import torch
import torch.nn as nn

class SLIMModel(nn.Module):
    def __init__(self, num_items, l1_reg=0.91, l2_reg=0.91, alpha=0.0002, max_iter=73, device='cpu'):
        super(SLIMModel, self).__init__()
        self.num_items = num_items
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.max_iter = max_iter
        self.device = device

        # Trainable weight matrix
        self.W = nn.Parameter(torch.zeros((num_items, num_items), device=self.device))

    def forward(self, X):
        """
        Forward method for SLIM Model.
        X: User-item interaction matrix (sparse or dense).
        """
        return torch.matmul(X, self.W)

    def train_slim(self, X):
        """
        Custom training logic for SLIM Model.
        """
        optimizer = torch.optim.SGD([self.W], lr=self.alpha)

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            # Predictions and loss computation
            preds = self.forward(X)
            errors = X - preds
            loss = (
                errors.square().sum() +
                self.l1_reg * self.W.abs().sum() +
                self.l2_reg * self.W.square().sum()
            ) / X.shape[0]

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()

            # Remove self-loops
            with torch.no_grad():
                self.W.fill_diagonal_(0)

        return self.W
