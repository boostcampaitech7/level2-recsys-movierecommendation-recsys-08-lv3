import torch
import torch.nn as nn

class SLIMModel(nn.Module):
    def __init__(self, num_items, l1_reg=0.91, l2_reg=0.91, alpha=0.0002, max_iter=73, device='cpu'):
        """
        l1_reg: L1 regularization term
        l2_reg: L2 regularization term
        alpha: Learning rate
        max_iter: Maximum number of iterations
        """
        super(SLIMModel, self).__init__()
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.max_iter = max_iter
        self.device = device

        # Trainable weight matrix
        self.W = nn.Parameter(torch.zeros((num_items, num_items), device=self.device))
        self.num_items = num_items
