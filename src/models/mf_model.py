import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        dot_product = (user_vec * item_vec).sum(dim=1)
        return self.sigmoid(dot_product)
