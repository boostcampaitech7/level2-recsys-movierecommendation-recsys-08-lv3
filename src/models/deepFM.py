import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class UnifiedDeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(UnifiedDeepFM, self).__init__()

        total_input_dim = int(sum(input_dims))
        # Unified embedding for all features
        self.embedding = nn.Embedding(total_input_dim, embedding_dim, padding_idx=sum(input_dims[:2]))
        self.fc = nn.Embedding(total_input_dim, 1, padding_idx=sum(input_dims[:2]))  # For linear terms
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.embedding_dim = embedding_dim * len(input_dims)  # User, item, genre, writer, director, year
        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i - 1], dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, user, item, genres, writers, directors, year):
        # Embedding lookup
        user_emb = self.embedding(user)  # Shape: (batch_size, embedding_dim)
        item_emb = self.embedding(item)  # Shape: (batch_size, embedding_dim)
        genre_emb = self.embedding(genres)  # Shape: (batch_size, max_genre_length, embedding_dim)
        writer_emb = self.embedding(writers)  # Shape: (batch_size, max_writer_length, embedding_dim)
        director_emb = self.embedding(directors)  # Shape: (batch_size, max_director_length, embedding_dim)
        year_emb = self.embedding(year)  # Shape: (batch_size, embedding_dim)

        # Masking for variable-length inputs
        genre_mask = (genres != self.embedding.padding_idx).unsqueeze(-1)  # Shape: (batch_size, max_genre_length, 1)
        writer_mask = (writers != self.embedding.padding_idx).unsqueeze(-1)
        director_mask = (directors != self.embedding.padding_idx).unsqueeze(-1)

        # Aggregate embeddings
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)  # (batch_size, embedding_dim)
        writer_emb = (writer_emb * writer_mask).sum(dim=1) / (writer_mask.sum(dim=1) + 1e-8)
        director_emb = (director_emb * director_mask).sum(dim=1) / (director_mask.sum(dim=1) + 1e-8)

        # Linear terms
        genre_fc = self.fc(genres)
        genre_fc = (genre_fc * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)
        writer_fc = self.fc(writers)
        writer_fc = (writer_fc * writer_mask).sum(dim=1) / (writer_mask.sum(dim=1) + 1e-8)
        director_fc = self.fc(directors)
        director_fc = (director_fc * director_mask).sum(dim=1) / (director_mask.sum(dim=1) + 1e-8)

        fm_y = self.bias + \
               torch.sum(self.fc(user), dim=1, keepdim=True) + \
               torch.sum(self.fc(item), dim=1, keepdim=True) + \
               torch.sum(genre_fc, dim=1, keepdim=True) + \
               torch.sum(writer_fc, dim=1, keepdim=True) + \
               torch.sum(director_fc, dim=1, keepdim=True) + \
               torch.sum(self.fc(year), dim=1, keepdim=True)

        # FM interaction terms
        embed_x = torch.cat([user_emb, item_emb, genre_emb, writer_emb, director_emb, year_emb], dim=1)
        square_of_sum = torch.sum(embed_x, dim=1, keepdim=True) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1, keepdim=True)
        fm_y += 0.5 * (square_of_sum - sum_of_square)

        return fm_y  # Shape: (batch_size, 1)

    def mlp(self, user, item, genres, writers, directors, year):
        # Embedding lookup
        user_emb = self.embedding(user)
        item_emb = self.embedding(item)
        genre_emb = self.embedding(genres)
        writer_emb = self.embedding(writers)
        director_emb = self.embedding(directors)
        year_emb = self.embedding(year)

        # Masking for variable-length inputs
        genre_mask = (genres != self.embedding.padding_idx).unsqueeze(-1)
        writer_mask = (writers != self.embedding.padding_idx).unsqueeze(-1)
        director_mask = (directors != self.embedding.padding_idx).unsqueeze(-1)

        # Aggregate embeddings
        genre_emb = (genre_emb * genre_mask).sum(dim=1) / (genre_mask.sum(dim=1) + 1e-8)
        writer_emb = (writer_emb * writer_mask).sum(dim=1) / (writer_mask.sum(dim=1) + 1e-8)
        director_emb = (director_emb * director_mask).sum(dim=1) / (director_mask.sum(dim=1) + 1e-8)

        # Concatenate embeddings
        embed_x = torch.cat([user_emb, item_emb, genre_emb, writer_emb, director_emb, year_emb], dim=1)

        # Pass through MLP layers
        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, user, item, genres, writers, directors, year):
        # FM component
        fm_y = self.fm(user, item, genres, writers, directors, year).squeeze(1)

        # MLP component
        mlp_y = self.mlp(user, item, genres, writers, directors, year).squeeze(1)

        # Final prediction
        y = torch.sigmoid(fm_y + mlp_y)
        return y


