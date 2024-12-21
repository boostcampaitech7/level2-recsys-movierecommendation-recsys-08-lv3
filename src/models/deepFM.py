import torch
import torch.nn as nn

class UnifiedDeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(UnifiedDeepFM, self).__init__()

        total_input_dim = int(sum(input_dims))
        self.embedding = nn.Embedding(total_input_dim, embedding_dim, padding_idx=sum(input_dims[:2]))
        self.fc = nn.Embedding(total_input_dim, 1, padding_idx=sum(input_dims[:2]))
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.embedding_dim = embedding_dim * len(input_dims)
        
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
        user_emb = self.embedding(user)
        item_emb = self.embedding(item)
        genre_emb = self._aggregate_embeddings(genres)
        writer_emb = self._aggregate_embeddings(writers)
        director_emb = self._aggregate_embeddings(directors)
        year_emb = self.embedding(year)

        linear_terms = (self.bias +
                        torch.sum(self.fc(user), dim=1, keepdim=True) +
                        torch.sum(self.fc(item), dim=1, keepdim=True) +
                        torch.sum(self.fc(year), dim=1, keepdim=True))

        interaction_terms = self._interaction_terms(user_emb, item_emb, genre_emb, writer_emb, director_emb, year_emb)
        return linear_terms + interaction_terms

    def mlp(self, user, item, genres, writers, directors, year):
        user_emb = self.embedding(user)
        item_emb = self.embedding(item)
        genre_emb = self._aggregate_embeddings(genres)
        writer_emb = self._aggregate_embeddings(writers)
        director_emb = self._aggregate_embeddings(directors)
        year_emb = self.embedding(year)

        embed_x = torch.cat([user_emb, item_emb, genre_emb, writer_emb, director_emb, year_emb], dim=1)
        return self.mlp_layers(embed_x)

    def _aggregate_embeddings(self, features):
        mask = (features != self.embedding.padding_idx).unsqueeze(-1)
        embeddings = self.embedding(features) * mask
        return embeddings.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    def _interaction_terms(self, *embeddings):
        embed_x = torch.cat(embeddings, dim=1)
        square_of_sum = torch.sum(embed_x, dim=1, keepdim=True) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1, keepdim=True)
        return 0.5 * (square_of_sum - sum_of_square)

    def forward(self, user, item, genres, writers, directors, year):
        fm_y = self.fm(user, item, genres, writers, directors, year).squeeze(1)
        mlp_y = self.mlp(user, item, genres, writers, directors, year).squeeze(1)
        return torch.sigmoid(fm_y + mlp_y)