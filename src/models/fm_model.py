import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def data_load(data_path):
    train_rating = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    title = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
    year = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
    director = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
    genre = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
    writer = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
    train_rating["rating"] = 1.0

    return train_rating, title, year, director, genre, writer


def sample_negative_items(data, num_negative):
    ran_num = np.random.default_rng(42)
    items = set(data["item"].unique())
    neg_samples = []

    for user, group in data.groupby("user"):
        interacted_items = set(group["item"])
        not_see_items = list(items - interacted_items)

        if len(not_see_items) < num_negative:
            sampled_items = not_see_items
        else:
            sampled_items = ran_num.choice(not_see_items, size=num_negative, replace=False)

        for item in sampled_items:
            neg_samples.append({"user": user, "item": item, "rating": 0})

    return pd.DataFrame(neg_samples)


def user_and_item_encoder(data):
    encode_user = LabelEncoder()
    encode_item = LabelEncoder()

    data["user"] = encode_user.fit_transform(data["user"])
    data["item"] = encode_item.fit_transform(data["item"])

    return data, encode_user, encode_item


def genre_encoder(data):
    encode_genre = LabelEncoder()
    data["genre"] = encode_genre.fit_transform(data["genre"])
    return data, encode_genre


def writer_encoder(data):
    encode_writer = LabelEncoder()
    data["writer"] = encode_writer.fit_transform(data["writer"])
    return data, encode_writer


def director_encoder(data):
    encode_director = LabelEncoder()
    data["director"] = encode_director.fit_transform(data["director"])
    return data, encode_director


def merge_data(train_ratings, genres, writers, directors, years, titles, encode_item):
    genres["item"] = encode_item.transform(genres["item"])
    writers["item"] = encode_item.transform(writers["item"])
    directors["item"] = encode_item.transform(directors["item"])
    years["item"] = encode_item.transform(years["item"])
    titles["item"] = encode_item.transform(titles["item"])
    genrelist = genres.groupby("item")["genre"].apply(list)
    writerlist = writers.groupby("item")["writer"].apply(list)
    directorlist = directors.groupby("item")["director"].apply(list)

    df1 = pd.merge(train_ratings, genrelist, on="item", how="left")
    df2 = pd.merge(df1, writerlist, on="item", how="left")
    df3 = pd.merge(df2, directorlist, on="item", how="left")
    df4 = pd.merge(df3, years, on="item", how="left")
    df5 = pd.merge(df4, titles, on="item", how="left")

    return df5


def data_fillna(df):

    df["extract_year"] = df["title"].str.extract(r"\((\d{4})")[0]
    df["year"] = df["year"].fillna(df["extract_year"].astype("float64"))
    df["year"] = df["year"].astype("int32")
    df["year"] = (df["year"] - 1922) / (2014 - 1920)
    df.drop(columns=["time"], inplace=True)
    df.drop(columns=["title"], inplace=True)
    df.drop(columns=["extract_year"], inplace=True)
    df = df.fillna("unknown")
    df["director"] = df["director"].apply(lambda x: [] if x == "unknown" else x)
    df["writer"] = df["writer"].apply(lambda x: [] if x == "unknown" else x)

    return df


def unique_each_size(encode_user, encode_item, encode_genre, encode_writer, encode_director):
    n_user = len(encode_user.classes_)
    n_item = len(encode_item.classes_)
    n_genre = len(encode_genre.classes_)
    n_writer = len(encode_writer.classes_)
    n_director = len(encode_director.classes_)

    return n_user, n_item, n_genre, n_writer, n_director


class FMDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return {
            "user": row["user"],
            "item": row["item"],
            "genre": row["genre"],
            "writer": row["writer"],
            "director": row["director"],
            "year": row["year"],
            "rating": row["rating"],
        }


def add_offsets(df, n_user, n_item, n_genre, n_writer, padding_idx):
    user = df["user"]
    item = df["item"]
    genre = df["genre"]
    writer = df["writer"]
    director = df["director"]

    user = user
    item += n_user
    genre = [g + n_user + n_item for g in genre]
    writer = [w + n_user + n_item + n_genre for w in writer] if len(writer) > 0 else [padding_idx]
    director = [d + n_user + n_item + n_genre + n_writer for d in director] if len(director) > 0 else [padding_idx]

    return user, item, genre, writer, director


def pad_sequences(sequences, max_length, padding_idx):
    padded_sequences = []

    for seq in sequences:
        padding_length = max_length - len(seq)
        padding = [padding_idx] * padding_length
        padded_seq = seq + padding
        padded_sequences.append(padded_seq)

    return padded_sequences


def prepare_batch(batch, n_user, n_item, n_genre, n_writer, n_director, padding_idx):
    users, items, genres, writers, directors, years, ratings = [], [], [], [], [], [], []

    for data in batch:
        user, item, genre, writer, director = add_offsets(data, n_user, n_item, n_genre, n_writer, padding_idx)
        users.append(user)
        items.append(item)
        genres.append(genre)
        writers.append(writer)
        directors.append(director)
        years.append(data["year"])
        ratings.append(data["rating"])

    user_tensor = torch.tensor(users, dtype=torch.long)
    item_tensor = torch.tensor(items, dtype=torch.long)
    year_tensor = torch.tensor(years, dtype=torch.float32)
    rating_tensor = torch.tensor(ratings, dtype=torch.float32)

    max_genre_length = max(len(g) for g in genres)
    max_writer_length = max(len(w) for w in writers)
    max_director_length = max(len(d) for d in directors)

    genre_tensor = torch.tensor(pad_sequences(genres, max_genre_length, padding_idx), dtype=torch.long)
    writer_tensor = torch.tensor(pad_sequences(writers, max_writer_length, padding_idx), dtype=torch.long)
    director_tensor = torch.tensor(pad_sequences(directors, max_director_length, padding_idx), dtype=torch.long)

    return user_tensor, item_tensor, genre_tensor, writer_tensor, director_tensor, year_tensor, rating_tensor


class FMModel(nn.Module):
    def __init__(self, num_features, embed_dim, padding_idx):
        super(FMModel, self).__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        self.linear = nn.Embedding(num_features, 1, padding_idx=padding_idx)
        self.bias = nn.Parameter(torch.zeros(1))

        self.embedding = nn.Embedding(num_features, embed_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.continuous_linear = nn.Linear(1, 1)
        self.continuous_embedding = nn.Linear(1, embed_dim)

    def forward(self, categorical_features, continuous_features):
        linear_output = self.linear(categorical_features).sum(dim=1) + self.bias
        continuous_linear_output = self.continuous_linear(continuous_features).squeeze()

        embed_x = self.embedding(categorical_features)
        embed_continuous = self.continuous_embedding(continuous_features)
        all_embeddings = torch.cat([embed_x, embed_continuous.unsqueeze(1)], dim=1)

        sum_square = torch.sum(all_embeddings, dim=1) ** 2
        square_sum = torch.sum(all_embeddings**2, dim=1)
        interaction_output = 0.5 * torch.sum(sum_square - square_sum, dim=1)

        return linear_output.squeeze() + continuous_linear_output + interaction_output


def train_fm_model(model, dataloader, epochs, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            user_tensor, item_tensor, genre_tensor, writer_tensor, director_tensor, year_tensor, rating_tensor = batch
            categorical_features = torch.cat(
                [user_tensor.unsqueeze(1), item_tensor.unsqueeze(1), genre_tensor, writer_tensor, director_tensor],
                dim=1,
            )

            categorical_features = categorical_features.to(device)
            year_tensor = year_tensor.to(device).unsqueeze(1)
            rating_tensor = rating_tensor.to(device)

            predictions = model(categorical_features, year_tensor)
            loss = criterion(predictions, rating_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

    return model


def recommend_for_all_users(model, encode_user, encode_item, user_interactions, features, device="cpu"):
    model.eval()
    all_recommendations = []

    n_user = len(encode_user.classes_)
    n_item = len(encode_item.classes_)

    user_ids = list(range(n_user))
    item_ids = list(range(n_user, n_user + n_item))

    for user_id in tqdm(user_ids, desc="Recommending for users", unit="user"):
        interacted_items = user_interactions.get(user_id, [])
        candidate_items = [item for item in item_ids if item not in interacted_items]

        if not candidate_items:
            continue

        user_tensor = torch.tensor([user_id] * len(candidate_items), dtype=torch.long, device=device)
        item_tensor = torch.tensor(candidate_items, dtype=torch.long, device=device)

        genre_tensor = torch.tensor([features["genre"]] * len(candidate_items), dtype=torch.long, device=device)
        writer_tensor = torch.tensor([features["writer"]] * len(candidate_items), dtype=torch.long, device=device)
        director_tensor = torch.tensor([features["director"]] * len(candidate_items), dtype=torch.long, device=device)
        year_tensor = torch.tensor([features["year"]] * len(candidate_items), dtype=torch.float32, device=device)

        categorical_features = torch.cat(
            [user_tensor.unsqueeze(1), item_tensor.unsqueeze(1), genre_tensor, writer_tensor, director_tensor], dim=1
        )

        with torch.no_grad():
            predictions = model(categorical_features, year_tensor.unsqueeze(1))

        sorted_indices = torch.argsort(predictions, descending=True)
        top_n_indices = sorted_indices[:10]

        recommended_encoded_items = [candidate_items[i] - n_user for i in top_n_indices]

        recommendations = encode_item.inverse_transform(recommended_encoded_items)

        original_user_id = encode_user.inverse_transform([user_id])[0]

        for item in recommendations:
            all_recommendations.append({"user": original_user_id, "item": item})

    recommendations_df = pd.DataFrame(all_recommendations)
    return recommendations_df


def get_features(df, n_user, n_item, n_genre, n_writer, n_director, padding_idx):
    features = {"genre": {}, "writer": {}, "director": {}, "year": {}}

    for item_id in df["item"].unique():
        item_data = df[df["item"] == item_id]

        genre_offset = [g + n_user + n_item for g in item_data["genre"].values[0]]
        writer_offset = (
            [w + n_user + n_item + n_genre for w in item_data["writer"].values[0]]
            if len(item_data["writer"].values[0]) > 0
            else []
        )
        director_offset = (
            [d + n_user + n_item + n_genre + n_writer for d in item_data["director"].values[0]]
            if len(item_data["director"].values[0]) > 0
            else []
        )

        year = item_data["year"].values[0]

        features["genre"][item_id] = genre_offset
        features["writer"][item_id] = writer_offset
        features["director"][item_id] = director_offset
        features["year"][item_id] = year

    max_genre_length = max(len(v) for v in features["genre"].values())
    max_writer_length = max(len(v) for v in features["writer"].values())
    max_director_length = max(len(v) for v in features["director"].values())

    for item_id in features["genre"].keys():
        features["genre"][item_id] = pad_sequences([features["genre"][item_id]], max_genre_length, padding_idx)[0]
        features["writer"][item_id] = pad_sequences([features["writer"][item_id]], max_writer_length, padding_idx)[0]
        features["director"][item_id] = pad_sequences(
            [features["director"][item_id]], max_director_length, padding_idx
        )[0]

    return features


if __name__ == "__main__":
    data_path = "/data/ephemeral/home/lee/data/train/"
    train_rating, title, year, director, genre, writer = data_load(data_path)
    train_ratings, titles, years, directors, genres, writers = (
        train_rating.copy(),
        title.copy(),
        year.copy(),
        director.copy(),
        genre.copy(),
        writer.copy(),
    )

    neg_df = sample_negative_items(train_ratings, 50)
    train_ratings = pd.concat([train_ratings, neg_df], axis=0)
    train_ratings = train_ratings.sort_values(by="user").reset_index(drop=True)

    train_ratings, encode_user, encode_item = user_and_item_encoder(train_ratings)
    user_interactions = train_ratings.groupby("user")["item"].apply(list).to_dict()

    genres, encode_genre = genre_encoder(genres)
    writers, encode_writer = writer_encoder(writers)
    directors, encode_director = director_encoder(directors)
    df = merge_data(train_ratings, genres, writers, directors, years, titles, encode_item)
    df = data_fillna(df)

    n_user, n_item, n_genre, n_writer, n_director = unique_each_size(
        encode_user, encode_item, encode_genre, encode_writer, encode_director
    )

    padding_idx = n_user + n_item + n_genre + n_writer + n_director

    print("make feature")
    features = get_features(df, n_user, n_item, n_genre, n_writer, n_director, padding_idx)
    dataset = FMDataset(df)

    collate_fn = partial(
        prepare_batch,
        n_user=n_user,
        n_item=n_item,
        n_genre=n_genre,
        n_writer=n_writer,
        n_director=n_director,
        padding_idx=padding_idx,
    )

    dataloader = DataLoader(dataset, batch_size=2048, collate_fn=collate_fn)
    num_features = padding_idx + 1
    embed_dim = 16
    print("model define")
    fm_model = FMModel(num_features=num_features, embed_dim=embed_dim, padding_idx=padding_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5
    learning_rate = 0.001
    print("train")
    fm_model = train_fm_model(fm_model, dataloader, epochs, learning_rate, device)

    print("predict")
    recommendations_df = recommend_for_all_users(
        fm_model, encode_user, encode_item, user_interactions, features, device=device
    )

    recommendations_df.to_csv("fm_model.csv", index=False)
