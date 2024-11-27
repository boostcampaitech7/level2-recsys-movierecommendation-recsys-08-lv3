import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


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

    # encoder
    train_ratings, encode_user, encode_item = user_and_item_encoder(train_ratings)
    genres, encode_genre = genre_encoder(genres)
    writers, encode_writer = writer_encoder(writers)
    directors, encode_director = director_encoder(directors)
    df = merge_data(train_ratings, genres, writers, directors, years, titles, encode_item)
    df = data_fillna(df)

    n_user, n_item, n_genre, n_writer, n_director = unique_each_size(
        encode_user, encode_item, encode_genre, encode_writer, encode_director
    )

    padding_idx = n_user + n_item + n_genre + n_writer + n_director
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
