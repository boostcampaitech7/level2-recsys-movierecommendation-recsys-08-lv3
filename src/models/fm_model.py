import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def data_load(data_path):
    train_rating = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    title = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
    year = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
    director = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
    genre = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
    writer = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")
    genrelist = genre.groupby("item")["genre"].apply(list)
    writerlist = writer.groupby("item")["writer"].apply(list)
    directorlist = director.groupby("item")["director"].apply(list)

    return train_rating, title, year, director, genre, writer


def user_and_item_encoder(data):
    encode_user = LabelEncoder()
    encode_item = LabelEncoder()

    data["user"] = encode_user.fit_transform(data["user"])
    data["item"] = encode_item.fit_transform(data["item"])

    return data, encode_user, encode_item


def genre_encoder(data):
    encoder_genre = LabelEncoder()
    data["genre"] = encoder_genre.fit_transform(data["genre"])
    return data, encoder_genre


def writer_encoder(data):
    encoder_writer = LabelEncoder()
    data["writer"] = encoder_writer.fit_transform(data["writer"])
    return data, encoder_writer


def director_encoder(data):
    encoder_director = LabelEncoder()
    data["director"] = encoder_director.fit_transform(data["director"])
    return data, encoder_director


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


def preprocess_for_fm(data_path):
    pass


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

    # encoder
    train_ratings, encode_user, encode_item = user_and_item_encoder(train_ratings)
    genres, encoder_genre = genre_encoder(genres)
    writers, encoder_writer = writer_encoder(writers)
    directors, encoder_director = director_encoder(directors)
    df = merge_data(train_ratings, genres, writers, directors, years, titles, encode_item)
    df = data_fillna(df)
    print(df.head())
