import os
import pandas as pd
from typing import Tuple

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드 함수. 지정된 경로에서 데이터를 읽어옵니다.

    Args:
        data_path (str): 데이터 파일들이 저장된 디렉토리 경로.

    Returns:
        Tuple[pd.DataFrame, ...]: 각 데이터프레임을 반환.
    """
    train_rating = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
    title = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep="\t")
    year = pd.read_csv(os.path.join(data_path, "years.tsv"), sep="\t")
    director = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep="\t")
    genre = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep="\t")
    writer = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep="\t")

    return train_rating, title, year, director, genre, writer
