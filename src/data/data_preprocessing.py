from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd


# 결측값 처리
def handle_all_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    director와 writer가 모두 NaN인 경우를 처리 → "unknown"으로 대체
    """
    data["director"] = data["director"].fillna("unknown")
    data["writer"] = data["writer"].fillna("unknown")
    return data

def handle_partial_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    director와 writer 중 하나만 NaN인 경우를 처리
    - 둘 다 NaN인 경우: "unknown"으로 대체
    - 하나만 NaN인 경우: NaN이 아닌 값으로 대체.
    """
    def fill_missing(row):
        if row["director"] == "unknown" and row["writer"] == "unknown":
            return "unknown", "unknown"
        if row["director"] == "unknown":
            return row["writer"], row["writer"]
        if row["writer"] == "unknown":
            return row["director"], row["director"]
        return row["director"], row["writer"]
    
    data[["director", "writer"]] = data.apply(
        lambda row: pd.Series(fill_missing(row)),
        axis=1
    )
    return data

def fill_missing_years_pytorch(data: pd.DataFrame) -> pd.DataFrame:
    """
    title에서 개봉년도를 추출하여 year의 결측치를 PyTorch 텐서 연산으로 채우는 함수.
    """
    # 문자열 데이터를 텐서로 변환
    title_tensor = torch.tensor(data["title"].to_numpy(), dtype=torch.object)
    year_tensor = torch.tensor(data["year"].to_numpy(), dtype=torch.object)

    # 연산
    def extract_year(title, year):
        if year is None and "(" in title:
            return title.split("(")[-1].split(")")[0]
        return year

    filled_years = [
        extract_year(title, year) for title, year in zip(title_tensor, year_tensor)
    ]
    data["year"] = filled_years
    return data

# 레이블 인코딩
def create_unique_person_id(data: pd.DataFrame) -> LabelEncoder:
    """
    작가(writer)와 감독(director) 정보를 결합하여 고유 ID 공간을 생성하고 레이블 인코딩합니다.
    """
    unique_persons = pd.concat([data['writer'], data['director']]).drop_duplicates()
    person_encoder = LabelEncoder()
    person_encoder.fit(unique_persons)
    return person_encoder

def encode_writer_director(ratings_merged: pd.DataFrame, person_encoder: LabelEncoder) -> pd.DataFrame:
    """
    writer와 director 열을 레이블 인코딩합니다.

    Args:
        ratings_merged (pd.DataFrame): 데이터프레임 (writer, director 열 포함).
        person_encoder (LabelEncoder): 작가와 감독에 대한 공통 ID 인코더.

    Returns:
        pd.DataFrame: writer와 director가 인코딩된 데이터프레임.
    """

    # writer와 director 각각 인코딩
    ratings_merged['writer_encoded'] = person_encoder.transform(ratings_merged['writer'])
    ratings_merged['director_encoded'] = person_encoder.transform(ratings_merged['director'])
    
    # 기존 컬럼 드랍 & 리네임
    ratings_merged = ratings_merged.drop(['writer', 'director'], axis=1)
    ratings_merged.rename(columns={'writer_encoded': 'writer'}, inplace=True)
    ratings_merged.rename(columns={'director_encoded': 'director'}, inplace=True)
    return ratings_merged


def encode_genre(ratings_merged: pd.DataFrame) -> pd.DataFrame:
    """
    장르(genre) 열을 레이블 인코딩합니다.

    Args:
        ratings_merged (pd.DataFrame): 데이터프레임 (genre 열 포함).

    Returns:
        pd.DataFrame: genre가 인코딩된 데이터프레임.
    """
    # 장르 인코딩
    genre_encoder = LabelEncoder()
    ratings_merged['genre_encoded'] = genre_encoder.fit_transform(ratings_merged['genre'])
    # 기존 컬럼 드랍 & 리네임
    ratings_merged = ratings_merged.drop(['genre'], axis=1)
    ratings_merged.rename(columns={'genre_encoded': 'genre'}, inplace=True)
    return ratings_merged