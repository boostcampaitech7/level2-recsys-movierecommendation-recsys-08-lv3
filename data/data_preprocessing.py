import pandas as pd
from sklearn.preprocessing import LabelEncoder

def merge_data(
    ratings: pd.DataFrame, genres: pd.DataFrame, directors: pd.DataFrame, 
    writers: pd.DataFrame, years: pd.DataFrame, titles: pd.DataFrame
) -> pd.DataFrame:
    """
    사용자-아이템 데이터와 타 데이터를 결합하고, rating 열을 추가합니다.

    Args:
        ratings (pd.DataFrame): 사용자-아이템 상호작용 데이터.
        genres (pd.DataFrame): 장르 데이터.
        directors (pd.DataFrame): 감독 데이터.
        writers (pd.DataFrame): 작가 데이터.
        years (pd.DataFrame): 개봉 연도 데이터.
        titles (pd.DataFrame): 영화 제목 데이터.

    Returns:
        pd.DataFrame: 결합된 데이터.
    """
    # 사용자-아이템 데이터와 장르 데이터 병합
    data = ratings.merge(genres, on="item", how="left")
    data = data.merge(directors, on="item", how="left")
    data = data.merge(writers, on="item", how="left")
    data = data.merge(years, on="item", how="left")
    data = data.merge(titles, on="item", how="left")

    # rating 열 추가
    data["rating"] = 1.0

    # 시간 순서로 정렬
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.sort_values(by=["user", "time"])

    return data


def handle_all_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    director와 writer가 모두 NaN인 경우를 처리 → "unknown"으로 대체

    Args:
        data (pd.DataFrame): 데이터프레임 (director, writer 열 포함).

    Returns:
        pd.DataFrame: 처리된 데이터프레임.
    """
    data["director"] = data["director"].fillna("unknown")
    data["writer"] = data["writer"].fillna("unknown")
    return data


def handle_partial_missing(data: pd.DataFrame) -> pd.DataFrame:
    """
    director와 writer 중 하나만 NaN인 경우를 처리
    - 둘 다 NaN인 경우: "unknown"으로 대
    - 하나만 NaN인 경우: NaN이 아닌 값으로 대체.

    Args:
        data (pd.DataFrame): 데이터프레임 (director, writer 열 포함).

    Returns:
        pd.DataFrame: 처리된 데이터프레임.
    """
    def fill_missing(row):
        if row["director"] == "unknown" and row["writer"] == "unknown":
            return "unknown", "unknown"  # 둘 다 없으면 unknown
        if row["director"] == "unknown":  # director가 없으면 writer로 대체
            return row["writer"], row["writer"]
        if row["writer"] == "unknown":  # writer가 없으면 director로 대체
            return row["director"], row["director"]
        return row["director"], row["writer"]  # 둘 다 있으면 그대로 반환

    # apply를 사용하여 결측값 처리
    data[["director", "writer"]] = data.apply(
        lambda row: pd.Series(fill_missing(row)),
        axis=1
    )
    return data

def fill_missing_years(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    title에서 개봉년도를 추출하여 year의 결측치를 채우는 함수.

    Args:
        merged_data (pd.DataFrame): 병합된 데이터프레임.

    Returns:
        pd.DataFrame: year의 결측치가 채워진 데이터프레임.
    """
    merged_data['year'] = merged_data.apply(
        lambda row: row['title'].split('(')[-1].split(')')[0] if pd.isnull(row['year']) and '(' in row['title'] else row['year'], axis=1
    )
    return merged_data

def create_unique_person_id(ratings_merged: pd.DataFrame) -> LabelEncoder:
    """
    작가(writer)와 감독(director) 정보를 결합하여 고유 ID 공간을 생성하고 레이블 인코딩합니다.

    Args:
        ratings_merged (pd.DataFrame): 데이터프레임 (writer, director 열 포함).

    Returns:
        LabelEncoder: 작가와 감독에 대한 공통 ID 인코더.
    """
    # 작가와 감독 컬럼 병합 후 고유 값 추출
    unique_persons = pd.concat([ratings_merged['writer'], ratings_merged['director']]).drop_duplicates()

    # 레이블 인코더 생성 및 학습
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
    return ratings_merged


