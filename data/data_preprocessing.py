from sklearn.preprocessing import LabelEncoder
from pandarallel import pandarallel
import pandas as pd

# Pandarallel 초기화
pandarallel.initialize(progress_bar=True)

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

def fill_missing_years_parallel(data: pd.DataFrame) -> pd.DataFrame:
    """
    title에서 개봉년도를 추출하여 year의 결측치를 병렬 처리로 채우는 함수.
    """
    data["year"] = data.parallel_apply(
        lambda row: row["title"].split('(')[-1].split(')')[0] if pd.isnull(row["year"]) and '(' in row["title"] else row["year"],
        axis=1
    )
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

def encode_writer_director(data: pd.DataFrame, person_encoder: LabelEncoder) -> pd.DataFrame:
    """
    writer와 director 열을 레이블 인코딩합니다.
    """
    data['writer_encoded'] = person_encoder.transform(data['writer'])
    data['director_encoded'] = person_encoder.transform(data['director'])
    return data

def encode_genre(data: pd.DataFrame) -> pd.DataFrame:
    """
    장르(genre) 열을 레이블 인코딩합니다.
    """
    genre_encoder = LabelEncoder()
    data['genre_encoded'] = genre_encoder.fit_transform(data['genre'])
    return data
