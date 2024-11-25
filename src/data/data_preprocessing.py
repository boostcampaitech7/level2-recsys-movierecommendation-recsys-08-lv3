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
def label_encode_columns(df, columns):
    """
    데이터 레이블링 함수: 지정된 열에서 고유값을 정수로 매핑하여 데이터프레임의 값을 변환합니다.

    Args:
        df (pd.DataFrame): 레이블링을 적용할 데이터프레임.
        columns (list): 레이블링을 적용할 열 이름의 리스트.

    Returns:
        pd.DataFrame: 지정된 열들이 레이블링된 새로운 데이터프레임.
    """
    for col in columns:
        # 문자열로 변환하여 정렬 (숫자와 문자열 혼합 처리)
        df[col] = df[col].astype(str)
        unique_values = sorted(set(df[col]))  # 고유값 정렬
        mapping_dict = {unique_values[i]: i for i in range(len(unique_values))}
        df[col] = df[col].map(lambda x: mapping_dict[x])
    
    # user 기준 정렬 및 인덱스 리셋
    df = df.sort_values(by=['user'])
    df.reset_index(drop=True, inplace=True)
    return df