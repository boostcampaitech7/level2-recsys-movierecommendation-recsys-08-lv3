import pandas as pd

from .data_loader import load_data


def data_merge(data_path: str) -> pd.DataFrame:
    """
    데이터 병합 함수. load_data 함수를 이용해 데이터를 로드하고 병합합니다.

    Args:
        data_path (str): 데이터 파일들이 저장된 디렉토리 경로.

    Returns:
        pd.DataFrame: 병합된 데이터.
    """
    # 데이터 로드
    _, title, year, director, genre, writer = load_data(data_path)

    # 데이터 병합
    merged_data = pd.merge(title, year, on="item", how="left")
    merged_data = pd.merge(merged_data, director, on="item", how="left")
    merged_data = pd.merge(merged_data, genre, on="item", how="left")
    merged_data = pd.merge(merged_data, writer, on="item", how="left")

    merged_data = fill_missing_years(merged_data)

    return merged_data


def fill_missing_years(merged_data: pd.DataFrame) -> pd.DataFrame:
    """
    title에서 개봉년도를 추출하여 year의 결측치를 채우는 함수.

    Args:
        merged_data (pd.DataFrame): 병합된 데이터프레임.

    Returns:
        pd.DataFrame: year의 결측치가 채워진 데이터프레임.
    """
    merged_data["year"] = merged_data.apply(
        lambda row: (
            row["title"].split("(")[-1].split(")")[0] if pd.isnull(row["year"]) and "(" in row["title"] else row["year"]
        ),
        axis=1,
    )

    return merged_data
