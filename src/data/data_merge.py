import pandas as pd

def merge_data(
    ratings: pd.DataFrame, genres: pd.DataFrame, directors: pd.DataFrame,
    writers: pd.DataFrame, years: pd.DataFrame, titles: pd.DataFrame
) -> pd.DataFrame:
    """
    데이터 병합 함수.

    Args:
        ratings (pd.DataFrame): 사용자-아이템 상호작용 데이터.
        genres (pd.DataFrame): 장르 데이터.
        directors (pd.DataFrame): 감독 데이터.
        writers (pd.DataFrame): 작가 데이터.
        years (pd.DataFrame): 개봉 연도 데이터.
        titles (pd.DataFrame): 영화 제목 데이터.

    Returns:
        pd.DataFrame: 병합된 데이터프레임.
    """
    data = ratings.merge(genres, on="item", how="left")
    data = data.merge(directors, on="item", how="left")
    data = data.merge(writers, on="item", how="left")
    data = data.merge(years, on="item", how="left")
    data = data.merge(titles, on="item", how="left")
    data["rating"] = 1.0  # rating 열 추가

    # 시간 순서로 정렬
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.sort_values(by=["user", "time"])
    
    return data