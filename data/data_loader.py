import os
from typing import Tuple
import pandas as pd


def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""
	데이터 로드 함수. 지정된 경로에서 데이터를 읽어옵니다.

	Args:
		data_path (str): 데이터 파일들이 저장된 디렉토리 경로.

	Returns:
		Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
			- 사용자-아이템 상호작용 데이터 (train_ratings.csv)
			- 영화 제목 데이터 (titles.tsv)
			- 영화 개봉 연도 데이터 (years.tsv)
			- 영화 감독 데이터 (directors.tsv)
			- 영화 장르 데이터 (genres.tsv)
			- 영화 작가 데이터 (writers.tsv)
	"""
	# 사용자-아이템 상호작용 데이터 로드
	train_rating = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))

	# 영화 제목 데이터 로드
	title = pd.read_csv(os.path.join(data_path, "titles.tsv"), sep='\t')

	# 영화 개봉 연도 데이터 로드
	year = pd.read_csv(os.path.join(data_path, "years.tsv"), sep='\t')

	# 영화 감독 데이터 로드
	director = pd.read_csv(os.path.join(data_path, "directors.tsv"), sep='\t')

	# 영화 장르 데이터 로드
	genre = pd.read_csv(os.path.join(data_path, "genres.tsv"), sep='\t')

	# 영화 작가 데이터 로드
	writer = pd.read_csv(os.path.join(data_path, "writers.tsv"), sep='\t')

	return train_rating, title, year, director, genre, writer