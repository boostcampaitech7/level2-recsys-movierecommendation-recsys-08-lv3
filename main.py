from data.data_loader import load_data
from data.data_merge import merge_data
from data.data_preprocessing import (
    handle_all_missing,
    handle_partial_missing,
    fill_missing_years_parallel,
    create_unique_person_id,
    encode_writer_director,
    encode_genre
)

def main(data_path: str):
    # 1. 데이터 로드
    ratings, titles, years, directors, genres, writers = load_data(data_path)

    # 2. 데이터 병합
    merged_data = merge_data(ratings, genres, directors, writers, years, titles)

    # 3. 결측값 처리
    merged_data = handle_all_missing(merged_data)
    merged_data = handle_partial_missing(merged_data)
    merged_data = fill_missing_years_parallel(merged_data)

    # 4. 레이블 인코딩
    person_encoder = create_unique_person_id(merged_data)
    merged_data = encode_writer_director(merged_data, person_encoder)
    merged_data = encode_genre(merged_data)

    return merged_data

if __name__ == "__main__":
    DATA_PATH = "/path/to/data"
    processed_data = main(DATA_PATH)
    print(processed_data.head())