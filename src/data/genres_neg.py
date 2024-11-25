import pandas as pd
import numpy as np
from tqdm import tqdm
from data_loader import load_data

def calculate_user_genre_counts(user_item_genre_df):
    """
    유저별로 관측된 아이템의 각 장르의 수를 계산합니다.

    Args:
        user_item_genre_df (pd.DataFrame): 유저-아이템-장르 데이터프레임 (컬럼: 'user', 'item', 'genre').

    Returns:
        dict: 유저별 장르별 아이템 수를 나타내는 딕셔너리.
    """
    # 유저별로 그룹화 후 장르별로 개수를 세기
    user_genre_counts = (
        user_item_genre_df
        .groupby(['user', 'genre'])['item']
        .count()
        .unstack(fill_value=0)
        .to_dict('index')
    )
    
    return user_genre_counts

def calculate_user_genre_probabilities(user_genre_counts, all_genres, smoothing_value=1):
    """
    유저별 장르 카운트에서 없는 장르를 추가하고, 모든 장르에 작은 값을 더해 확률 분포를 계산합니다.

    Args:
        user_genre_counts (dict): 유저별 장르별 아이템 카운트.
        all_genres (list): 모든 가능한 장르 리스트.
        smoothing_value (int or float): 없는 장르에 추가할 작은 값 (기본값: 1).

    Returns:
        dict: 유저별 장르 선호 확률 분포.
    """
    user_genre_probs = {}

    for user, genre_counts in user_genre_counts.items():
        # 모든 장르를 포함한 초기화
        adjusted_counts = {genre: genre_counts.get(genre, 0) + smoothing_value for genre in all_genres}
        
        # 전체 카운트 합 계산
        total_count = sum(adjusted_counts.values())
        
        # 확률 계산
        genre_probs = {genre: count / total_count for genre, count in adjusted_counts.items()}
        user_genre_probs[user] = genre_probs

    return user_genre_probs
  
def label_encode_genres(train_with_genres):
  """
  문자열 장르를 숫자로 변환 (Label Encoding).
  
  Args:
      train_with_genres (pd.DataFrame): 사용자-아이템-장르 데이터프레임.

  Returns:
      tuple:
          - pd.DataFrame: Label Encoding된 데이터프레임.
          - dict: 장르 문자열과 숫자 간의 매핑 딕셔너리.
  """
  unique_genres = train_with_genres['genre'].unique()
  genre_to_label = {genre: idx for idx, genre in enumerate(unique_genres)}
  train_with_genres['genre'] = train_with_genres['genre'].map(genre_to_label)
  return train_with_genres, genre_to_label
  
def filter_low_prob_genres(user_genre_probs, threshold=0.05):
    """
    각 유저별로 prob 값이 threshold 이하인 장르만 필터링합니다.

    Args:
        user_genre_probs (dict): 유저별 장르 확률 분포.
        threshold (float): 확률 필터링 기준값.

    Returns:
        dict: 유저별 필터링된 장르 리스트.
    """
    filtered_genres = {
        user: [genre for genre, prob in genres.items() if prob <= threshold]
        for user, genres in user_genre_probs.items()
    }
    return filtered_genres  
    
def calculate_sampling_weights(filtered_genres, user_genre_probs, exponent=2):
    """
    Filtered Genres에 대한 확률을 기반으로 Negative Sampling 가중치를 계산합니다.

    Args:
        filtered_genres (dict): 각 유저별 필터링된 장르 리스트.
        user_genre_probs (dict): 유저별 장르 확률 분포.
        exponent (float): 반전된 확률에 적용할 지수 (기본값: 2).

    Returns:
        dict: 유저별 필터링된 장르의 가중치.
    """
    sampling_weights = {}

    for user, genres in filtered_genres.items():
        if not genres:  # 필터링된 장르가 없는 경우 무시
            continue

        # 필터링된 장르에 대한 확률만 추출 및 반전
        inverted_probs = {genre: (1 - user_genre_probs[user][genre]) ** exponent for genre in genres}
        
        # 정규화 (Optional: 확률 합이 1이 되도록)
        total_weight = sum(inverted_probs.values())
        normalized_weights = {genre: weight / total_weight for genre, weight in inverted_probs.items()}
        
        sampling_weights[user] = normalized_weights

    return sampling_weights


def merge_item_with_genres(train_rating, genre):
    """
    아이템 데이터와 장르 데이터를 병합합니다.

    Args:
        train_rating (pd.DataFrame): 사용자-아이템 평점 데이터프레임.
        genre (pd.DataFrame): 아이템-장르 데이터프레임.

    Returns:
        pd.DataFrame: 병합된 데이터프레임.
    """
    return pd.merge(train_rating, genre, on='item', how='left')


def calculate_sample_weights_for_user(user_genre_probs, filtered_genres, exponent=2):
    """
    유저별 샘플 가중치를 계산합니다.

    Args:
        user_genre_probs (dict): 유저별 장르 확률 분포.
        filtered_genres (dict): 유저별 필터링된 장르 리스트.
        exponent (float): 반전 확률에 적용할 지수 (기본값: 2).

    Returns:
        dict: 유저별 필터링된 장르의 가중치.
    """
    sample_weights = {}
    for user, genres in user_genre_probs.items():
        # Filtered genres만 반영
        user_weights = {}
        for genre_label, prob in genres.items():
            if genre_label in filtered_genres[user]:  # 낮은 확률의 장르만 고려
                # 가중치 계산: (1 - 확률)^exponent
                user_weights[genre_label] = (1 - prob) ** exponent
        sample_weights[user] = user_weights
    return sample_weights


def prepare_data_and_calculate_weights(
    train_rating, genre, prob_threshold=0.05, smoothing_value=0.001, exponent=3
):
    """
    데이터를 준비하고 샘플링 가중치를 계산합니다.

    Args:
        train_rating (pd.DataFrame): 사용자-아이템 평점 데이터프레임.
        genre (pd.DataFrame): 아이템-장르 데이터프레임.
        all_genre_labels (list): 모든 장르 레이블 리스트.
        prob_threshold (float): 확률 필터링 기준값 (기본값: 0.05).
        smoothing_value (float): smoothing 값 (기본값: 0.001).
        exponent (float): 샘플링 가중치 계산 시 지수 (기본값: 3).

    Returns:
        tuple:
            - dict: 유저별 장르 확률 분포.
            - dict: 유저별 샘플링 가중치.
    """
    # Step 1: 아이템 데이터와 장르 데이터를 병합
    user_item_genre_df = merge_item_with_genres(train_rating, genre)

    # Step 2: 문자열 장르를 숫자로 변환 (Label Encoding)
    user_item_genre_df, genre_mapping = label_encode_genres(user_item_genre_df)

    all_genre_labels = list(genre_mapping.values())

    # Step 3: 유저별 장르별 아이템 카운트 계산
    user_genre_counts = calculate_user_genre_counts(user_item_genre_df)

    # Step 4: 유저별 장르 확률 계산
    user_genre_probs = calculate_user_genre_probabilities(
        user_genre_counts, all_genre_labels, smoothing_value
    )

    # Step 5: 낮은 확률 장르 필터링
    filtered_genres = filter_low_prob_genres(user_genre_probs, prob_threshold)

    # Step 6: Negative Sampling 가중치 계산
    sampling_weights = calculate_sampling_weights(filtered_genres, user_genre_probs, exponent)

    return user_genre_probs, sampling_weights


data_path = "../../../data/train"
train_rating, _, _, _, genre, _ = load_data(data_path)

user_genre_probs, sampling_weights = prepare_data_and_calculate_weights(
    train_rating = train_rating,
    genre = genre,
    prob_threshold = 0.05,
    smoothing_value = 0.001,
    exponent = 3
)

def negative_sampling_from_train_ratings(train_ratings, sampling_weights, num_neg_samples=1, device="cuda"):

    negative_samples = []

    # 사용자별 Positive Samples 캐싱
    user_to_positive_samples = {
        user: set(train_ratings[train_ratings['user'] == user]['item'].unique())
        for user in train_ratings['user'].unique()
    }

    train_with_genres = merge_item_with_genres(train_rating, genre)

    # 장르별 아이템 캐싱
    genre_to_items = {
        genre: set(train_with_genres[train_with_genres['genre'] == genre]['item'].unique())
        for genre in train_with_genres['genre'].unique()
    }

    # 전체 아이템
    all_items = set(train_ratings['item'].unique())

    for user, group in tqdm(train_with_genres.groupby('user'), desc="Generating Negative Samples"):
        # 유저별 Positive Samples
        positive_samples_set = user_to_positive_samples.get(user, set())

        # 유저별 장르 가중치
        genre_weights = sampling_weights.get(user, {})
        if not genre_weights:
            print(f"User {user} - No sampling weights.")
            continue

        # 장르별 가중치를 리스트로 변환
        genres, weights = zip(*genre_weights.items())
        weights = np.array(weights)

        for _ in range(num_neg_samples):
            retry_limit = 10
            while retry_limit > 0:
                # 가중치에 따라 장르 샘플링
                sampled_genre = np.random.choice(genres, p=weights / weights.sum())

                # 샘플링된 장르에 해당하는 아이템 필터링
                candidate_items_set = genre_to_items.get(sampled_genre, set()) - positive_samples_set

                # 후보가 없으면 전체 아이템에서 샘플링
                if not candidate_items_set:
                    print(f"User {user} - No candidate items for genre '{sampled_genre}'. Fallback to all items.")
                    candidate_items_set = all_items - positive_samples_set

                if candidate_items_set:
                    sampled_item = np.random.choice(list(candidate_items_set))
                    negative_samples.append({'user': user, 'item': sampled_item, 'rating': 0})
                    break

                retry_limit -= 1

            if retry_limit == 0:
                print(f"User {user} - Retry limit reached. No Negative Sample generated.")

    return pd.DataFrame(negative_samples)


negative_samples = negative_sampling_from_train_ratings(
        train_ratings = train_rating,
        sampling_weights = sampling_weights,
        num_neg_samples = 50,
        device="cuda"  # "cuda" 또는 "cpu"
    )
    
combined_data = pd.concat([train_rating, negative_samples], ignore_index=True)
combined_data['rating'] = combined_data['rating'].fillna(1)
combined_data = combined_data.drop(columns=['time'])