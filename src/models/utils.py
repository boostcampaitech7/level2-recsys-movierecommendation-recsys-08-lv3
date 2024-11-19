import os
import random


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = '../../data'
rating_data_df = pd.read_csv(data_path + 'train_ratings.csv')

rating_data_df['rating'] = 1.0  # implicit feedback
rating_data_df.drop(['time'], axis=1, inplace=True) 
users = set(rating_data_df.loc[:, 'user'])
items = set(rating_data_df.loc[:, 'item'])

# 1. 유저별 데이터 처리를 위한 함수
def process_by_user(data, user_column, process_fn):
    """
    입력받는 데이터를 유저별로 그룹화시켜주는 함수
    Parameters:
    - data (DataFrame): 전체 데이터프레임
    - user_column (str): 사용자 ID 열 이름
    - process_fn (function): 각 사용자 데이터를 처리할 함수. process_fn(user_data, **kwargs) 형태로 호출됨.

    Returns:
    - processed_data (list): 각 사용자별 처리된 데이터 리스트
    """
    processed_data = []
    for _, user_data in data.groupby(user_column):
        processed_result = process_fn(user_data)
        processed_data.append(processed_result)
    return processed_data



# 2. 유저별 train_valid_split
# 유저 데이터를 train_test_split에 적용시키는 함수
def split_user_data(user_data, test_size, random_state):
    """
    단순 train_test_split 적용

    Parameters:
    - user_data (DataFrame): 특정 사용자의 데이터
    - test_size (float): 검증 데이터 비율
    - random_state (int): 랜덤 시드

    Returns:
    - train_data (DataFrame): 훈련 데이터
    - valid_data (DataFrame): 검증 데이터
    """
    return train_test_split(
            user_data, test_size=test_size, random_state=random_state
        )

# 최종 train_test_split 도출
def train_valid_split(data, user_column='user', test_size=0.2, random_state=42):
    """
    각 유저별로 test 및 valid 데이터셋으로 분해

    Parameters:
    - rating_data_df (DataFrame): 전체 데이터프레임
    - test_size (float): 검증 데이터 비율
    - random_state (int): 랜덤 시드

    Returns:
    - train_data (DataFrame): 훈련 데이터
    - valid_data (DataFrame): 검증 데이터
    """
    # process_by_user에 데이터를 대입하여 반환된 결과 도출
    split_results = process_by_user(data, user_column, split_user_data)     # 각 사용자의 train, valid 데이터 포함(by tuple) 
    
    # train, valid data를 각각 별도 튜플로 분리
    train_data_list, valid_data_list = zip(*split_results)
    train_data = pd.concat(train_data_list, axis=0).reset_index(drop=True)
    valid_data = pd.concat(valid_data_list, axis=0).reset_index(drop=True)
    
    return train_data, valid_data



# 3. 유저별 negative_sampling(Uniform Sampling)
# 개별 유저당 negative sampling
def sample_negative_items(user_data, items, num_negative, random_state):
    """
    Parameters:
    - user_data (DataFrame): 특정 사용자의 데이터
    - items (set): 전체 아이템 집합
    - num_negative (int): 생성할 부정 샘플 개수
    - random_state (int): 랜덤 시드

    Returns:
    - negative_samples (DataFrame): 부정 샘플 데이터프레임
    """
    rng = np.random.default_rng(random_state)   # 난수 생성기를 시드와 함께 초기화(재현성 보장)
    interacted_items = set(user_data['item'])   # 사용자가 이미 본 아이템 추출
    non_interacted_items = list(items - interacted_items)   # 사용자가 보지 않은 아이템 추출
    
    # non_interacted_items의 갯수에 따라 negative sampling 추출하여 sampled_items에 저장
    if len(non_interacted_items) < num_negative:
        sampled_items = non_interacted_items
    else:
        sampled_items = rng.choice(non_interacted_items, size=num_negative, replace=False)  # replace: 중복없는 sampling
    
    return pd.DataFrame({
        'user': [user_data['user'].iloc[0]] * len(sampled_items),   # '사용자의 ID 복제' * 'negative sampling 갯수'
        'item': sampled_items,
        'rating': [0] * len(sampled_items)
    })


# 전체 데이터에서, 유저별 negative sampling 수행
def negative_sampling(rating_data_df, items, num_negative=50, random_state=42):
    """
    Parameters:
    - user_item_df (DataFrame, include user_id, item_id)
    - all_items (set)
    - k_samples (int) : negative samples for each user
    - random_state (int)

    Returns:
    - negative_samples (DataFrame) : contain user, item column
    """
    # 유저별 negative sampling 수행(for process_by_user)
    negative_samples_list = process_by_user(
        rating_data_df, 'user', sample_negative_items,
        items=items, num_negative=num_negative, random_state=random_state
    )
    
    return pd.concat (negative_samples_list, axis=0).reset_index(drop=True)
        