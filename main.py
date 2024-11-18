from data.data_merge import data_merge

data_path = "./../../data/train/"

# 데이터 로드 및 병합
merged_data = data_merge(data_path)

print(merged_data.head())
