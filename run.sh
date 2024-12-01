
python main.py  -c config/config_baseline.yaml  -m DeepFM 
python main.py  -c config/config_baseline.yaml  -m EASE 
python main.py  -c config/config_baseline.yaml  -m SLIM


### 사용법 ###
# 실행 코드 : bash run.sh
# -c 는 필수사항이므로 무조건 적어야합니다.
# -m 은 모델의 약자로 DeepFM 또는 slim만 적어주세요
# config 풀더안에 각 모델의 파라미터 및 디바이스 설정이 있습니다
# basic 은 validation data를 만들지 않아 main.py 가 보기 좋지 않을수있습니다
# Predict False 는 오직 훈련만 진행합니다 (validation 에서 테스트를 진행)
# Predict True 는 output까지 제출을합니다.
