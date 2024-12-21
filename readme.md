# Movie Recommendation
## 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 태스크



<br/>

## Team

| icon | <img src="./img/user_icon_1.webp" alt="user_icon_1" style="zoom:100%;" /> | <img src="./img/user_icon_2.webp" alt="user_icon_2" style="zoom:100%;" /> | <img src="./img/user_icon_3.webp" alt="user_icon_3" style="zoom:100%;" /> | <img src="./img/user_icon_4.webp" alt="user_icon_4" style="zoom:100%;" /> | <img src="./img/user_icon_5.webp" alt="user_icon_5" style="zoom:100%;" /> | <img src="./img/user_icon_6.webp" alt="user_icon_6" style="zoom:100%;" /> |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 이름 |                            한선우                            |                            신승훈                            |                            이경민                            |                            김민준                            |                            박광진                            |                            김영찬                            |
| 담당 |                          Model                            |                          Model                            |                          Model                            |                          Model                            |                          Model                            |                          Model                            |
| 역할 |             Negative Sampling, GNN, SASRec               |         NCF, DeepFM                |             FM, EASE, MF                |             Negative Sampling, Recbole             |                   RBM with LSTM, NARM, RecVAE, SLIM, MultiVAE, Ensemble                    |          SASRec, S^3 Rec, BERTRec, NCF, Ensemble                    |




<br/>

## 폴더구조

![image-20241028223652317](./img/dir_img.png.png)



<br/>

## 프로젝트

### 목적

이 프로젝트는 사용자의 시청 기록과 영화에 대한 추가 메타데이터를 활용하여 사용자가 각 영화를 어떻게 평가할지 예측함으로써 사용자에게 영화를 추천하는 것을 목표로 합니다. 추천 시스템은 명시적 평점 대신 시청 순서와 같은 암시적 피드백을 기반으로 학습됩니다.

<br/>

### 데이터

- **train_ratings.csv**:  31,360명의 사용자(user)가 6,807개의 영화(item)에 대해 남긴 5,154,471건의 평점을 남긴 시간(time) 데이터 입니다.
- **Ml_item2attributes.json**: item과 genre의 mapping 데이터입니다.
- **titles.tsv**: 6,807개의 영화(item)에 대한 제목(title)정보를 담고 있는 메타데이터입니다.
- **years.tsv**: 6,799개의 영화(item)에 대한 개봉연도(year) 정보를 담고 있는 메타데이터입니다.
- **directors.tsv**: 5,905개의 영화(item)에 대한 감독(director) 정보를 담고 있는 메타데이터입니다.
- **genres.tsv**: 15,934개의 영화(item)에 대한 장르(genre) 정보를 담고 있는 메타데이터입니다.
- **writers.tsv**: 11,307개의 영화(item)에 대한 작가(writer) 정보를 담고 있는 메타데이터입니다.
<br/>

### 평가지표

**Recall@K**

- **Recall@K**는 는 사용자가 선호할 수 있는 아이템들 중에서 상위 K개에 실제로 포함된 아이템의 비율을 측정하는 지표입니다. 이 지표는 모델의 추천 성능을 평가하는 데 사용됩니다.

- **Recall@K**는 다음과 같이 정의됩니다:

  $$
  Recall@K= \frac{1}{|U|} \sum_{u \in U} \frac{| \{ i \in I_u \ | \ \text{rank}_u(i) \leq K \} |}{\min(K, |I_u|)}
  $$
  
  $$
  |U| : 총 사용자 수, \\ \\  I_u : 사용자가 상호작용한 아이템의 수, \\ \\ {rank}_u(i) : 사용자의 아이템에 대한 순위, \\ \\ K : 평가하려는 추천 목록의 상위 K개의 아이템, \\ \\
  $$



<br/>

### 협업 방법

- 구글 공유 폴더 (데이터 공유 용)
- GitHub (issue, pr 로 전체적인 흐름 관리)
- Notion (회의록, 보드로 프로젝트 관리)
- RecBole

<br/>


<br/>

### Install

python version: 3.11x

```
pip install -r requirements.txt
```



<br/>

### 코드 및 설명
- **`src`**
  - **`models/`**: 구현된 모델의 아키텍처 파일을 포함합니다. (예: UnifiedDeepFM, SLIMModel, EASE).
  - **`results`**: 학습 및 평가 결과가 저장되는 디렉토리입니다.
    - **`models/`**: 학습된 모델 가중치 저장.
    - **`logs/`**: 실행 및 학습 로그 저장.
    - **`predictions/`**: 평가 시 생성된 예측 결과 저장.
  - **`evaluate.py`**: 각 모델을 평가하는 함수들이 포함되어 있습니다.
  - **`preprocessing.py`**: 데이터 로드 및 데이터 전처리와 관련된 함수들이 포함되어 있습니다.
  - **`train.py`**: 각 모델의 학습 로직을 구현한 학습 함수가 포함되어 있습니다. (예: train_deepfm, train_slim, train_ease).
  - **`utils.py`**: 공통적으로 사용되는 유틸리티 함수들을 포함합니다.
- **`main.py`**: **Argparser**를 사용하여 실행 모드를 선택(train 또는 evaluate)하고, config.yaml 파일을 기반으로 전체 파이프라인을 실행합니다. 학습(train) 모드에서는 모델 학습과 저장을 수행하며, 평가(evaluate) 모드에서는 모델 평가 및 예측 결과를 저장합니다.



<br/>

### 최종 제출 결과

| Model      | 리더보드 Recall@10 (중간) | 리더보드 Recall@10 (최종)|
| ---------- | -------------- | ------------ |
| ease_slim_multivae_ensemble_optuna   | 0.1600          | 0.1601        |
| ease(lambda : 350)   | 0.1600           | 0.1596        |



<br/>
