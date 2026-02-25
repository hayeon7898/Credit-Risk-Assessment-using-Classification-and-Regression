# 💳 Credit Score Classification and Overdue Days Regression

신용정보 데이터셋을 이용한 신용점수 Classification과 연체일수 Regression

---

## 🏗️ 프로젝트 구조

```
📁 credit-score-classification/
├── 📁 data/
│   ├── raw_data.csv                  # 원본 데이터
│   └── preprocessed_data.pkl         # 전처리된 데이터
├── 📁 models/
│   └── 📁 01_classification/
│       └── best_lr_model.pkl         # Logistic Regression
├── 📁 notebooks/
│   ├── 📁 01_eda/
│   │   └── eda.ipynb
│   ├── 📁 02_classification/
│   │   ├── random_forest.ipynb
│   │   ├── xgboost.ipynb
│   │   ├── ft_transformer.ipynb
│   │   ├── logistic_regression.ipynb
│   │   └── ensemble.ipynb
│   └── 📁 03_regression/              
|── 📁 reports/
│   ├── 01_Proposal report.pdf
│   ├── 02_Progress report.pdf
│   └── 03_Final report.pdf
└── README.md
```

> ⚠️ `best_rf_model.pkl`, `best_xgb_model.pkl`, `best_ft_model.pkl`은 GitHub 용량 제한(100MB)을 초과하여 제외됨

---

## 📌 Task 1: 신용등급 분류 (Classification) 프로젝트 개요

### 문제 정의

금융기관은 고객의 대출 상환 가능성을 정확히 예측함으로써 신용 리스크를 최소화하고, 동시에 고객에게 맞춤형 금융 서비스를 제공해야 하는 중요한 과제를 안고 있다. 그러나 각 고객의 재무 상황, 소득 구조, 지출 습관, 과거 상환 이력 등은 매우 다양하며, 단순히 신용등급이나 제한된 재무 지표만으로는 연체 위험을 충분히 평가하기 어렵다.

이러한 복잡한 환경에서는 전통적인 점수 기반 신용평가 방식만으로는 한계가 존재하며, 보다 정밀하고 데이터 기반의 분석이 요구된다. 이에 본 프로젝트에서는 금융기관이 합리적으로 대출 승인 여부와 한도를 결정할 수 있도록, 머신러닝 기반의 분류(Classification) 모델을 개발하였다.

개인 고객의 금융 데이터를 기반으로 신용 점수를 세 단계(Good/Standard/Bad)로 분류하여 고객의 연체 위험을 정량적으로 평가한다. 이를 통해 금융기관은 대출 승인 여부와 한도를 보다 합리적으로 결정할 수 있다.

| 항목       | 내용                                   |
| ---------- | -------------------------------------- |
| **목표**   | 고객 데이터 기반 신용등급 3-class 분류 |
| **데이터** | 96,696개 샘플, 24개 피처               |
| **타겟**   | 0: Bad, 1: Standard, 2: Good           |

---

## 🤖 사용 모델

### 1. Logistic Regression

- 선형 분류 모델로 baseline 성능 확인
- StandardScaler 적용 (필수)
- L1/L2 정규화 및 Optuna 하이퍼파라미터 튜닝

### 2. Random Forest

- 다수의 결정트리를 결합하여 변수 간 비선형 관계와 복잡한 상호작용 학습
- 피처 엔지니어링 (12개 신규 피처 생성)
- Feature Selection (상위 20개 피처)
- Optuna 하이퍼파라미터 튜닝

### 3. XGBoost

- 학습 과정에서 잔여 오차를 반복적으로 보정(Boosting)하여 예측 정확도 향상
- 과적합(overfitting) 방지를 위한 정규화 적용
- Optuna 하이퍼파라미터 튜닝

### 4. FT-Transformer

- 표형(tabular) 데이터에 특화된 Transformer 모델
- 기존 트리 기반 모델이 포착하기 어려운 고차원적 피처 상호작용과 비선형 패턴 학습
- 순수 PyTorch로 직접 구현
- Label Smoothing, Cosine Annealing, Gradient Clipping 적용

### 5. Ensemble (Soft Voting)

- RF + XGBoost + FT-Transformer 예측 확률 결합
- 최적 가중치 자동 탐색

---

## 📊 성능 비교

| Model | Accuracy | F1 Score (macro) |
|-------|:--------:|:----------------:|
| Logistic Regression | 0.6543 | 0.6489 |
| Random Forest | 0.7967 | 0.7850 |
| **🏆XGBoost** | **0.8296** | **0.8239** |
| FT-Transformer | 0.7347 | 0.7179 |
| Ensemble | 0.8199 | 0.8131 |

### 결과 분석

- **XGBoost**가 단일 모델 중 가장 높은 성능 (Acc: 81.33%)
- **Ensemble**이 전체 최고 성능 달성 (Acc: 81.99%, F1: 81.31%)
- **FT-Transformer**는 Tabular 데이터 특성상 트리 모델 대비 낮은 성능
- **Logistic Regression**은 선형 모델의 한계로 baseline 수준

---

## 🔧 성능 개선 기법

| 기법              | 설명                                     | 적용 모델 |
| ----------------- | ---------------------------------------- | --------- |
| 피처 엔지니어링   | 비율, 복합 피처 생성 (debt_to_income 등) | RF        |
| Feature Selection | 중요도 기반 상위 20개 피처 선택          | RF        |
| Optuna 튜닝       | 베이지안 최적화 기반 하이퍼파라미터 탐색 | All       |
| Label Smoothing   | 과적합 방지, 일반화 성능 향상            | FT        |
| Cosine Annealing  | 학습률 스케줄링                          | FT        |
| Gradient Clipping | 학습 안정화                              | FT        |
| Soft Voting       | 예측 확률 가중 평균 앙상블               | Ensemble  |

---

## 🚀 실행 방법

### 환경 설정

```bash
pip install pandas numpy scikit-learn xgboost torch optuna
```

### 1. 데이터 로드

```python
import pickle

with open('data/preprocessed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

X_train = data_dict['X_train_clf']
X_test = data_dict['X_test_clf']
y_train = data_dict['y_train_clf']
y_test = data_dict['y_test_clf']
```

### 2. 모델 로드 및 예측

```python
# XGBoost 예시
with open('models/01_classification/best_xgb_model.pkl', 'rb') as f:
    xgb_dict = pickle.load(f)

model = xgb_dict['model']
predictions = model.predict(X_test)
```

### 3. FT-Transformer 로드

```python
import torch

with open('models/01_classification/best_ft_model.pkl', 'rb') as f:
    ft_dict = pickle.load(f)

# 모델 재구성 후 state_dict 로드
model.load_state_dict(ft_dict['model_state'])
scaler = ft_dict['scaler']

# 예측
X_test_scaled = scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled)
predictions = model(X_test_tensor).argmax(dim=1)
```

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
torch>=1.10.0
optuna>=3.0.0
```

---

## 📝 Notes

- 본 프로젝트는 Google Colab 환경에서 개발됨
- FT-Transformer는 GPU 환경 권장
- Tabular 데이터에서는 일반적으로 트리 기반 모델(RF, XGBoost)이 딥러닝보다 우수한 성능을 보임
- 앙상블을 통해 개별 모델의 장점을 결합하여 최고 성능 달성

---

## 👤 Author

- GitHub: [@hayeon7898](https://github.com/hayeon7898) : EDA & Classification
- GitHub: [@seoyuncho](https://github.com/seoyuncho) : EDA & Regression

---
