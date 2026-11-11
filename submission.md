# Training configuration

## Run instructions

From the project folder, run:

```powershell
python -m src.train --config
```

Follow the prompts to choose the threshold range, evaluation percentage,
model, minimum leaf size where applicable, and number of cross-validation folds.

Press Enter to accept a default. After reviewing the settings, press Enter to
start training or Ctrl+C to cancel.

## Submission list

### 1_2_1

- Model: Decision Tree (maximum depth 5) + Ridge
- Minimum threshold: 0.05
- Maximum threshold: 0.25
- Evaluation/training split: 20% / 80%
- Holdout MAE: 108.50

### 1_2_2

- Model: Logistic Regression + `feature1` polynomial terms + Ridge
- Minimum threshold: 0.05
- Maximum threshold: 0.25
- Evaluation/training split: 20% / 80%
- Holdout MAE: 108.52

### 1_2_3

- Model: Logistic Regression + `feature1` polynomial terms + Ridge
- Evaluation/training split: 38% / 62%
- Minimum threshold: 1
- Maximum threshold: 1
- Training MAE: 103.33
- Holdout MAE: 102.59

### 1_2_4

- Model: Random Forest + Ridge
- Minimum threshold: 0.18
- Maximum threshold: 0.18
- Evaluation/training split: 38% / 62%
- Minimum samples per leaf: 5
- Training MAE: 98.42
- Holdout MAE: 104.39

### 1_2_5

- Model: Extra Trees + Ridge
- Minimum threshold: 0.18
- Maximum threshold: 0.18
- Evaluation/training split: 38% / 62%
- Minimum samples per leaf: 5
- Training MAE: 98.71
- Holdout MAE: 104.34

### 1_2_6

- Model: Extra Trees + Ridge
- Minimum threshold: 0.18
- Maximum threshold: 0.18
- Evaluation/training split: 38% / 62%
- Minimum samples per leaf: 15
- Training MAE: 102.16
- Holdout MAE: 102.78

### 1_2_7

### 1_2_8

### 1_2_9

### 1_2_10
