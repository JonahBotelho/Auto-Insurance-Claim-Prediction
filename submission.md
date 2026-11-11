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
- Minimum samples per leaf: 50
- Training MAE: 101.80
- Holdout MAE: 108.50

### 1_2_2

- Model: Logistic Regression + `feature1` polynomial terms + Ridge
- Minimum threshold: 0.05
- Maximum threshold: 0.20
- Evaluation/training split: 20% / 80%
- Training MAE: 101.77
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

Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.55
- Maximum threshold: 0.55
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- Training MAE: 102.02
- Holdout MAE: 103.65
### 1_2_8

Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.50
- Maximum threshold: 0.50
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- Training MAE: 102.35
- Holdout MAE: 103.99
### 1_2_9

Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.60
- Maximum threshold: 0.60
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- Training MAE: 102.65
- Holdout MAE: 104.66
### 1_2_10

Model: LightGBM + CatBoost blend (with group claim-history features)
- Minimum threshold: 0.50
- Maximum threshold: 0.50
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100 (LightGBM; CatBoost uses depth-4 trees with no per-leaf minimum)
- Training MAE: 102.47
- Holdout MAE: 103.32