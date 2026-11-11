# Training configuration

## Run instructions

From the repository root, install the dependencies for submissions 1-6 and run:

```powershell
python -m pip install -r src/requirements.txt
python -m src.train --config
python -m src.predict --output data/regenerated_1_2_1.csv
```

Follow the prompts to choose the threshold range, evaluation percentage,
model, minimum leaf size where applicable, and number of cross-validation folds.

Press Enter to accept a default. After reviewing the settings, press Enter to
start training or Ctrl+C to cancel. Repeat for files 1-6 with their settings
below and a different output filename. Each training run replaces the saved
model, so predict immediately after each run. Training selects a threshold by
five-fold CV, reports training-partition/holdout MAE, and refits on all rows.

For submissions 7-10, use the separate workflow:

```powershell
python -m pip install -r claimproj/requirements.txt
python -m claimproj.gen_submissions
```

It writes only files 7-10 and their full-training MAE to `data/regenerated/`.
See `claimproj/README.md` for exact settings and metric definitions. Generated
outputs may differ with dependency versions; original submitted files remain
in `data/`. The original trained models for files 2-10 were not retained.

The included saved Decision Tree can generate file 1 without retraining:
`python -m src.predict --output data/regenerated_1_2_1.csv`.
Run this before another training run replaces that model.

Scores below are historical records. For files 1-6, training MAE refers to the
fitted training partition before the full-data refit. For files 7-10, the old
values labeled training MAE were actually CV MAE. The corrected training MAEs
below were measured on all training rows using the generator, whose predictions
matched the submitted CSVs within floating-point precision. These scores use
different training partitions from files 1-6 and should not be ranked together.
The verified run used Python 3.12, LightGBM 4.7.0, CatBoost 1.2.10,
scikit-learn 1.9.1, pandas 3.0.1, and NumPy 2.4.2.
`submission_comments.txt` contains the corresponding upload comments.

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

- Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.55
- Maximum threshold: 0.55
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- CV MAE: 102.02
- Training MAE (full training set): 95.5046
- Holdout MAE: 103.65

### 1_2_8

- Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.50
- Maximum threshold: 0.50
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- CV MAE: 102.35
- Training MAE (full training set): 95.2502
- Holdout MAE: 103.99

### 1_2_9

- Model: LightGBM (with group claim-history features)
- Minimum threshold: 0.60
- Maximum threshold: 0.60
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100
- CV MAE: 102.65
- Training MAE (full training set): 99.8865
- Holdout MAE: 104.66

### 1_2_10

- Model: LightGBM + CatBoost blend (with group claim-history features)
- Minimum threshold: 0.50
- Maximum threshold: 0.50
- Evaluation/training split: 20% / 80%
- Minimum samples per leaf: 100 (LightGBM; CatBoost uses depth-4 trees with no per-leaf minimum)
- Amount scale: 0.8
- CV MAE: 102.47
- Training MAE (full training set): 94.2232
- Holdout MAE: 103.32
