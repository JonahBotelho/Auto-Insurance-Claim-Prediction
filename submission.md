# Training configuration

The final submission model is dashboard run **1200001 / 1200002**
(`configs/claim_history/em_timeline_sibling_rich_pricing_ensemble.json`,
pre-trained at `models/em_timeline_sibling_rich_pricing_ensemble.joblib`).

For predict-only scoring in a local browser UI (select `competitionset.csv`
and save `predictedclaimamount.csv` to a directory of your choice),
double-click `run_predictor.bat` on Windows (auto-creates `.venv` and installs
`src/predict_requirements.txt` on a fresh machine) or run:

```powershell
python -m pip install -r src/predict_requirements.txt
python -m src.predict_ui
```

Headless CLI equivalent:

```powershell
python -m src.predict --model-path models/em_timeline_sibling_rich_pricing_ensemble.joblib --input competitionset.csv --output predictedclaimamount.csv
```

See [final competition submission](#final-competition-submission) for full CLI
training, prediction, and benchmark details.

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
in `data/checkpoint1/`. The original trained models for files 2-10 were not retained.

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

## Checkpoint 2 submissions

Install dependencies and generate all ten files from the repository root:

```powershell
python -m pip install -r src/requirements.txt
python -m src.generate_checkpoint2_candidates --seed 42
```

Use `--submission 2_2_9` to generate one file. CSVs go to `data/checkpoint2/`,
training metrics to `data/checkpoint2 metrics/`, and upload comments to
`data/checkpoint2_upload_comments.txt`. A custom `--output-dir "path with spaces"`
receives its CSVs, metrics and comments together. Existing destination files are replaced.

To run submission 1 on another CSV, use its full file path (including the filename):

```powershell
python -m src.generate_checkpoint2_candidates --submission 2_2_1 --seed 42 --input "C:\path to data\new_testset.csv" --output-dir "C:\path to results"
```

The new CSV needs `rowIndex` and the same 18 feature columns as `data/testset.csv`;
claim labels are unnecessary. Training continues to use `data/trainingset.csv`.

All final models use the full **70,000 training rows (3,335 positive claims)** and
seed **42**. Training MAE is in-sample; it does not estimate test-set accuracy.
Validation means below use five seeds and the evaluation splits shown.
Improvement averages each seed's percentage MAE reduction against all-zero predictions.

| Submission | Eval % | Seeds | Mean MAE | All-zero MAE | MAE reduction | Improvement | Precision | Recall | Mean F1 | Aggregate | Threshold |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2_2_1 | 5% | 41–45 | 84.1730 | 91.7301 | 7.5572 | 7.96% | 51.48% | 28.86% | 0.3687 | 22.7866 | 0.470 |
| 2_2_2 | 28% | 41–45 | 97.1558 | 100.3546 | 3.1987 | 3.22% | 54.81% | 34.28% | 0.4215 | 18.2315 | 0.420 |
| 2_2_3 | 25% | 41–45 | 97.7759 | 101.5314 | 3.7555 | 3.72% | 54.94% | 33.02% | 0.4123 | 18.9064 | 0.440 |
| 2_2_4 | 25% | 41–45 | 98.8756 | 101.5314 | 2.6557 | 2.62% | 45.10% | 54.03% | 0.4915 | 18.5607 | 0.125 |
| 2_2_5 | 20% | 41–45 | 99.6769 | 102.5418 | 2.8648 | 2.74% | 50.00% | 43.96% | 0.4678 | 18.1863 | 0.340 |
| 2_2_6 | 20% | 41–45 | 100.0412 | 102.5418 | 2.5005 | 2.39% | 45.42% | 54.81% | 0.4968 | 18.1292 | 0.190 |
| 2_2_7 | 25% | 42–46 | 100.3675 | 104.0996 | 3.7321 | 3.55% | 49.60% | 38.56% | 0.4338 | 18.9425 | 0.340 |
| 2_2_8 | 25% | 42–46 | 100.6874 | 104.0996 | 3.4122 | 3.25% | 46.68% | 47.72% | 0.4717 | 19.3554 | 0.280 |
| 2_2_9 | 25% | 42–46 | 100.7687 | 104.0996 | 3.3309 | 3.18% | 45.97% | 51.70% | 0.4866 | 19.6294 | 0.220 |
| 2_2_10 | 25% | 42–46 | 100.9740 | 104.0996 | 3.1256 | 2.99% | 45.67% | 53.02% | 0.4906 | 19.3410 | 0.160 |

The screenshot shows 0.200 for the former `2_2_8`; the saved configuration and
pictured scores correspond to **0.220**, retained here as confirmed. The final
row shows 0.240, but its pictured scores reproduce at the recorded **0.160**;
0.240 gives different scores. This selection preserves the pictured scores
and recorded 0.160 configuration for `2_2_10`.

### 2_2_1

- Previous name: Historical 95/5 — 3 seeds, Historical 95/5 — 5 seeds
- Exact model settings: `configs/checkpoint2/2_2_1.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **103.110913**.
- Per validation seed: 66,500 training rows / 3,500 held-out rows.

### 2_2_2

- Previous name: Tuned R11
- Exact model settings: `configs/checkpoint2/2_2_2.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **94.951417**.
- Per validation seed: 50,399 training rows / 19,601 held-out rows.

### 2_2_3

- Previous name: Tuned R192
- Exact model settings: `configs/checkpoint2/2_2_3.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **95.786657**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

### 2_2_4

- Previous name: Tuned R314
- Exact model settings: `configs/checkpoint2/2_2_4.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **92.671840**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

### 2_2_5

- Previous name: Tuned R142
- Exact model settings: `configs/checkpoint2/2_2_5.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **93.201715**.
- Per validation seed: 56,000 training rows / 14,000 held-out rows.

### 2_2_6

- Previous name: Tuned R345
- Exact model settings: `configs/checkpoint2/2_2_6.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **92.790901**.
- Per validation seed: 56,000 training rows / 14,000 held-out rows.

### 2_2_7

- Previous name: 2_2_2
- Exact model settings: `configs/checkpoint2/2_2_7.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **92.344242**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

### 2_2_8

- Previous name: 2_2_5
- Exact model settings: `configs/checkpoint2/2_2_8.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **91.688114**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

### 2_2_9

- Previous name: 2_2_8
- Exact model settings: `configs/checkpoint2/2_2_9.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **91.556272**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

### 2_2_10

- Previous name: 2_2_10
- Exact model settings: `configs/checkpoint2/2_2_10.json`.
- Training data: all 70,000 rows (3,335 positive claims); seed 42; training MAE **91.589141**.
- Per validation seed: 52,500 training rows / 17,500 held-out rows.

## Final competition submission

The selected final submission model is dashboard run **1200001** (20% matched)
/ **1200002** (25% matched), both evaluating the same configuration
(`configs/claim_history/em_timeline_sibling_rich_pricing_ensemble.json`). It
extends previous leader **1190003 / 1190004** with complementary-exposure
sibling effective roster sizing in `EMReconciledVeto` (`sibling_effective_em`)
and multi-expert warm mixture-median pricing with a `$0.01` low-confidence
floor (`rich_ensemble_pricing`). The dashboard lists both runs under
**Sibling-roster EM and rich-ensemble pricing (seeds 41–45)**.

Install dependencies and train from the repository root.

```powershell
python -m pip install -r src/predict_requirements.txt
python -m src.train --config configs/claim_history/em_timeline_sibling_rich_pricing_ensemble.json --full-fit-only --model-path models/em_timeline_sibling_rich_pricing_ensemble.joblib
```

Press Enter at the confirmation prompt. Training uses seed 42 and all
70,000 rows from `data/trainingset.csv`. Use the saved model to score
`competitionset.csv` and write `predictedclaimamount.csv` (or replace
`--input` with `data/testset.csv`):

```powershell
python -m src.predict --model-path models/em_timeline_sibling_rich_pricing_ensemble.joblib --input competitionset.csv --output predictedclaimamount.csv
```

The input needs `rowIndex` and the same 18 feature columns as `data/testset.csv`.
The output contains `rowIndex` and `ClaimAmount` in input row order. To score
CSVs interactively offline without retraining, launch the predict-only local
browser UI with `python -m src.predict_ui` (or `run_predictor.bat` on Windows).

| Dashboard run | Holdout cohort | Split seeds | Mean F1 | Mean MAE |
| --- | --- | --- | ---: | ---: |
| **1200001** (selected) | 20% matched | 41–45 | **0.52346** | **91.92771** |
| **1200002** (selected) | 25% matched | 41–45 | **0.51741** | **91.04931** |
| *(5-fold OOF, all 70k rows)* | 5-fold CV | 42 | 0.50856 | **91.51996** |
| *(fresh confirmation)* | 20% matched | 46–50 | 0.52274 | 95.50189 |
| *(fresh confirmation)* | 25% matched | 46–50 | 0.50634 | 94.52112 |
| *(transfer)* | 25% mixed | 41–45 | 0.41504 | 95.08075 |
| *(transfer)* | 25% profile-disjoint | 41–45 | 0.15717 | 105.67332 |
| 1190003 (previous) | 20% matched | 41–45 | 0.52348 | 92.52804 |
| 1190004 (previous) | 25% matched | 41–45 | 0.51593 | 91.57719 |

Reproduce the five-seed matched holdout scores.

```powershell
python -m src.reproduce_timeline_models --config configs/claim_history/em_timeline_sibling_rich_pricing_ensemble.json --cohorts matched_20 matched_25
```

See the [benchmark report](results/em_timeline_benchmark.md) for earlier paired
comparisons and transfer results.
