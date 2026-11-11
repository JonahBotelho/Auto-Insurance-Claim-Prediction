# COMP4983-Final-Project

Predict insurance claim amounts (`ClaimAmount`) from 18 anonymized customer and
vehicle features. `rowIndex` identifies rows and is excluded from model inputs.

Input datasets and the submission template are in `data/`:
`trainingset.csv`, `testset.csv`, and `submission.csv`.

## Training and prediction

Use Python 3.10 or newer and run commands from the repository folder:

```powershell
python -m pip install -r src/requirements.txt
python -m src.train
python -m src.predict
```

The two-stage model combines a claim-probability classifier (logistic regression
by default) with Ridge regression on `log1p` positive claim amounts. Preprocessing
scales numeric features and one-hot encodes categorical features, grouping rare
categories. Histogram Gradient Boosting instead uses native categorical features
for its classifier; Ridge retains the shared preprocessing. The workflow also
derives squared and cubed terms from `feature1`.

Training reserves a stratified 30% holdout and selects a claim-probability
threshold from 0.10–0.25 inclusive in steps of 0.01 (16 candidates) using five-fold
cross-validation on the remaining 70%. The holdout does not influence threshold
selection. Training reports CV MAE, holdout MAE, classifier AUC, and a classification
report, then refits on all training rows and saves
`models/two_stage_claim_model.joblib`, replacing the file if it exists.
Use `python -m src.train --n-splits 3` to change the fold count.

Run `python -m src.train --config` for terminal prompts to adjust:

- Minimum and maximum thresholds: 0.00–1.00 in 0.01 steps, including both bounds.
  Minimum must not exceed maximum; equal bounds evaluate one threshold.
- Evaluation percentage: a whole number from 1–99; the remainder is for training.
- Model: logistic regression, Random Forest, Extra Trees, or Histogram Gradient
  Boosting, each paired with Ridge for claim severity.
- CV folds: a whole number of at least 2.

Models use fixed presets, not tuned configurations. Tree ensembles can take
longer to train: threshold selection refits both stages for every threshold/fold.

Enter accepts the displayed defaults, which match ordinary training;
`--n-splits` sets the initial fold count and can be overridden at the prompt.
Both partitions must contain zero and positive claims, and the training partition
must have at least one example of each class per CV fold. Infeasible settings stop
the run before fitting. After validation, review the summary and enter `y` or
`yes` to train; any other response cancels. Ctrl+C or closed input during
configuration also cancels before fitting. Settings apply only to that run.

Prediction loads the saved model and writes nonnegative, probability-weighted
claim amounts, with zero below the selected threshold. The default output
**overwrites `data/submission.csv`**, preserving test-set row order and the
`rowIndex,ClaimAmount` columns. It prints the non-zero claim count, total rows,
and non-zero percentage to four decimal places. Override paths as needed:

```powershell
python -m src.predict --model-path models/two_stage_claim_model.joblib --input data/testset.csv --output data/predictions.csv
```

This is a prototype: reported validation scores estimate performance on held-out
training data; test-set accuracy is unknown without its claim labels.

## Analysis

`analysis/` contains exploratory scripts; `src/` contains the prediction workflow.
Install the analysis dependencies separately:

```powershell
python -m pip install -r analysis/requirements.txt
```

### Basic linear regression

Run `python analysis/linear_regression.py` for a baseline using all 18 features
without preprocessing. Its MAE is measured on the same rows used to fit the
model, so it does not measure performance on unseen data.

### Correlation analysis

Generate training-data correlation plots:

```powershell
python analysis/correlations.py
```

Open `analysis/output/index.html` to browse the 171 scatterplots and correlation
results, heatmaps, and coefficient CSVs. To share the gallery, include the entire
`analysis/output/` folder. The script supports `--input` and `--output` paths.
