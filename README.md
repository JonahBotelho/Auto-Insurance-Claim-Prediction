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
The Decision Tree preset has maximum depth 5 and uses the original 18 features
for both classification and Ridge. To reproduce submission 1 closely, select it
with 50 samples per leaf, threshold bounds both 0.25, 20% evaluation, five CV folds,
and oversampling set to 0.

Classifier training uniformly duplicates positive rows to target 15% claims
by default. Each CV fit samples only its training rows; Ridge uses the original
positive rows. Prediction corrects classifier probabilities back toward the
original claim rate before thresholding and weighting severity. This correction
is an approximation. Targets below the existing claim rate leave the rows unchanged.

Training reserves a stratified 38% holdout and selects a claim-probability
threshold from 0.10–0.20 inclusive in steps of 0.01 (11 candidates) using five-fold
cross-validation on the remaining 62%. The holdout does not influence threshold
selection. Training reports CV MAE, training MAE on the fitted training partition,
holdout MAE, classifier AUC, and a classification
report with true/false positive and negative counts, treating actual and predicted
`ClaimAmount > 0` as claims. It then refits on all training rows and saves
`models/two_stage_claim_model.joblib`, replacing the file if it exists.
Use `python -m src.train --n-splits 3` to change the fold count.

Run `python -m src.train --config` to adjust thresholds (up to three decimal
places), the evaluation percentage, model, leaf size, oversampling, profile-history
features and smoothing, severity method, Ridge alpha, amount multiplier and CV folds.
Histogram Gradient Boosting also prompts for iterations, learning rate, maximum
leaf nodes and L2 regularization. Threshold ranges advance by 0.01 and include
both bounds; equal bounds evaluate one threshold. Oversampling accepts 0-99
(0 disables it); the default is 15% of final classifier training rows.

Profile history groups exact feature profiles. Each fit builds statistics from
its own training partition and excludes each row's own label when fitting the
classifier. Profile-median severity uses matching positive training claims,
falling back to log-target Ridge for profiles without positive history.
Training MAE is in-sample and profile summaries include those training rows.

Enter accepts the displayed defaults, which match ordinary training;
`--n-splits` sets the initial fold count and can be overridden at the prompt.
Both partitions must contain zero and positive claims, and the training partition
must have at least one example of each class per CV fold. Infeasible settings stop
the run before fitting. After validation, review the summary and press Enter to
start training. Ctrl+C or closed input during configuration cancels before fitting.
Settings apply only to that run.

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

## Checkpoint submissions

See `submission.md` for the settings and recorded scores for files 1-10.
Submissions 1-6 use `src/`; submissions 7-10 use the separate LightGBM/CatBoost
workflow documented in `claimproj/README.md`. Its generator writes only files
7-10 to `data/regenerated/`, leaving the submitted CSVs in `data/` intact.

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
