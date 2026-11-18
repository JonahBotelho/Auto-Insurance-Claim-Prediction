# COMP4983-Final-Project

Predict insurance claim amounts (`ClaimAmount`) from 18 anonymized customer and
vehicle features. `rowIndex` identifies rows and is excluded from model inputs.

Input datasets and the submission template are in `data/`:
`trainingset.csv`, `testset.csv`, and `submission.csv`.

We evaluate claim amounts by **MAE** (lower is better) and claim detection by
**F1**, the harmonic mean of precision and recall. A claim is `ClaimAmount > 0`.

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
selection. Each fold fits once and scores all thresholds on the same predictions.
Training picks the lowest mean CV MAE among thresholds with mean claim F1 of at
least 0.10. If none qualify, it picks the highest F1 and breaks ties by MAE.
Use `--min-claim-f1` to change the minimum or set it to 0 for MAE-only selection.
A config with one fixed threshold uses that threshold. Training reports CV MAE
and F1, training MAE, holdout MAE and F1, classifier AUC, and a classification
report with true/false positive and negative counts, treating actual and predicted
`ClaimAmount > 0` as claims. It then refits on all training rows and saves
`models/two_stage_claim_model.joblib`, replacing the file if it exists.
Use `python -m src.train --n-splits 3` to change the fold count.

Run `python -m src.train --config` to adjust thresholds (up to three decimal
places), the evaluation percentage, model, leaf size, oversampling, profile-history
features and smoothing, severity method, Ridge alpha, amount multiplier and CV folds.
The last prompt sets the evaluation/model seed (default 42); `--seed 44` also
sets its initial value. It controls the holdout, CV folds and classifier.
Oversampling keeps its fixed sampling seed of 42 to match the metric study.
Histogram Gradient Boosting also prompts for iterations, learning rate, maximum
leaf nodes and L2 regularization. Threshold ranges advance by 0.01 and include
both bounds; equal bounds evaluate one threshold. Oversampling accepts 0-99
(0 disables it); the default is 15% of final classifier training rows.

To reproduce dashboard run **1338**, use `python -m src.train --config --seed 44`
and copy its dashboard config inputs. Settings: threshold bounds both **0.361**,
25% evaluation, Histogram Gradient Boosting, leaf size 50, 125 iterations,
learning rate 0.15, 9 leaf nodes, L2 5, 30% oversampling, profile history enabled,
smoothing 0.25, profile-median severity, Ridge alpha 0.01 and multiplier 1.4.
Keep seed **44** at the last prompt; the fixed threshold makes CV fold count
irrelevant to its holdout score. Expected holdout: MAE 94.1284, precision 47.40%,
recall 44.72%. Saving still refits on all rows, so those are evaluation scores.
For any dashboard result, choose **View → Download configuration**, place
`run-<id>.json` in the repository folder, then run:

```powershell
python -m src.train --config run-1517.json
```

This supports all dashboard classifiers, past-amount history, unweighted medians,
quantile severity and amount caps. LightGBM configs can set the minimum leaf
size with `classifier.leaf` (default 50). For CatBoost or XGBoost, first install
`src/optional_requirements.txt`. Review the summary and press Enter to train;
the saved model works with `src.predict`. Plain `--config` remains interactive.
In JSON, `weighted_profile` weights profile medians by probability;
`profile_median` uses the median directly.
Runs without retained score settings export documented defaults for those
missing values; their exact historical result cannot be recovered.

Two extended-history LightGBM configs use a 0.25 claim threshold and a 2,500
amount cap before the 0.9 multiplier. On 25% holdouts with seeds 41–45, the 45%
oversampling config had mean MAE 97.31 and F1 0.4927. The config with oversampling
disabled had mean MAE 97.36 and F1 0.4958. Run either with its saved evaluation seed:

```powershell
python -m src.train --config configs/claim_history/lightgbm_45_percent.json
python -m src.train --config configs/claim_history/lightgbm_no_oversampling.json
```

Profile history groups exact feature profiles. Each fit builds statistics from
its own training partition and excludes each row's own label when fitting the
classifier. With `extended_amount_history` enabled, the classifier also gets the
variance of past positive log amounts and the share recorded to whole cents.
Profile-median severity uses matching positive training claims,
falling back to log-target Ridge for profiles without positive history.
Training MAE is in-sample and profile summaries include those training rows.

Enter accepts the displayed defaults, which match ordinary training;
`--n-splits` sets the initial fold count and can be overridden at the prompt.
Both partitions must contain zero and positive claims, and the training partition
must have at least one example of each class per CV fold. Infeasible settings stop
the run before fitting. After validation, review the summary and press Enter to
start training. Ctrl+C or closed input during configuration cancels before fitting.
Settings apply only to that run.

Prediction loads the saved model and writes nonnegative claim amounts using the
configured severity rule, with zero below the selected threshold. The default output
**overwrites `data/submission.csv`**, preserving test-set row order and the
`rowIndex,ClaimAmount` columns. It prints the non-zero claim count, total rows,
and non-zero percentage to four decimal places. Override paths as needed:

```powershell
python -m src.predict --model-path models/two_stage_claim_model.joblib --input data/testset.csv --output data/predictions.csv
```

This is a prototype: reported validation scores estimate performance on held-out
training data; test-set accuracy is unknown without its claim labels.

## Checkpoint submissions

For checkpoint 2, run `python -m pip install -r src/requirements.txt` then
`python -m src.generate_checkpoint2_candidates --seed 42` from the extracted
`sourcecode_2_2.zip` folder. The archive includes source, frozen configs, input
data and selected submissions. Install dependencies locally; saved model files,
the historical dashboard and analysis caches are unnecessary for this generator.

See `submission.md` for both checkpoint selections and their recorded scores.
Checkpoint 1 submissions 1-6 use `src/`; submissions 7-10 use the separate LightGBM/CatBoost
workflow documented in `claimproj/README.md`. Its generator writes only files
7-10 to `data/regenerated/`, leaving the submitted CSVs in `data/checkpoint1/` intact.

The current checkpoint 2 selection is in `submission.md`, with frozen settings in
`configs/checkpoint2/`. Install `src/requirements.txt`, then run
`python -m src.generate_checkpoint2_candidates` to write `2_2_1.csv` through
`2_2_10.csv` to `data/checkpoint2/`, training metrics to `data/checkpoint2 metrics/`,
and `checkpoint2_upload_comments.txt` to `data/`.
Use `--submission 2_2_1` for one file. A custom `--output-dir "path with spaces"`
receives CSVs, metrics and comments together.
To predict a different CSV, supply its full file path (including the filename):

```powershell
python -m src.generate_checkpoint2_candidates --submission 2_2_1 --seed 42 --input "C:\path to data\new_testset.csv" --output-dir "C:\path to results"
```

The input must have `rowIndex` and the same 18 feature columns as `data/testset.csv`;
claim labels are unnecessary. Training still uses the supplied `data/trainingset.csv`.
The generator replaces the current `data/checkpoint2/2_2_*.csv` files when rerun. Keep source,
configs and documentation in the repository; package the source manually when submitting.

## Analysis

`analysis/` contains exploratory scripts; `src/` contains the prediction workflow.
Install the analysis dependencies separately:

```powershell
python -m pip install -r analysis/requirements.txt
```

### Model dashboard

Open `dashboard/index.html` for historical runs, both seed searches, fresh
historical evaluations, and all ten checkpoint 2 submissions.
Select **Extended claim history (seeds 41-45)** to see the two LightGBM results,
their per-seed metrics, and downloadable training configurations.
Select **Extended claim history tuning (500 runs)** for 500 operating points
scored on the same 25% holdouts with seeds 41–45. The source JSON includes a
comparison of the selected leaders and earlier baselines on seeds 301–305.
The dashboard opens on **Claim F1 tuning (500 runs)**, sorted by mean claim
F1. These results use seeds 41–45 only. Select **All sources** to include
historical scores. Rebuild the F1 study with `python -m analysis.tune_claim_f1`.
Matching operating points average unique seeds; source,
seed and coverage filters control comparisons. Mean MAE, matched all-zero mean
MAE, absolute/percentage improvement, precision, recall, mean F1 and aggregate are shown.
Mean F1 averages each seed's claim-detection F1 (0–1; higher is better),
with zero when precision and recall are both zero. It is sortable and included in exports.
The historical checkpoint source retains its original names and five 25% holdouts
(seeds 42–46); the current selection's names and cohorts are in `submission.md`.
Star rows to favourite their exact configuration and seed cohort; **Favourites only**
shows saved rows alongside the other filters. Bookmarks persist in browser local
storage for the same page/browser; if storage is blocked, they last for the session.
Clearing filters keeps bookmarks. Multiple operating points can be bookmarked per model.

The dashboard is a saved standalone page. Its historical results and optional
analysis caches are separate from the checkpoint 2 generator. Regenerate the
extended-history tuning scores with `python -m analysis.tune_claim_history --output analysis/output/claim_history_tuning.json --dashboard dashboard/index.html`
after installing `src/requirements.txt` and `src/optional_requirements.txt`.

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
