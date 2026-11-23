# COMP4983-Final-Project

Predict insurance claim amounts (`ClaimAmount`) from 18 anonymized customer and
vehicle features. `rowIndex` identifies rows and is excluded from model inputs.

Input datasets and the submission template are in `data/`:
`trainingset.csv`, `testset.csv`, and `submission.csv`.

We evaluate claim amounts by **MAE** (lower is better) and claim detection by
**F1**, the harmonic mean of precision and recall. A claim is `ClaimAmount > 0`.

See the [claim model handoff](CLAIM_MODEL_HANDOFF.md) for verified results,
competition setup, and directions for future experiments.

## Final competition prediction

The selected submission candidate is dashboard run **1190003**. Its
Bonus-Malus timeline uses exact successor links, competing discounts, and
conditional payout quotes for cold profiles. It averaged **0.52348 F1 /
92.52804 MAE** on 20% matched holdouts across seeds 41–45.

Install dependencies and train from the repository root.

```powershell
python -m pip install -r src/optional_requirements.txt
python -m src.train --config configs/claim_history/em_timeline_discount_quantile_ensemble.json --full-fit-only --model-path models/submission_1190003.joblib
```

Press Enter at the confirmation prompt. Training uses seed 42 and all
70,000 rows from `data/trainingset.csv`. Use the saved model to score the
test CSV. For a new competition CSV, replace the `--input` path.

```powershell
python -m src.predict --model-path models/submission_1190003.joblib --input data/testset.csv --output submission_1190003.csv
```

The input needs `rowIndex` and the same 18 feature columns as `data/testset.csv`.
The output contains `rowIndex` and `ClaimAmount` in input row order.

Reproduce the five-seed matched holdout scores.

```powershell
python -m src.reproduce_timeline_models --config configs/claim_history/em_timeline_discount_quantile_ensemble.json --cohorts matched_20 matched_25
```

See [submission details](submission.md#final-competition-submission) for the
25% benchmark and the [benchmark report](results/em_timeline_benchmark.md)
for transfer results.

The earlier `configs/claim_history/em_timeline_adaptive_ensemble.json` adapts
ranking to batch retention and caps quotes at observed payout support. On native
25% matched holdouts (seeds 41–45), F1 rose from **0.51401 to 0.51550** and
MAE moved from **91.62561 to 91.62271**. Fresh seeds 46–50 also improved
from **0.50396 to 0.50551 F1** at 25%. Use a separate model path to train and
predict with this configuration.

See the [four claim model screens](results/claim_avenue_screens.md) for the
cached vote-pattern, pricing, and cold timeline comparisons.
The [follow-up screens](results/claim_avenue_followup_screens.md) cover
warm counts, quote adjustments, payout multiplicity, and one-to-one links.

Earlier deployable alternatives (same `src.train --config` and `src.predict` workflow):

- `configs/claim_history/count_retrieval_bm_timeline_ensemble.json` (**0.52064 F1 /
  93.03928 MAE** at 20%, **0.51241 F1** at 25%, **0.41303 F1** on 25% mixed, **0.15702 F1**
  on 25% disjoint) and lighter `configs/claim_history/pairwise_bm_timeline_ensemble.json`
  (**0.51671 F1 / 93.03937 MAE** at 20%, **0.51049 F1 / 92.11566 MAE** at 25%). Both
  reproduce natively via `python -m src.reproduce_timeline_models --cohorts matched_20 matched_25 mixed_25 disjoint_25`.
- `configs/claim_history/pairwise_payout_ensemble.json` (**0.51018 F1 / 93.04831 MAE** at 20%,
  **95.33733 MAE** on 25% mixed) and `configs/claim_history/payout_source_local_tail_ensemble.json`
  (requires only `src/requirements.txt`).

## Reproduce the historical F1 benchmark

From the repository root, run `python -m pip install -r src/requirements.txt`
and `python -m src.reproduce_claim_benchmark`. The five-seed mean on 20% holdouts is
**0.520785 F1 / 93.02023 MAE** for `review_additions_bm_timeline` (and **0.51468628 F1 /
93.02916839 MAE** for `review_additions`). Add `--predictions claim_benchmark.csv.gz` to export the per-row predictions.

The benchmark uses the original `data/trainingset.csv` and the frozen upstream
anchor scores and baseline quotes in `src/reference/`. It refits the downstream
models and scores the held-out rows. This command reproduces the historical
benchmark and requires no `analysis/` files. The reference manifest records the
package versions used for verification.

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
For a single-model dashboard result, choose **View → Download configuration**, place
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

To use tune 264 with CatBoost median pricing, install
`src/optional_requirements.txt` and run
`python -m src.train --config configs/claim_history/tune_264_median_payout.json`.
It keeps tune 264's claim flags and reprices matching profiles with positive
history. The amount model fits on training rows with at least one other positive
match and uses the same-profile payouts and zeros, excluding the current row.

Run `python -m src.train --config configs/claim_history/tune_264_aggregate_payout.json`
to price claims using the relationship between scaled totals and whole-cent payouts.
For the experimental detector with that pricer, use
`configs/claim_history/profile_history_ensemble.json`. The ensemble fits a
calibrator on out-of-fold scores from the histogram, count and payout models.
The count model uses a 100,000-row reference population. Probabilities can vary
slightly with the rows scored in the same batch.

Run `python -m src.train --config configs/claim_history/template_claim_ensemble.json`
for the payout-template detector with profile-relative pricing. It chooses a
cutoff from expected F1 for the scoring batch and caps prices at the fitting
positives' 97th percentile, with a minimum cap of 2,250.

With `src/optional_requirements.txt` installed, run
`python -m src.train --config configs/claim_history/payout_source_claim_ensemble.json`
for the opt-in variant that estimates separate whole-cent and scaled payout
probabilities. It uses a 20% holdout for the training report.

Use `configs/claim_history/payout_source_tail_ensemble.json` with the same
training command to fit the payout floor from the lower tail of each source.
The detector produces the same claim flags.
Use `configs/claim_history/payout_source_local_tail_ensemble.json` to also reduce
unsupported low quotes on low-risk profiles. On 20% and 25% holdouts with seeds
41–45, mean MAE fell by 0.0080 and 0.0073 with identical claim flags.

Use `configs/claim_history/payout_source_cold_profile_ensemble.json` to add
CatBoost scoring for batches with no exact profiles in the fitting data.
It uses rating features and counts of matching rows in the scoring batch.
Use `configs/claim_history/payout_source_cold_profile_tail_ensemble.json` to
also cap low-confidence quotes in those batches at the observed 2nd-percentile
positive payout, after applying the configured amount multiplier.

Use `configs/claim_history/payout_source_cold_exposure_hazard.json` to require
exposure-hazard confirmation and price low-confidence claims by payout source
when the scoring batch has entirely unfamiliar profiles.

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

### Final candidate search

Fit the count-retrieval timeline model once per holdout and cache its per-row
outputs, then score pricing and flag-volume variants by mean F1, then mean MAE:

```powershell
python -m analysis.cache_final_model_outputs --cohort matched_20 --seeds 41 42 43 44 45
python -m analysis.evaluate_final_candidates --cohort matched_20
```

Writing $1 instead of the quoted price for flagged rows with risk below 0.45 kept F1
and lowered mean MAE by 0.28 to 0.49 on 20% and 25% holdouts and fresh seeds 46–50.

After caching `matched_20` and `matched_25`, run `python -m analysis.publish_final_candidates`
and select **Near-zero pricing for low-confidence flags (seeds 41–45)** in the dashboard.
It shows the rule and a rerun control for each holdout size. The rule is not yet
applied by `src.predict`.

Run `python -m analysis.evaluate_and_publish_em_timeline` (after caching `matched_20` and
`matched_25`) to reproduce and publish **EM reconciled veto + BM timeline v3 & calibrated L1
pricing (seeds 41–45)** (runs `1180001–1180004`), which reached **0.52301 F1 / 92.585 MAE** on
20% holdouts and **0.51424 F1 / 91.640 MAE** on 25% holdouts. Run `1180002` is deployable via
`configs/claim_history/em_reconciled_bm_v3_l1_ensemble.json`.

### Known-profile reranking archive

Select **Known-profile reranking** in the dashboard to review the matched
development scores for seeds 41–45. Neither challenger passed the development
gate, so tune 264 was retained. The split manifest, paired scores and row-level
predictions are archived in `analysis/output/known_profile_reranking_41_45/`.
The experimental training and prediction code was removed.

### Fixed-flag severity archive

Select **Fixed-flag severity** in the dashboard for the matched baseline and
best challenger on seeds 41–45. The study retained tune 264 after development.
Paired scores, row-level predictions and the split manifest are archived in
`analysis/output/fixed_flag_severity/`.

### No positive claim history archive

Select **No positive claim history** in the dashboard for tune 264 and the
best flag policy on matched seeds 41–45. All six specialist policies fell
short of the development gate. Results and the split manifest are archived
in `analysis/output/no_positive_history/`.

### Historical-size coverage comparison

Run `python -m analysis.evaluate_known_profile_coverage --output-dir analysis/output/known_profile_coverage_rerun`
to compare tune 264 with broader flagging rules on the historical holdouts.
Use a fresh output directory. The study retained tune 264. The dashboard
shows the five-seed comparison, and the saved reports include the
42,000-row fitting check on the same holdouts.

### Claim pruning archive

Select **Claim pruning (seeds 41–45)** in the dashboard for tune 264 and the
best development policy. All four policies lost mean F1, so the study retained
tune 264. Per-split scores, split IDs and predictions are archived in
`analysis/output/claim_pruning/`.

### Batch claim detection archive

Select **Batch claim detection (seeds 41–45)** in the dashboard for tune 264
and the refit control. Both challengers fell short of the F1 gate. The split
IDs, thresholds, predictions, and batch-size check are saved in
`analysis/output/batch_claim_detection/`.

### F1 detector comparison

Run `python -m analysis.evaluate_claim_detectors --output-dir analysis/output/claim_detector_comparison_rerun`
to compare disagreement resolution and an all-row LightGBM ranker on seeds
41–45. Install dependencies with `python -m pip install -r src/optional_requirements.txt`. The dashboard
reports the highest F1 and the highest F1 within 2% of tune 264's MAE.
Scores, predictions, and verdicts are under
`analysis/output/claim_detector_comparison/`.

### Model dashboard

Start the local dashboard server from the repository root.

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory dashboard
```

Open `http://127.0.0.1:8000/` for historical runs, both seed searches, fresh
historical evaluations, and checkpoint results. The server works offline.
Select **Checkpoint 2** for the ten final submissions from `submission.md` and
their recorded holdouts,
including the 95/5 run as `2_2_1`.
Their aggregate points use the dashboard formula below. **Historical checkpoint
blends** contains the earlier five-seed blend evaluations.

Select **Median pricing and claim detection (seeds 41–45)** for the matched 25%
holdouts. Median pricing reduced mean MAE from 97.66 to 96.72 and kept F1 at
0.4989. The best detector reached 0.4993 F1 in development. The reserved split
confirmed the pricing gain with identical flags. Reproduce the study with
`python -m analysis.evaluate_payout_signatures`, then run
`python -m analysis.screen_claim_coverage` for the additional detection screen.

Select **Profile structure and payout totals (seeds 41–45)** for the count,
payout-component and ensemble experiments. On the 25% holdouts, the experimental
ensemble with aggregate pricing reached 0.5031 F1 and 95.21 MAE. Tune 264 with
aggregate pricing reached 95.18 MAE with the original flags. The dashboard
includes a separate 20% fitting-size comparison.

Select **Payout-model verification and claim risk (seeds 41–45)** for the
relative-pricing model and new detection screens. The model scored 0.5036 F1
and 93.13 MAE on 25% holdouts, and 0.5046 F1 and 93.68 MAE on 20% holdouts.
Reproduce either comparison with `python -m analysis.verify_template_ensemble`
and the matching `--percent` value.
The optional TabICLv2 screen uses TabICL 2.2.0 and the
[official classifier checkpoint](https://huggingface.co/jingang/TabICL/blob/main/tabicl-classifier-v2-20260212.ckpt)
in `models/`. Run `python -m analysis.evaluate_foundation_profile` locally.

Select **Claim decisions and portfolio transfer (seeds 41–45)** for the
experiments on payout-source, F1 and prior calibration. Using separate
whole-cent and scaled payout probabilities reached 0.5041 F1 and 92.80 MAE on
25% holdouts, and 0.5069 F1 and 93.56 MAE on 20% holdouts. The paired uncertainty
intervals include zero for the full model's incremental changes at both sizes.
Reproduce the comparisons with `python -m analysis.confirm_marked_payouts` and
`--percent 20` or `--percent 25` after the corresponding template verification.
Independent motor risk scores from [Belgium](https://dutangc.github.io/CASdatasets/reference/beMTPL97.html)
and [Spain](https://data.mendeley.com/datasets/sw4jmdb2sm/1) rank profiles without
positive history. For those screens, use `pyreadr` in the optional analysis runtime
and place the source files named by the scripts in `analysis/output/`.
Each dashboard run lists its reproduction command.

**Profile presence and fixed-flag pricing (seeds 41–45)** compares a detector
that predicts whether any held-out row in a profile claims, then estimates how
many claim. Blending it with the payout-source ensemble reached 0.5075 F1 and
93.53 MAE on 20% holdouts, and 0.5062 F1 and 92.79 MAE on 25% holdouts.
The incremental F1 intervals include zero at both sizes.

**Profile ranking and payout composition (seeds 41–45)** compares pairwise
reranking, payout-component likelihoods and learned pricing. Pairwise reranking
reached 0.5086 F1 and 93.50 MAE on 20% holdouts. The payout-component classifier
reached 0.5067 F1 and 92.78 MAE on 25% holdouts. These are experimental results.

**Generalization to unseen profiles (seeds 41–45)** holds out all rows of
each selected profile. The payout-source ensemble reached 0.1415 F1 in this
stress test. CatBoost with unlabeled batch profile counts reached 0.1491 F1.
Reproduce it with `python -m analysis.evaluate_disjoint_claim_profiles`.

**Profile coverage and payout floors (seeds 41–45)** adds mixed-profile
holdouts and a source-tail pricing comparison. The best standard-holdout
rankings kept their F1 of 0.5086 at 20% and 0.5067 at 25% while MAE fell to
93.42 and 92.69. The mixed-profile holdouts averaged about 53% unfamiliar
rows and 0.395 F1.

**Claim-count decisions and cold-profile fallback (seeds 41–45)** compares
full claim-count distributions, boosted ranking and detector pools. The
count detector reached 0.5087 F1 and 93.47 MAE on 20% holdouts, with weaker
results on 25% holdouts. The opt-in fallback reached 0.1491 F1 and 114.51 MAE
when holding out entire profiles. Even perfect resolution of model disagreements
would reach only 0.5232 F1 on the 20% cohort.

**Cold risk and supported payouts (seeds 41–45)** compares new detectors for
profiles without positive claim history and tests payout cap exceptions. Adding
cold-profile flags lowered F1. Allowing a higher quote with exactly one matching
fitting payout kept F1 at 0.5086 and 0.5067 and reduced MAE to 93.07 and 92.15 on
20% and 25% holdouts, respectively. On 30% holdouts, the frozen pairwise
reranker reached 0.4915 F1, and the cap exception reduced MAE from 95.86 to
94.46. The pricing intervals include zero and the rule remains experimental.
Reproduce the 30% comparison with `python -m analysis.confirm_supported_payout_tail`.

**Contextual claim templates (seeds 41–45)** compares rating-dependent template
priors on matched and profile-disjoint holdouts. The contextual F1 gains varied
by cohort. With CatBoost's flags fixed, the pooled lower-tail quote reduced MAE
from 114.51 to 112.96 at 25% and from 120.90 to 119.28 at 35% of profiles held
out. Reproduce it with `python -m analysis.evaluate_disjoint_contextual_templates --percent 35`.

**Claim evidence replays (seeds 41–45)** compares near-exposure linkage,
marked Bayesian classifiers, simulated payout histories, and group cutoffs.
Even an oracle choosing a cutoff per payout-history group averaged 0.5113 F1.
The cents-based payout correction reached 0.5055 F1 and 92.65 MAE on the
25% payout-source comparison. With the observed payout cap exception, the
20% pairwise reranker reached 0.5093 F1 and 93.05 MAE. The paired F1 gain
interval includes zero and the correction lost F1 with the 25% leader. Reproduce the
combined comparison with `python -m analysis.replay_residue_and_cap --percent 20`.

**Profile calibration stability (seeds 41–45)** resamples fitting profiles
for calibration and payout-head refits. Calibration voting reached 0.5095 F1
and 93.04 MAE on the 20% holdouts. Its incremental F1 interval includes zero,
and the 25% comparison favors the earlier detector for F1. Reproduce it with
`python -m analysis.evaluate_profile_calibration_uncertainty --percent 20`.

**Claim history calibration (seeds 41–45)** tests single-observation
sensitivity, scoring-history reweighting, and kernel classifiers. Reweighting
reached 0.5096 F1 and 93.05 MAE on 20% holdouts. Its paired gain interval
includes zero and the same correction lost F1 at 25%. Reproduce it with
`python -m analysis.evaluate_history_importance_calibration --percent 20`.

**Payout outcome evidence (seeds 41–45)** compares distribution-based claim
ranking and payout-size classes. On 20% holdouts, the distribution correction
reached 0.5101 F1 and 93.02 MAE. The fixed-volume variant reached 0.5102 F1
and 93.05 MAE. F1 fell on 25% and 30% holdouts. Reproduce the comparison
with `python -m analysis.evaluate_payout_distribution_evidence --percent 20`.

**Masked profile-set networks (seeds 41–45)** learn claim counts from
partially hidden fitting histories. The networks fell short on standard
holdouts. Neural flags confirmed by the cold-start detectors reduced
profile-disjoint MAE to 112.41 at 25% and 118.66 at 35%. Reproduce the
agreement with `python -m analysis.replay_cold_payout_agreement --profile-set --percent 25`.

**Payout routing and profile activity (seeds 41–45)** compares expert
disagreement rules and latent claim activity. The hurdle blend reached
0.1522 F1 on 25% profile-disjoint holdouts with higher MAE. Even resolving
all 17 detectors’ disagreements perfectly would average 0.5365 F1 at 20%.
Reproduce the diagnostic with `python -m analysis.audit_payout_routing_headroom`.

**Finite payout inventory (seeds 41–45)** conditions on the number of feature
rows for each profile. Its best matched 25% result was 0.5043 F1 and 92.72 MAE.
On unfamiliar profiles, adding inventory to the detector vote raised F1 at 25%
and lowered it at 35%. Reproduce those comparisons with
`python -m analysis.evaluate_finite_roster_claims` and
`python -m analysis.replay_cold_inventory_votes --percent 35`.

**Marked payout components (seeds 41–45)** tests payout-size likelihoods and
structural quotes on profiles with an unmatched payout residual. F1 reranking
fell short. Repricing improved MAE on 25% holdouts and lost at 20%. An audit
using held-out labels put the pruning-only ceiling at 0.5573 F1 on 20% holdouts.
Reproduce the audit with `python -m analysis.audit_marked_component_headroom`.
The fitted veto did not approach that ceiling. Reproduce it with
`python -m analysis.evaluate_component_claim_veto --percent 20`.

**Profile neighborhood transfer (seeds 41–45)** tests residuals and
multiplicity from related profiles. The matched 25% comparison gave no F1
gain. Adding neighborhood support to the neural fallback reduced MAE to
112.39 and 118.64 on 25% and 35% profile-disjoint holdouts, with nearly
identical F1. Reproduce it with
`python -m analysis.replay_residual_neighbor_agreement --percent 25`.

**Joint payout capture (seeds 41–45)** fits payout-copy templates using fitting
histories and unlabeled scoring counts. Neither direct use nor out-of-fold
calibration beat the 20% and 25% leaders. On 20% holdouts, perfect pruning by
payout structure reached 0.5203 F1. Adding finer groups with at least ten
distinct fitting profiles yielded a ceiling of 0.5151 for those groups.
Reproduce the diagnostic with `python -m analysis.audit_joint_capture_headroom`.

**Marked exposure hazard (seeds 41–45)** models separate whole-cent and scaled
payout rates. The opt-in fallback reached 111.69 MAE at 25% and 117.96 MAE at
35% on profile-disjoint holdouts, with F1 of 0.1494 and 0.1544. The matched
20% reranker reached 0.5103 F1, with weaker results at 25%. Replay the saved
configuration with `python -m analysis.confirm_cold_exposure_hazard --percent 25`.

**Mixed portfolio routing and F1 (seeds 41–45)** tests separate risk models by
profile familiarity and reranking across payout-history mixes. Neither F1 gain
repeated on the heavier 35% mixed holdout. On the 25% mixed holdouts, fitted
payout tails and the cap exception reduced MAE from 96.14 to 95.38 with the
same claim flags. Reproduce the pricing sensitivity with
`python -m analysis.audit_mixed_payout_changes`.

**Ordinal payout distribution (seeds 41–45)** learns the probability of
exceeding several payout thresholds in one model. Direct ordinal prediction
lowered matched F1. An audit using evaluation outcomes put the 20% ceiling
from resolving detector disagreements at 0.5521 F1. The cross-fitted gate
showed no repeatable F1 gain. Reproduce the diagnostic with
`python -m analysis.audit_ordinal_disagreements`.
The conservative family vote removed five false flags across the ten matched
holdouts. Replay it with `python -m analysis.replay_ordinal_resolution --percent 20`.

**Masked payout and roster context (seeds 41–45)** compares neighbors drawn
from masked payout histories and models that use unlabeled portfolio context.
The neighbor corrections did not consistently improve F1 across holdout sizes,
and the nonlinear payout composition model also trailed the existing vote.
Reproduce with `python -m analysis.evaluate_masked_payout_neighbors --percent 20`
or `python -m analysis.evaluate_roster_context_risk --percent 20`.
Trimming quotes where the risk models agreed lowered MAE by 0.013 at 20% and
0.010 at 25% with identical flags. Replay with
`python -m analysis.replay_uncertain_claim_pricing --percent 20`.

**Verified freMPL pre-claim covariates (seeds 41–45)** matches all 30,595
training profiles to the [CASdatasets freMPL feature dictionary](https://dutangc.github.io/CASdatasets/reference/freMPL.html)
and recovers prior claim counts for 7,856 profiles. The experiments use public
coverage, deductible, and prior-history fields. They require access to equivalent
metadata for a new portfolio. To prepare the historical data, install `pyreadr`,
place `freMPL1.rda` through `freMPL10.rda` from
[CASdatasets](https://github.com/dutangc/CASdatasets/tree/master/data) in one
directory, and run `python -m analysis.prepare_frempl_covariates --source-dir <directory>`.
The calendar studies also use published policy start dates to test term timing.
Run `python -m analysis.evaluate_disjoint_policy_calendar --percent 25` for a
holdout that keeps compatible consecutive terms together. On 35% of linked
components, adding pre-claim history and dates raised mean F1 from 0.1558 to
0.1621 and reduced MAE from 115.66 to 115.08.

**Payout composition evidence (seeds 41–45)** compares a binary detector with
models that predict the kind of payment missing from a partial history.
Reproduce with `python -m analysis.evaluate_masked_payout_composition --percent 20`.

**Profile count distributions (seeds 41–45)** learns the number of claims left
in a profile using simulated scoring subsets. Agreement between two count
models improved mean F1 to 0.5137 at 20%, with no F1 gain at 25%.
Reproduce with `python -m analysis.evaluate_profile_count_hurdle --percent 20`.

**Claim history retrieval (seeds 41–45)** learns similar payout histories with
trees trained on separate fitting profiles. Admissions confirmed by both
retrieval models reached 0.5144 F1 on 20% holdouts. Reproduce with
`python -m analysis.evaluate_ranked_history_leaf_retrieval --percent 20`.
The same dashboard section includes the cross-fitted combination of count,
payout-type, and analogue scores. Its best F1 was 0.5112 at 20% and 0.5057 at
25%. Reproduce with `python -m analysis.evaluate_cross_family_claims --percent 20`.
The section also includes separate models for reviewing missed claims and false
alarms. Reproduce with `python -m analysis.evaluate_directional_claim_review --percent 20`.
The companion-roster experiment uses only the 18 input fields from `data/testset.csv`
to refine profile sizes. Reproduce with `python -m analysis.evaluate_companion_claim_roster --percent 20`.
The rating-mixture transfer study holds out entire profiles. Reproduce with
`python -m analysis.confirm_disjoint_cold_mixture`.
The rebuilt pairwise and ordinal vote experiment fits from the training rows.
Run `python -m analysis.evaluate_live_claim_vote --percent 20` for the matched
holdout or `python -m analysis.confirm_live_claim_vote_transfer` for mixed profiles.
The pairwise cutoff audit uses held-out outcomes to measure the ceiling of
threshold changes. Run `python -m analysis.audit_pairwise_volume_headroom`.

Select **Extended claim history (seeds 41-45)** to see the two LightGBM results,
their per-seed metrics, and downloadable training configurations.
Select **Extended claim history tuning (500 runs)** for 500 operating points
scored on the same 25% holdouts with seeds 41–45. The source JSON includes a
comparison of the selected leaders and earlier baselines on seeds 301–305.
The dashboard opens on **All sources**, sorted by mean MAE, with **5 or more** seeds.
Selecting an individual split seed sets the minimum to one. Use **Minimum F1**
to filter mean claim F1 on a 0–1 scale. The other metric filters set a maximum
mean MAE and minimum precision and recall. Filters update automatically, and
**Apply filters** refreshes the results using the current values. The table shows
**Applying filters…** until the updated results are ready. When Minimum F1 is set,
the filter panel shows the applied cutoff and match count. Select **Claim F1
tuning (500 runs)** to see results from seeds 41–45 sorted by mean claim F1.
Rebuild the F1 study with `python -m analysis.tune_claim_f1`.
Matching operating points average unique seeds. Grouped results select the
lowest mean MAE among matching points. Fixed-point studies show every matching
point. The results table shows mean MAE,
the lowest saved MAE across the included seeds, matched all-zero mean MAE,
percentage improvement, precision, recall, mean F1 and aggregate points.
The pagination controls below the table show the total number of matching results.
Hover over a run label or select **View** to see its full name.
Aggregate points score mean holdout MAE and mean F1 linearly on 0–50 scales
each: MAE earns 0 points at 120 or above up to 50 points at 70 or below, and F1
earns 0 points at 0 up to 50 points at 0.80 or above. Each metric is rounded to
two decimal places before the two scores are added (0–100 total).
Mean F1 averages each seed's claim-detection F1 (0–1; higher is better),
with zero when precision and recall are both zero. It is sortable and included in the run's observations download.
Star rows to favourite their exact configuration and seed cohort; **Favourites only**
shows saved rows alongside the other filters. Bookmarks persist in browser local
storage for the same address and browser. Bookmarks from the old local file stay
with that file's address. If storage is blocked, bookmarks last for the session.
Clearing filters keeps bookmarks. Multiple operating points can be bookmarked per model.

The dashboard data and optional analysis caches are separate from the checkpoint 2 generator.
Regenerate the extended-history tuning scores with
`python -m analysis.tune_claim_history --output analysis/output/claim_history_tuning.json --dashboard dashboard/index.html`
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
