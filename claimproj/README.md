# LightGBM/CatBoost claim workflow

Run all commands from the repository root (the folder containing `src/`,
`claimproj/`, and `data/`). Use Python 3.10 or newer. The supplied datasets stay
in the root `data/` folder; no `claimproj/data/` copy is needed.

```powershell
python -m pip install -r claimproj/requirements.txt
python -m claimproj.gen_submissions
```

The generator refits on all training rows and writes **only submissions 7-10**
to `data/regenerated/`. Settings come from `src/feature_schema.py`:

| File | Classifier | Threshold | Amount scale |
|---|---|---|---|
| 1_2_7.csv | LightGBM | 0.55 | 1.0 |
| 1_2_8.csv | LightGBM | 0.50 | 1.0 |
| 1_2_9.csv | LightGBM | 0.60 | 1.0 |
| 1_2_10.csv | Equal LightGBM/CatBoost blend | 0.50 | 0.8 |

It also prints full-training MAE and saves `training_mae.csv` in that directory.
These are in-sample scores: classification features exclude each training row's
own claim label, but amount estimates use all training claims. They do not
estimate unseen-data performance. Do not upload `training_mae.csv` as predictions.
Use `--output-dir path/to/folder` to change the destination; existing files in
that destination are replaced. Submissions 1-6 use `src/`; see `submission.md`.

Optional evaluation (writes files under root `results/`):

```powershell
python -m claimproj.cv_harness --repeats 3 --models lgbm,cat
python -m claimproj.holdout_eval --splits 3
```

CV and holdout MAE are validation scores, separate from training MAE.
`claimproj/results/cv_summary.csv` contains the historical CV results.

To save a single LightGBM model and predict another dataset:

```powershell
python -m claimproj.train_final --threshold 0.55
python -m claimproj.predict --input data/testset.csv --output predictions.csv
```

These commands write/load `model.joblib` in the repository root. This single
model does not implement submission 10's blend or amount multiplier.
