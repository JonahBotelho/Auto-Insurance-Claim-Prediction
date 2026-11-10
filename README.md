# COMP4983-Final-Project

This project aims to predict insurance claim for insured customers, given unlabled features related to the customer and vehicle. 

Input datasets and the submission template are in `data/`:
`trainingset.csv`, `testset.csv`, and `submission.csv`.

## Prediction

Prediction code is in `src/`. Requires Python 3.10 or newer. Run from the
repository folder:

```powershell
python -m pip install -r requirements.txt
python src/predict.py                 # $0 baseline -> submissions/1_2_1.csv
python src/predict.py --model lasso   # lasso       -> submissions/1_2_2.csv
```

Each run prints the training MAE. Models:

- `zero`: predicts $0 for every customer (training MAE 103.05).
- `lasso`: lasso on log(1 + ClaimAmount), with one-hot encoded categories and
  standardized numeric features (training MAE 103.40). Set `--alpha` to change
  the regularization strength (default 0.1).

Shared preprocessing is in `src/preprocessing.py`: one-hot encoding (levels
with fewer than 20 training rows share an "infrequent" column), standardized
numeric features, and an optional log(1 + ClaimAmount) target. Other candidate
models (ridge, polynomial, regression tree, k-NN, two-part) are in
`src/claim_models.py` and `src/two_part_claim_model.py`.

Use `--input` and `--output` for other files, for example:

```powershell
python src/predict.py --input data/competitionset.csv --output predictedclaimamount.csv
```

Run the unit tests with `python -m unittest discover -s src/tests`.

## Analysis

The analysis directory is made for short scripts to provide insights on data, not for main prediction logic.

### Basic linear regression

Fit a linear regression using NumPy and scikit-learn to predict `ClaimAmount`
from the 18 feature columns in `data/trainingset.csv` (`rowIndex` is excluded):

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/linear_regression.py
```

The script prints mean absolute error (MAE) on the same training rows used to fit
the model. This is training error, not a measure of performance on unseen data.

### Model comparison

Cross-validate every candidate model with the same grouped 5-fold split and
print one table (training MAE, CV MAE, gap, run time), sorted best first.
Takes about 3 minutes; use `--only` to run a subset:

```powershell
python analysis/model_comparison.py
python analysis/model_comparison.py --only tree two-part
```

Results are saved to `analysis/output/model_comparison.csv`.

### Lasso cross-validation

Compare lasso on raw and log-transformed claims across a grid of alphas, using
5-fold cross-validation that keeps duplicate rows in the same fold:

```powershell
python analysis/lasso_cross_validation.py
```

Results are saved to `analysis/output/lasso_cv_results.csv`.

### Correlation analysis

Generate training-data correlation plots with Python 3.10 or newer. Run from the repository folder:

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/correlations.py
```

Open `analysis/output/index.html` to browse the 171 scatterplots and correlation
results. To share the gallery, include the entire `analysis/output/` folder.
