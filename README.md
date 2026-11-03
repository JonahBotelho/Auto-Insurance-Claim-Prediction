# COMP4983-Final-Project

This project aims to predict insurance claim for insured customers, given unlabled features related to the customer and vehicle. 

Input datasets and the submission template are in `data/`:
`trainingset.csv`, `testset.csv`, and `submission.csv`.

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

### Correlation analysis

Generate training-data correlation plots with Python 3.10 or newer. Run from the repository folder:

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/correlations.py
```

Open `analysis/output/index.html` to browse the 171 scatterplots and correlation
results. To share the gallery, include the entire `analysis/output/` folder.
