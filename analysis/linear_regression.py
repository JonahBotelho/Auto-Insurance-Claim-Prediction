"""
Fit a basic claim amount regression and report MAE on the training data.
Note that this is meant as baseline regression calcualtion with no preprocessing or modification.
Do not expand on this file's logic.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def main():
    source = Path(__file__).resolve().parents[1] / "data" / "trainingset.csv"
    data = np.loadtxt(source, delimiter=",", skiprows=1)
    # Exclude rowIndex and keep ClaimAmount as the target, not a predictor.
    features, claim_amounts = data[:, 1:-1], data[:, -1]
    model = LinearRegression().fit(features, claim_amounts)
    predictions = model.predict(features)
    print(f"Training MAE: {mean_absolute_error(claim_amounts, predictions):.4f}")


if __name__ == "__main__":
    main()
