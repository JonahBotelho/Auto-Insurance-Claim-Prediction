"""
Saves and loads the trained two-stage model so train.py only needs to
fit it once; predict.py then reuses the exact same fitted model
instead of retraining from scratch every time predictions are needed.
"""

from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "two_stage_claim_model.joblib"


def save_model(model, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path: Path = DEFAULT_MODEL_PATH):
    return joblib.load(path)
