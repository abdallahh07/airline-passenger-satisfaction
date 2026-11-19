import joblib
import pandas as pd

from utils.config import ID_COL, MODEL_DIR, SUBMISSION_DIR, TARGET


def load_model():
    return joblib.load(MODEL_DIR / "lgbm_model.joblib")


def predict(model, x):
    """Probability of 'satisfied' for each row."""
    return model.predict_proba(x)[:, 1]


def make_submission(ids, pred, name="submission.csv"):
    sub = pd.DataFrame({ID_COL: ids, TARGET: pred})

    assert sub[ID_COL].is_unique, "duplicate ids"
    assert sub[TARGET].notna().all(), "NaN predictions"
    assert sub[TARGET].between(0, 1).all(), "predictions must be probabilities"

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / name
    sub.to_csv(path, index=False)
    return path