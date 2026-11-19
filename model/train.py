import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils.config import LGBM_PARAMS, MODEL_DIR, N_SPLIT, RANDOM_STATE


def engine(x, y, params=LGBM_PARAMS):
    """5-fold CV. Returns out-of-fold predictions and the score of each fold."""
    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    scores = []
    for tr_idx, val_idx in skf.split(x, y):
        model = LGBMClassifier(**params)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], oof[val_idx]))
    return oof, scores


def train_final(x, y, params=LGBM_PARAMS):
    """Fit one model on all the data and save it for predict.py and the API."""
    model = LGBMClassifier(**params).fit(x, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "lgbm_model.joblib")
    return model