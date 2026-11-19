import numpy as np
from sklearn.metrics import roc_auc_score


def summarize_scores(scores):
    """Mean and standard deviation of the fold scores."""
    scores = np.array(scores)
    return round(scores.mean(), 5), round(scores.std(), 5)


def oof_auc(y, oof):
    """AUC on all out-of-fold predictions together."""
    return round(roc_auc_score(y, oof), 5)


def compare(scores_a, scores_b):
    """Fold-by-fold comparison: how often b beat a, and by how much on average."""
    delta = np.array(scores_b) - np.array(scores_a)
    return {
        "mean_delta": round(delta.mean(), 5),
        "folds_improved": f"{int((delta > 0).sum())}/{len(delta)}",
    }