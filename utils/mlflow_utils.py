import mlflow

from utils.config import PROJECT_ROOT

TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "airline-satisfaction"


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(run_name, params, scores, oof_score, model=None, n_features=None):
    """Log one experiment: its settings, its fold scores and (optionally) the model."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        if n_features is not None:
            mlflow.log_param("n_features", n_features)
        for fold, score in enumerate(scores, start=1):
            mlflow.log_metric("fold_auc", score, step=fold)
        mlflow.log_metric("cv_auc_mean", sum(scores) / len(scores))
        mlflow.log_metric("oof_auc", oof_score)
        if model is not None:
            mlflow.lightgbm.log_model(model, name="model")