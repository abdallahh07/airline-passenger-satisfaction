from data_processing.load_data import load_data
from data_processing.preprocess import imputer, x_y
from model.evaluate import oof_auc, summarize_scores
from model.predict import make_submission, predict
from model.train import engine, train_final
from utils.config import FEATURED_RAW, ID_COL, LGBM_PARAMS
from utils.mlflow_utils import log_run, setup_mlflow


def main():
    setup_mlflow()

    # 1. load and prepare
    train, test, orig = load_data()
    train = imputer(train)
    test = imputer(test)

    # 2. features: the 21 raw columns (the engineered ones did not help)
    x, y = x_y(train)
    x = x[FEATURED_RAW]
    x_test = test[FEATURED_RAW]

    # 3. cross-validation
    oof, scores = engine(x, y)
    mean, std = summarize_scores(scores)
    print(f"CV AUC {mean} ± {std}   folds {[round(s, 5) for s in scores]}")
    print(f"OOF AUC {oof_auc(y, oof)}")

    # 4. final model on all train rows, then the submission
    model = train_final(x, y)
    pred = predict(model, x_test)
    path = make_submission(test[ID_COL], pred)
    print(f"saved {path}  rows {len(pred)}")

    # 5. record the run in MLflow
    log_run("lgbm_tuned_raw", LGBM_PARAMS, scores, oof_auc(y, oof),
            model=model, n_features=x.shape[1])
    print("logged to MLflow")


if __name__ == "__main__":
    main()