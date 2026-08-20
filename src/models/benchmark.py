from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from src.features.preprocessor import build_preprocessor
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def get_candidate_models(config: dict) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            learning_rate=config["model"]["learning_rate"],
            subsample=config["model"]["subsample"],
            colsample_bytree=config["model"]["colsample_bytree"],
            min_child_weight=config["model"]["min_child_weight"],
            gamma=config["model"]["gamma"],
            reg_alpha=config["model"]["reg_alpha"],
            reg_lambda=config["model"]["reg_lambda"],
            random_state=config["model"]["random_state"],
            eval_metric=config["model"]["eval_metric"],
            verbosity=0,
            n_jobs=-1,
        ),
    }

def benchmark_models(X_train, y_train, config: dict) -> pd.DataFrame:
    """
    Run 5-fold stratified CV on all candidate models.
    Returns sorted results by ROC-AUC.
    """
    preprocessor = build_preprocessor(config)
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    for name, model in get_candidate_models(config).items():
        logger.info(f"Benchmarking: {name}")

        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", model),
        ])

        # Score on multiple metrics
        for metric in ["roc_auc", "f1", "average_precision"]:
            scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv, scoring=metric, n_jobs=-1
            )
            results.append({
                "model": name,
                "metric": metric,
                "mean": round(scores.mean(), 4),
                "std": round(scores.std(), 4),
            })

    df_results = pd.DataFrame(results)

    # Print clean summary
    summary = df_results[df_results["metric"] == "roc_auc"].sort_values(
        "mean", ascending=False
    )
    logger.info(f"\nBenchmark Results (ROC-AUC):\n{summary.to_string(index=False)}")

    return df_results