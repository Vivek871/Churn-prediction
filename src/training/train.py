# ── Standard library (no path needed) ────────────────────────────────────────
import os
import sys
import joblib
from pathlib import Path

# ── Path fix MUST happen before any src imports ───────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Third party ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

load_dotenv()

# ── Local imports (src/) ──────────────────────────────────────────────────────
from src.models.registry import register_model, promote_to_staging  # noqa: E402
from src.data.loader import load_raw_data, basic_clean  # noqa: E402
from src.data.validator import validate  # noqa: E402
from src.features.engineer import create_domain_features  # noqa: E402
from src.models.pipeline import build_pipeline  # noqa: E402
from src.models.evaluator import (  # noqa: E402
    find_optimal_threshold,
    evaluate_model,
    compare_thresholds,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__, log_file="logs/training.log")

def run_cross_validation(pipeline, X_train, y_train) -> dict:
    logger.info("Running 5-fold stratified cross validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "avg_precision": "average_precision",
    }

    results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    summary = {}
    for key, values in results.items():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            summary[metric] = {
                "mean": round(values.mean(), 4),
                "std": round(values.std(), 4),
            }
            logger.info(f"  CV {metric}: {values.mean():.4f} ± {values.std():.4f}")

    return summary


def log_to_mlflow(
    config: dict,
    pipeline,
    metrics: dict,
    cv_results: dict,
    optimal_threshold: float,
    X_train,
    X_test,
    y_test,
) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns/"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction"))

    with mlflow.start_run(run_name="xgb-churn") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # ── 1. Tags ───────────────────────────────────────
        mlflow.set_tags({
            "model_type": "XGBoostClassifier",
            "imbalance_strategy": "SMOTE",
            "dataset": "Telco-Customer-Churn",
            "engineer": "mrityunjay",
        })

        # ── 2. Hyperparameters ────────────────────────────
        mlflow.log_params({
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "learning_rate": config["model"]["learning_rate"],
            "subsample": config["model"]["subsample"],
            "colsample_bytree": config["model"]["colsample_bytree"],
            "min_child_weight": config["model"]["min_child_weight"],
            "gamma": config["model"]["gamma"],
            "reg_alpha": config["model"]["reg_alpha"],
            "reg_lambda": config["model"]["reg_lambda"],
            "smote_sampling_strategy": 0.7,
            "optimal_threshold": round(optimal_threshold, 4),
        })

        # ── 3. Test metrics ───────────────────────────────
        mlflow.log_metrics({
            "test_roc_auc": metrics["roc_auc"],
            "test_f1": metrics["f1"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_avg_precision": metrics["avg_precision"],
            "test_churn_caught_rate": metrics["churn_caught_rate"],
            "test_false_alarm_rate": metrics["false_alarm_rate"],
            "test_tp": float(metrics["true_positives"]),
            "test_fp": float(metrics["false_positives"]),
            "test_tn": float(metrics["true_negatives"]),
            "test_fn": float(metrics["false_negatives"]),
        })

        # ── 4. CV metrics ─────────────────────────────────
        for metric_name, values in cv_results.items():
            mlflow.log_metric(f"cv_{metric_name}_mean", values["mean"])
            mlflow.log_metric(f"cv_{metric_name}_std", values["std"])

        # ── 5. Model with signature ───────────────────────
        sample_input = X_train.head(5)
        sample_output = pipeline.predict_proba(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=sample_input,
        )

        # ── 6. Config + threshold as artifacts ────────────
        mlflow.log_artifact("configs/config.yaml")
        mlflow.log_artifact("artifacts/threshold.txt")

        logger.info("MLflow run logged successfully")
        logger.info("View at: http://localhost:5000")

        return run_id


def main():
    config = load_config()

    # ── 1. Data ───────────────────────────────────────────
    df = load_raw_data()
    df = basic_clean(df)
    validate(df)
    df = create_domain_features(df)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=config["data"]["random_state"],
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.2%}")
    logger.info(f"Test churn rate:  {y_test.mean():.2%}")

    # ── 2. Cross Validation ───────────────────────────────
    pipeline = build_pipeline(config)
    cv_results = run_cross_validation(pipeline, X_train, y_train)

    # ── 3. Final Training ─────────────────────────────────
    logger.info("Fitting final pipeline on full training set...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # ── 4. Threshold Tuning ───────────────────────────────
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    compare_thresholds(y_test.values, y_proba)

    optimal_threshold, best_f1 = find_optimal_threshold(
        y_test.values, y_proba,
        metric="f1",
        search_range=(0.2, 0.6),
        steps=41,
    )

    # ── 5. Final Evaluation ───────────────────────────────
    metrics = evaluate_model(y_test.values, y_proba, optimal_threshold)

    # ── 6. Save Artifacts ─────────────────────────────────
    artifacts_path = Path(os.getenv("ARTIFACTS_PATH", "artifacts/"))
    artifacts_path.mkdir(exist_ok=True)

    model_path = artifacts_path / "churn_pipeline.joblib"
    threshold_path = artifacts_path / "threshold.txt"

    joblib.dump(pipeline, model_path)
    threshold_path.write_text(str(optimal_threshold))

    logger.info(f"Model saved:     {model_path}")
    logger.info(f"Threshold saved: {threshold_path}")
    logger.info(f"Final ROC-AUC:   {metrics['roc_auc']}")
    logger.info(f"Final F1:        {metrics['f1']}")
    logger.info(f"Churners caught: {metrics['churn_caught_rate']:.2%}")

    # ── 7. Log to MLflow ──────────────────────────────────
    run_id = log_to_mlflow(
        config=config,
        pipeline=pipeline,
        metrics=metrics,
        cv_results=cv_results,
        optimal_threshold=optimal_threshold,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )

    logger.info(f"Run ID: {run_id}")



    # ── 8. Register and promote to Staging ───────────────
    model_name = os.getenv("MLFLOW_MODEL_NAME", "churn-xgboost")
    version = register_model(run_id, model_name)
    promote_to_staging(model_name, version)

    logger.info(f"Model '{model_name}' v{version} is in Staging ✓")
    logger.info("Day 1 complete ✓")

    return pipeline, metrics, cv_results, optimal_threshold


if __name__ == "__main__":
    main()