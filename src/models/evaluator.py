import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    search_range: tuple = (0.2, 0.6),
    steps: int = 41,
) -> tuple[float, float]:
    """
    Search for threshold that maximizes chosen metric.

    metric="f1"     → balanced precision/recall (default)
    metric="recall" → catch maximum churners (high cost of missing churn)
    """
    thresholds = np.linspace(search_range[0], search_range[1], steps)
    scores = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == "f1":
            scores.append(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            scores.append(recall_score(y_true, y_pred, zero_division=0))
        elif metric == "precision":
            scores.append(precision_score(y_true, y_pred, zero_division=0))

    best_idx = np.argmax(scores)
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    logger.info(f"Threshold search ({metric}): best={best_threshold:.3f} score={best_score:.4f}")
    return best_threshold, best_score


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Full evaluation at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        # Threshold-independent
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),

        # Threshold-dependent
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "threshold": round(threshold, 4),

        # Confusion matrix
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),

        # Business metrics
        "churn_caught_rate": round(tp / (tp + fn), 4),
        "false_alarm_rate": round(fp / (fp + tn), 4),
    }

    # Log full report
    logger.info(f"\nMetrics at threshold={threshold:.3f}:")
    logger.info(f"  ROC-AUC:    {metrics['roc_auc']}")
    logger.info(f"  F1:         {metrics['f1']}")
    logger.info(f"  Precision:  {metrics['precision']}")
    logger.info(f"  Recall:     {metrics['recall']}")
    logger.info(f"  TP:{tp} FP:{fp} TN:{tn} FN:{fn}")
    logger.info(
        f"\n{classification_report(y_true, y_pred, target_names=['Stay', 'Churn'])}"
    )

    return metrics


def compare_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Show metrics at multiple thresholds side by side.
    Helps business understand precision/recall tradeoff.
    """
    rows = []
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred = (y_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
            "churners_caught": int((y_pred == 1).sum()),
        })

    df = pd.DataFrame(rows)
    logger.info(f"\nThreshold comparison:\n{df.to_string(index=False)}")
    return df