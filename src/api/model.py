import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from src.features.engineer import create_domain_features
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class ChurnModel:
    """
    Wraps the trained pipeline + threshold.
    Loaded once at startup, reused for every request.
    """

    def __init__(self):
        self.pipeline = None
        self.threshold = None
        self.model_version = None
        self.is_loaded = False

    def load(self):
        """
        Load model from MLflow registry (Staging)
        Falls back to local joblib if MLflow unavailable.
        """
        try:
            self._load_from_mlflow()
        except Exception as e:
            logger.warning(f"MLflow load failed: {e}")
            logger.info("Falling back to local artifact...")
            self._load_from_local()

        self._load_threshold()
        self.is_loaded = True
        logger.info(
            f"Model loaded ✓ | version={self.model_version} "
            f"| threshold={self.threshold}"
        )

    def _load_from_mlflow(self):
        """Load pipeline from MLflow Staging model."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns/"))
        model_name = os.getenv("MLFLOW_MODEL_NAME", "churn-xgboost")

        client = MlflowClient()

        # Suppress deprecation warning — stages still work in 2.14
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            versions = client.get_latest_versions(
                model_name, stages=["Staging"]
            )

        if not versions:
            raise ValueError(f"No Staging model found for '{model_name}'")

        latest = versions[0]
        self.model_version = latest.version

        model_uri = f"models:/{model_name}/Staging"
        self.pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded from MLflow: '{model_name}' v{self.model_version}")

    def _load_from_local(self):
        """Fallback: load from local joblib file."""
        model_path = Path(
            os.getenv("ARTIFACTS_PATH", "artifacts/")
        ) / "churn_pipeline.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        self.pipeline = joblib.load(model_path)
        self.model_version = "local"
        logger.info(f"Loaded from local: {model_path}")


    def _load_threshold(self):
        """Load optimal threshold from file."""
        threshold_path = Path(
            os.getenv("ARTIFACTS_PATH", "artifacts/")
        ) / "threshold.txt"

        if threshold_path.exists():
            self.threshold = round(float(threshold_path.read_text().strip()), 4)
        else:
            self.threshold = 0.33  # fallback default
            logger.warning("threshold.txt not found, using default 0.33")

    def predict(self, data: dict) -> dict:
        """
        Run prediction on a single customer record.
        Returns probability, prediction, and risk level.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert to DataFrame — pipeline expects DataFrame not dict
        df = pd.DataFrame([data])

        # Apply same feature engineering as training
        df = create_domain_features(df)

        # Get churn probability
        churn_proba = float(
            self.pipeline.predict_proba(df)[:, 1][0]
        )

        # Apply threshold
        churn_prediction = churn_proba >= self.threshold

        # Risk level for business users
        if churn_proba >= 0.7:
            risk_level = "HIGH"
        elif churn_proba >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "churn_probability": round(churn_proba, 4),
            "churn_prediction": bool(churn_prediction),
            "risk_level": risk_level,
            "threshold_used": self.threshold,
            "model_version": str(self.model_version),
        }


# Global singleton — one instance for the entire app lifetime
churn_model = ChurnModel()