import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns/"))
    return MlflowClient()


def register_model(run_id: str, model_name: str) -> str:
    """
    Register model from a run into the Model Registry.
    Returns the version number.
    """
    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Registering model '{model_name}' from run {run_id}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = result.version
    logger.info(f"Model registered as version {version}")
    return version


def promote_to_staging(model_name: str, version: str) -> None:
    """
    Promote a model version to Staging.
    Archives any existing Staging version automatically.
    """
    client = get_client()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,  # old Staging → Archived automatically
    )

    # Add useful tags to this version
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="promoted_by",
        value="train.py",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="status",
        value="awaiting_review",
    )

    logger.info(f"Model '{model_name}' v{version} promoted to Staging ✓")


def promote_to_production(model_name: str, version: str) -> None:
    """
    Promote a Staging model to Production.
    Use this after manual review/testing confirms model is good.
    """
    client = get_client()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="status",
        value="production_live",
    )

    logger.info(f"Model '{model_name}' v{version} promoted to Production ✓")


def get_production_model(model_name: str):
    """
    Load the current Production model.
    This is what your API will call at inference time.
    """
    client = get_client()

    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        raise ValueError(f"No Production model found for '{model_name}'")

    latest = versions[0]
    logger.info(f"Loading Production model: '{model_name}' v{latest.version}")

    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    return model, latest.version


def list_model_versions(model_name: str) -> None:
    """Print all versions and their stages."""
    client = get_client()

    try:
        versions = client.get_latest_versions(model_name)
        logger.info(f"Model versions for '{model_name}':")
        for v in versions:
            logger.info(
                f"  v{v.version} | stage={v.current_stage} | run_id={v.run_id[:8]}..."
            )
    except Exception as e:
        logger.warning(f"Could not list versions: {e}")