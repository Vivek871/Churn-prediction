from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.features.preprocessor import build_preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_pipeline(config: dict) -> ImbPipeline:
    """
    Builds full ML pipeline:
    preprocessor → SMOTE → XGBoost

    ImbPipeline ensures SMOTE only runs during fit(),
    never during transform() or predict().
    """

    # --- Step 1: Preprocessor ---
    preprocessor = build_preprocessor(config)

    # --- Step 2: SMOTE ---
    # sampling_strategy=0.7 means:
    # minority class will be 70% the size of majority class
    # NOT fully balanced (1.0) — preserves some natural class signal
    smote = SMOTE(
        sampling_strategy=0.7,
        random_state=config["model"]["random_state"],
        k_neighbors=5,
    )

    # --- Step 3: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        subsample=config["model"]["subsample"],
        colsample_bytree=config["model"]["colsample_bytree"],
        random_state=config["model"]["random_state"],
        eval_metric=config["model"]["eval_metric"],
        n_jobs=-1,
        verbosity=0,
    )

    # --- Combine into ImbPipeline ---
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),   # Step 1: clean + encode
        ("smote", smote),                  # Step 2: balance classes
        ("classifier", xgb),              # Step 3: train model
    ])

    logger.info("Pipeline built: preprocessor → SMOTE → XGBoost")
    return pipeline