from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
    - Scales numeric columns
    - Encodes categorical columns
    - Handles each column type independently
    """

    # --- Numeric columns (from config + engineered features) ---
    numeric_cols = config["features"]["numeric_cols"] + [
        "charges_per_tenure",
        "service_count",
    ]

    # --- Categorical columns (from config + engineered features) ---
    categorical_cols = config["features"]["categorical_cols"] + [
        "tenure_band",
    ]

    # --- Binary columns (already 0/1 — no transformation needed) ---
    binary_cols = [
        "SeniorCitizen",
        "is_month_to_month",
        "is_electronic_check",
        "no_support_services",
    ]


    logger.info(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"Binary cols ({len(binary_cols)}): {binary_cols}")


    # --- Numeric transformer ---
    # Impute missing values first, then scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy= "median")),
        ("scaler", StandardScaler()),
    ])

    # --- Categorical transformer ---
    # Impute missing values first, then encode
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown = "use_encoded_value",
                unknown_value = -1,   # unseen categories at inference → -1
            ),
        ),
    ])

    # --- Binary transformer ---
    # Already 0/1 integers — just impute in case of nulls
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy= "most_frequent")),
    ])

    # --- ColumnTransformer: applies each transformer to its columns ---
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
            ("bin", binary_transformer, binary_cols),
        ],
        remainder = "drop",        # drop any column not listed above
        verbose_feature_names_out= True,
    )

    return preprocessor



def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Get output feature names after fitting."""
    return list(preprocessor.get_feature_names_out())







