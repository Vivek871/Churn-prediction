import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

# These columns MUST exist for the pipeline to work
REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

# Expected dtypes after basic_clean()
EXPECTED_DTYPES = {
    "tenure": "int64",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
    "Churn": "int64",
    "SeniorCitizen": "int64",
}


def check_required_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info(f"Column check passed — all {len(REQUIRED_COLUMNS)} required columns present")


def check_dtypes(df: pd.DataFrame) -> None:
    errors = []
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != expected:
            errors.append(f"  Column '{col}': expected {expected}, got {actual}")

    if errors:
        raise TypeError("Dtype validation failed:\n" + "\n".join(errors))
    logger.info("Dtype check passed")


def check_nulls(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]

    if len(null_cols) > 0:
        # Warning not error — some nulls may be handled downstream
        logger.warning(f"Nulls detected:\n{null_cols}")
    else:
        logger.info("Null check passed — no missing values")


def check_target_binary(df: pd.DataFrame, target_col: str = "Churn") -> None:
    unique_values = set(df[target_col].unique())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"Target column '{target_col}' must be binary (0/1). Found: {unique_values}")
    logger.info(f"Target check passed — values: {unique_values}")


def check_value_ranges(df: pd.DataFrame) -> None:
    errors = []

    if df["tenure"].min() < 0:
        errors.append("tenure has negative values")

    if df["MonthlyCharges"].min() < 0:
        errors.append("MonthlyCharges has negative values")

    if df["TotalCharges"].min() < 0:
        errors.append("TotalCharges has negative values")

    if errors:
        raise ValueError(f"Range validation failed: {errors}")
    logger.info("Range check passed")


def validate(df: pd.DataFrame, target_col: str = "Churn") -> None:
    """
    Run all validations in order.
    Call this after basic_clean(), before feature engineering.
    """
    logger.info("Starting data validation...")
    check_required_columns(df)
    check_dtypes(df)
    check_nulls(df)
    check_target_binary(df, target_col)
    check_value_ranges(df)
    logger.info("All validation checks passed ✓")