import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

def load_raw_data(path : str | None = None) -> pd.DataFrame :
    path = path or os.getenv("RAW_DATA_PATH")

    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # never mutate the original

    # TotalCharges has whitespace strings ("  ") instead of NaN for new customers
    # pd.to_numeric with errors='coerce' converts those to NaN safely
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # New customers (tenure=0) have no TotalCharges — fill with MonthlyCharges
    null_mask = df["TotalCharges"].isnull()
    df.loc[null_mask, "TotalCharges"] = df.loc[null_mask, "MonthlyCharges"]
    logger.info(f"Fixed {null_mask.sum()} TotalCharges nulls")

    # Convert target: "Yes"/"No" → 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Log class distribution — critical to know imbalance upfront
    churn_rate = df["Churn"].mean()
    logger.info(f"Churn rate: {churn_rate:.2%}  |  Class counts: {df['Churn'].value_counts().to_dict()}")

    return df