import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Feature 1: Revenue intensity ---
    # Hypothesis: High spenders with short tenure = not getting value = churn risk
    # +1 avoids division by zero for tenure=0 customers
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    logger.info("Created: charges_per_tenure")

    # --- Feature 2: Tenure band ---
    # Hypothesis: Churn behavior differs by customer lifecycle stage
    # new(0-12m), early(12-24m), mid(24-48m), loyal(48m+)
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "early", "mid", "loyal"],
        include_lowest=True,
    ).astype(str)
    logger.info("Created: tenure_band")

    # --- Feature 3: Service count ---
    # Hypothesis: More services = more engaged = less likely to churn
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)
    logger.info("Created: service_count")

    # --- Feature 4: Contract risk flag ---
    # Hypothesis: Month-to-month = no commitment = easiest to leave
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    logger.info("Created: is_month_to_month")

    # --- Feature 5: Payment method risk flag ---
    # Hypothesis: Electronic check = manual payment = more friction = churn signal
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    logger.info("Created: is_electronic_check")

    # --- Feature 6: No online security AND no tech support ---
    # Hypothesis: Customers with no support services feel underserved = churn risk
    df["no_support_services"] = (
        (df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No")
    ).astype(int)
    logger.info("Created: no_support_services")

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df