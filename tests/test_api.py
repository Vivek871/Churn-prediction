import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pytest
import pandas as pd
from src.data.loader import load_raw_data, basic_clean
from src.data.validator import validate
from src.features.engineer import create_domain_features


# ── Data Tests ────────────────────────────────────────────────────────────────
def test_load_raw_data():
    df = load_raw_data()
    assert df.shape[0] == 7043
    assert df.shape[1] == 21
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_basic_clean():
    df = load_raw_data()
    df = basic_clean(df)
    assert df["TotalCharges"].dtype == "float64"
    assert df["Churn"].isin([0, 1]).all()
    assert df["TotalCharges"].isnull().sum() == 0


def test_validate_passes_on_clean_data():
    df = load_raw_data()
    df = basic_clean(df)
    validate(df)  # should not raise


def test_validate_fails_on_bad_data():
    bad_df = pd.DataFrame({
        "tenure": [1, 2],
        "Churn": [0, 1]
    })
    with pytest.raises(ValueError):
        validate(bad_df)


# ── Feature Engineering Tests ─────────────────────────────────────────────────
def test_domain_features_created():
    df = load_raw_data()
    df = basic_clean(df)
    df = create_domain_features(df)

    expected_features = [
        "charges_per_tenure",
        "tenure_band",
        "service_count",
        "is_month_to_month",
        "is_electronic_check",
        "no_support_services",
    ]
    for feat in expected_features:
        assert feat in df.columns, f"Missing feature: {feat}"


def test_charges_per_tenure_no_division_by_zero():
    df = load_raw_data()
    df = basic_clean(df)
    df = create_domain_features(df)
    assert df["charges_per_tenure"].isnull().sum() == 0
    assert (df["charges_per_tenure"] >= 0).all()


def test_is_month_to_month_binary():
    df = load_raw_data()
    df = basic_clean(df)
    df = create_domain_features(df)
    assert df["is_month_to_month"].isin([0, 1]).all()


def test_service_count_range():
    df = load_raw_data()
    df = basic_clean(df)
    df = create_domain_features(df)
    assert df["service_count"].min() >= 0
    assert df["service_count"].max() <= 7


# ── Schema Tests ──────────────────────────────────────────────────────────────
def test_churn_request_valid():
    from src.api.schemas import ChurnRequest
    valid = {
        "tenure": 12,
        "MonthlyCharges": 65.50,
        "TotalCharges": 786.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }
    req = ChurnRequest(**valid)
    assert req.tenure == 12
    assert req.MonthlyCharges == 65.50


def test_churn_request_invalid_tenure():
    from src.api.schemas import ChurnRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ChurnRequest(
            tenure=-5,
            MonthlyCharges=65.50,
            TotalCharges=786.0,
            Contract="Month-to-month",
            PaymentMethod="Electronic check",
            PaperlessBilling="Yes",
            gender="Male",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="Fiber optic",
            OnlineSecurity="No",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
        )