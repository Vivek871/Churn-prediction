from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class ChurnRequest(BaseModel):
    # Account info
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    Contract: str
    PaymentMethod: str
    PaperlessBilling: str

    # Demographics
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str

    # Phone services
    PhoneService: str
    MultipleLines: str

    # Internet services
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class ChurnResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float
    churn_prediction: bool
    risk_level: str
    threshold_used: float
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
    threshold: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None