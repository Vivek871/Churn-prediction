from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.api.schemas import ChurnRequest, ChurnResponse, HealthResponse
from src.api.model import churn_model
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


# ── Lifespan: runs on startup and shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — runs before first request
    logger.info("Starting up — loading model...")
    churn_model.load()
    logger.info("App ready ✓")
    yield
    # SHUTDOWN — runs when server stops
    logger.info("Shutting down...")


# ── App definition ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    Used by Docker, Kubernetes, and load balancers to verify app is alive.
    Always implement this — infrastructure depends on it.
    """
    return HealthResponse(
        status="ok",
        model_loaded=churn_model.is_loaded,
        model_version=str(churn_model.model_version),
        threshold=round(churn_model.threshold, 4),
    )


@app.post(
    "/predict",
    response_model=ChurnResponse,
    tags=["Prediction"],
    summary="Predict churn for a single customer",
)
async def predict(request: ChurnRequest):
    """
    Accepts customer data, returns churn probability and risk level.

    - **churn_probability**: 0.0 to 1.0
    - **churn_prediction**: True if probability >= threshold
    - **risk_level**: LOW / MEDIUM / HIGH
    """
    if not churn_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Try again in a moment."
        )

    try:
        # Convert Pydantic model → dict → prediction
        customer_data = request.model_dump()
        result = churn_model.predict(customer_data)

        logger.info(
            f"Prediction: prob={result['churn_probability']} "
            f"risk={result['risk_level']}"
        )

        return ChurnResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """API info — useful for quick sanity check."""
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }