# importing the necessary libraries
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import json
import os
import pandas as pd
import uvicorn
import logging

# Set up logging for error handling visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initializing FastAPI and Jinja2
app = FastAPI(
    title="Electricity Cost Predictor API",
    description=(
        "Machine Learning API for predicting electricity costs based on building metrics. "
        "Built with a Gradient Boosting Regressor achieving R²=0.96 on a held-out test set. "
        "Use the /api/predict endpoint for programmatic access or the web UI at /."
    ),
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")

# ── Pydantic model for Swagger API docs + validation ─────────────────────────
class PredictionRequest(BaseModel):
    site_area: float = Field(..., description="Site area in square meters", gt=0)
    structure_type: str = Field(..., description="Type of structure (Residential, Commercial, Mixed-use)")
    water_consumption: float = Field(..., description="Water consumption in liters", ge=0)
    recycling_rate: float = Field(..., description="Recycling rate percentage (0–100)", ge=0, le=100)
    utilization_rate: float = Field(..., description="Utilization rate ratio (0–1)", ge=0, le=1)
    air_quality_index: float = Field(..., description="Air Quality Index (AQI)", ge=0)
    issue_resolution_time: float = Field(..., description="Issue resolution time in hours", ge=0)
    resident_count: int = Field(..., description="Total number of residents", ge=1)

# ── Load the trained pipeline ─────────────────────────────────────────────────
try:
    model = joblib.load("model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# ── Load model performance metrics ───────────────────────────────────────────
try:
    with open("model_metrics.json") as f:
        model_metrics = json.load(f)
    logger.info(f"Metrics loaded: {model_metrics}")
except Exception as e:
    logger.warning(f"Could not load model_metrics.json, using defaults: {e}")
    model_metrics = {"r2": "N/A", "rmse": "N/A", "mae": "N/A"}

# ── Input validation helper ───────────────────────────────────────────────────
def validate_inputs(
    site_area: float,
    structure_type: str,
    water_consumption: float,
    recycling_rate: float,
    utilization_rate: float,
    air_quality_index: float,
    issue_resolution_time: float,
    resident_count: float,
) -> str | None:
    """Returns a human-readable error string if validation fails, else None."""
    valid_structure_types = {"Residential", "Commercial", "Mixed-use"}

    if site_area <= 0:
        return "Site Area must be greater than 0."
    if structure_type not in valid_structure_types:
        return f"Structure Type must be one of: {', '.join(valid_structure_types)}."
    if water_consumption < 0:
        return "Water Consumption cannot be negative."
    if not (0 <= recycling_rate <= 100):
        return "Recycling Rate must be between 0 and 100."
    if not (0 <= utilization_rate <= 1):
        return "Utilization Rate must be between 0 and 1 (e.g., 0.75 for 75%)."
    if air_quality_index < 0:
        return "Air Quality Index cannot be negative."
    if issue_resolution_time < 0:
        return "Issue Resolution Time cannot be negative."
    if resident_count < 1:
        return "Resident Count must be at least 1."
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def main_page(request: Request):
    """Serve the prediction web UI."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "metrics": model_metrics,
    })


@app.post("/predict", response_class=HTMLResponse, tags=["UI"])
async def predict(
    request: Request,
    site_area: float = Form(...),
    structure_type: str = Form(...),
    water_consumption: float = Form(...),
    recycling_rate: float = Form(...),
    utilization_rate: float = Form(...),
    air_quality_index: float = Form(...),
    issue_resolution_time: float = Form(...),
    resident_count: float = Form(...),
):
    """Handle form submission and return a prediction."""
    try:
        if not model:
            raise RuntimeError("Machine learning model is not available.")

        # Validate all inputs with friendly messages
        error_msg = validate_inputs(
            site_area, structure_type, water_consumption,
            recycling_rate, utilization_rate, air_quality_index,
            issue_resolution_time, resident_count,
        )
        if error_msg:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "prediction": f"⚠️ Input Error: {error_msg}",
                "error": True,
                "metrics": model_metrics,
            })

        # Build DataFrame matching training column names
        input_data = pd.DataFrame([{
            "site area": site_area,
            "structure type": structure_type,
            "water consumption": water_consumption,
            "recycling rate": recycling_rate,
            "utilisation rate": utilization_rate,
            "air qality index": air_quality_index,      # matches training typo
            "issue reolution time": issue_resolution_time,  # matches training typo
            "resident count": resident_count,
        }])

        prediction = model.predict(input_data)[0]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Predicted Electricity Cost: ${prediction:,.2f}",
            "metrics": model_metrics,
        })

    except ValueError as ve:
        logger.warning(f"Validation Error: {ve}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"⚠️ Validation Error: {str(ve)}",
            "error": True,
            "metrics": model_metrics,
        })
    except Exception as error:
        logger.error(f"Prediction Error: {error}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": "An unexpected error occurred during prediction. Please try again.",
            "error": True,
            "metrics": model_metrics,
        })


@app.post("/api/predict", tags=["API"])
async def api_predict(request_data: PredictionRequest):
    """
    JSON API endpoint for programmatic predictions.

    Returns the predicted electricity cost in USD given building metrics.
    Swagger documentation is available at **/docs**.

    **Example request body:**
    ```json
    {
      "site_area": 1200.5,
      "structure_type": "Residential",
      "water_consumption": 3500.0,
      "recycling_rate": 65.0,
      "utilization_rate": 0.8,
      "air_quality_index": 45.0,
      "issue_resolution_time": 2.5,
      "resident_count": 120
    }
    ```
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is currently unavailable")

    try:
        input_data = pd.DataFrame([{
            "site area": request_data.site_area,
            "structure type": request_data.structure_type,
            "water consumption": request_data.water_consumption,
            "recycling rate": request_data.recycling_rate,
            "utilisation rate": request_data.utilization_rate,
            "air qality index": request_data.air_quality_index,
            "issue reolution time": request_data.issue_resolution_time,
            "resident count": request_data.resident_count,
        }])

        prediction = model.predict(input_data)[0]
        return {
            "predicted_cost_usd": round(float(prediction), 2),
            "model_version": "1.0.0",
        }

    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please check your inputs.")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Returns model performance metrics from the test set evaluation.

    - **r2**: R² score (coefficient of determination, 1.0 = perfect)
    - **rmse**: Root Mean Squared Error in USD
    - **mae**: Mean Absolute Error in USD
    """
    return {
        "r2_score": model_metrics.get("r2"),
        "rmse_usd": model_metrics.get("rmse"),
        "mae_usd": model_metrics.get("mae"),
        "algorithm": "Gradient Boosting Regressor (with RandomizedSearchCV tuning)",
        "test_split": "20%",
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Health check endpoint. Returns 200 if the service is up and the model is loaded."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "model_loaded": True,
        "metrics_loaded": model_metrics.get("r2") != "N/A",
    }
