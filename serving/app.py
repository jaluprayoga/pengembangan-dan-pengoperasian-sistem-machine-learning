"""FastAPI Model Serving API for Pima Indians Diabetes Prediction.

This module loads the exported TensorFlow SavedModel and serves inference requests
via REST endpoints while exporting metrics to Prometheus for monitoring.
"""

import os
from typing import Any, Dict, List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import tensorflow as tf

app = FastAPI(
    title="Diabetes Prediction System API",
    description="MLOps Model Serving API for Pima Indians Diabetes Prediction deployed on Railway.",
    version="1.0.0",
)

# Instrument FastAPI with Prometheus metrics exporter
Instrumentator().instrument(app).expose(app)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DEFAULT_MODEL_DIR = (
    os.path.join(PROJECT_ROOT, "serving_model")
    if os.path.exists(os.path.join(PROJECT_ROOT, "serving_model"))
    else "serving_model"
)
MODEL_DIR = os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR)
model = None

NUMERICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


class DiabetesFeatures(BaseModel):
    """Input features model for a single patient record."""

    Pregnancies: float = Field(..., example=6.0, description="Number of times pregnant")
    Glucose: float = Field(..., example=148.0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., example=72.0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., example=35.0, description="Triceps skin fold thickness (mm)")
    Insulin: float = Field(..., example=0.0, description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., example=33.6, description="Body mass index (weight in kg/(height in m)^2)")
    DiabetesPedigreeFunction: float = Field(
        ..., example=0.627, description="Diabetes pedigree function score"
    )
    Age: float = Field(..., example=50.0, description="Age in years")


class DiabetesRequest(BaseModel):
    """Input payload model supporting single or batch patient records."""

    inputs: Union[DiabetesFeatures, List[DiabetesFeatures]]


class PredictionResult(BaseModel):
    """Output payload model for single diabetes prediction result."""

    features: Dict[str, float]
    prediction: str
    outcome: int
    probability: float
    confidence: float
    risk_level: str


class DiabetesResponse(BaseModel):
    """API response model containing list of prediction results."""

    status: str
    predictions: List[PredictionResult]


def get_latest_model_path(base_dir: str) -> str:
    """Find the latest timestamped directory in the serving model folder.

    Args:
        base_dir (str): Base model directory.

    Returns:
        str: Absolute or relative path to latest model directory.
    """
    if not os.path.exists(base_dir):
        return base_dir

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        return base_dir

    latest = max(subdirs, key=os.path.getmtime)
    return latest


@app.on_event("startup")
def load_model():
    """Load TensorFlow SavedModel upon application startup."""
    global model  # pylint: disable=global-statement
    try:
        model_path = get_latest_model_path(MODEL_DIR)
        print(f"Loading TensorFlow SavedModel from: {model_path}")
        if os.path.exists(model_path):
            model = tf.saved_model.load(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model path {model_path} does not exist yet.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error loading model: {exc}")


def _serialize_to_example(record: Dict[str, Any]) -> bytes:
    """Serialize patient numerical features into tf.train.Example bytes.

    Args:
        record (Dict[str, Any]): Dictionary of numerical features.

    Returns:
        bytes: Serialized Example protobuf string.
    """
    feature_dict = {}
    for key in NUMERICAL_FEATURES:
        val = float(record[key])
        feature_dict[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[val])
        )
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


@app.get("/")
def read_root():
    """Root welcome endpoint providing service information and API endpoints."""
    return {
        "service": "Diabetes Prediction ML System",
        "dataset": "Pima Indians Diabetes Dataset",
        "platform": "Railway Cloud Deployment",
        "status": "Running",
        "endpoints": {
            "swagger_docs": "/docs (Interactive API Inference UI)",
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint for container probes and monitoring."""
    global model  # pylint: disable=global-statement
    if model is None:
        load_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=DiabetesResponse)
def predict_diabetes(payload: DiabetesRequest):
    """Predict diabetes risk based on patient measurements.

    Args:
        payload (DiabetesRequest): Patient features payload.

    Returns:
        DiabetesResponse: Predicted outcome and risk assessment.
    """
    global model  # pylint: disable=global-statement
    if model is None:
        load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure TFX pipeline has pushed a valid model.",
        )

    # Standardize input into list of dictionaries
    raw_inputs = (
        [payload.inputs.dict()]
        if isinstance(payload.inputs, DiabetesFeatures)
        else [item.dict() for item in payload.inputs]
    )

    try:
        serialized_examples = [_serialize_to_example(item) for item in raw_inputs]
        input_tensor = tf.constant(serialized_examples, dtype=tf.string)

        # Call serving_default signature
        if hasattr(model, "signatures") and "serving_default" in model.signatures:
            infer_fn = model.signatures["serving_default"]
            preds_dict = infer_fn(examples=input_tensor)
            if "outputs" in preds_dict:
                preds = preds_dict["outputs"].numpy()
            elif "output_0" in preds_dict:
                preds = preds_dict["output_0"].numpy()
            else:
                preds = next(iter(preds_dict.values())).numpy()
        else:
            preds = model(input_tensor).numpy()

        results = []
        for feat_dict, pred_val in zip(raw_inputs, preds):
            prob = (
                float(pred_val[0])
                if hasattr(pred_val, "__len__")
                else float(pred_val)
            )
            is_diabetic = int(prob >= 0.5)
            label = "Diabetic" if is_diabetic == 1 else "Non-Diabetic"
            confidence = prob if is_diabetic == 1 else (1.0 - prob)

            if prob >= 0.70:
                risk_level = "High"
            elif prob >= 0.40:
                risk_level = "Moderate"
            else:
                risk_level = "Low"

            results.append(
                PredictionResult(
                    features=feat_dict,
                    prediction=label,
                    outcome=is_diabetic,
                    probability=round(prob, 4),
                    confidence=round(confidence, 4),
                    risk_level=risk_level,
                )
            )

        return DiabetesResponse(status="success", predictions=results)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc
