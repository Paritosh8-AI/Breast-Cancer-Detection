"""
REST API Endpoints for Breast Cancer Sentinel 2.0.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from cancer_ai.schemas import (
    PatientFNAInput,
    PredictionResult,
    BatchPatientInput,
    BatchPredictionResult,
    ModelMetricsResponse,
    SystemHealth
)
from cancer_ai.service import service

router = APIRouter(prefix="/api/v1", tags=["Breast Cancer Clinical AI"])


@router.post("/predict", response_model=PredictionResult)
def predict_patient(patient: PatientFNAInput):
    """
    Evaluates a single patient FNA sample across 30 nuclear characteristics.
    Returns calibrated malignancy risk, clinical action, and XAI biomarker attribution.
    """
    try:
        return service.evaluate_patient(patient)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic scoring failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(batch: BatchPatientInput):
    """
    Evaluates a cohort batch of patient samples with clinical population distribution.
    """
    try:
        return service.evaluate_batch(batch)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.get("/metrics", response_model=ModelMetricsResponse)
def get_model_metrics():
    """
    Returns clinical diagnostic accuracy, sensitivity, specificity, and ROC-AUC metrics.
    """
    return service.get_metrics()


@router.get("/health", response_model=SystemHealth)
def health_check():
    """
    Returns system status, active inference counts, and latency telemetry.
    """
    return service.get_health()


@router.get("/sample-cases")
def get_sample_cases() -> Dict[str, Dict[str, Any]]:
    """
    Provides typical benign, equivocal, and malignant FNA cases for clinical demonstration.
    """
    return {
        "typical_benign": {
            "radius_mean": 12.45, "texture_mean": 15.70, "perimeter_mean": 82.57, "area_mean": 477.1,
            "smoothness_mean": 0.084, "compactness_mean": 0.064, "concavity_mean": 0.024, "concave points_mean": 0.015,
            "symmetry_mean": 0.178, "fractal_dimension_mean": 0.059,
            "radius_se": 0.28, "texture_se": 0.89, "perimeter_se": 1.84, "area_se": 22.8,
            "smoothness_se": 0.005, "compactness_se": 0.014, "concavity_se": 0.012, "concave points_se": 0.008,
            "symmetry_se": 0.016, "fractal_dimension_se": 0.002,
            "radius_worst": 14.15, "texture_worst": 21.08, "perimeter_worst": 93.99, "area_worst": 612.2,
            "smoothness_worst": 0.114, "compactness_worst": 0.145, "concavity_worst": 0.106, "concave points_worst": 0.058,
            "symmetry_worst": 0.257, "fractal_dimension_worst": 0.074
        },
        "typical_malignant": {
            "radius_mean": 17.99, "texture_mean": 21.60, "perimeter_mean": 122.8, "area_mean": 1001.0,
            "smoothness_mean": 0.118, "compactness_mean": 0.277, "concavity_mean": 0.300, "concave points_mean": 0.147,
            "symmetry_mean": 0.242, "fractal_dimension_mean": 0.078,
            "radius_se": 1.095, "texture_se": 0.905, "perimeter_se": 8.589, "area_se": 153.4,
            "smoothness_se": 0.006, "compactness_se": 0.049, "concavity_se": 0.053, "concave points_se": 0.015,
            "symmetry_se": 0.030, "fractal_dimension_se": 0.006,
            "radius_worst": 25.38, "texture_worst": 28.50, "perimeter_worst": 184.6, "area_worst": 2019.0,
            "smoothness_worst": 0.162, "compactness_worst": 0.665, "concavity_worst": 0.711, "concave points_worst": 0.265,
            "symmetry_worst": 0.460, "fractal_dimension_worst": 0.118
        }
    }
