"""
Pydantic v2 validation models for Breast Cancer Sentinel 2.0.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid
import time


class DiagnosisEnum(str, Enum):
    BENIGN = "Benign"
    MALIGNANT = "Malignant"
    EQUIVOCAL = "Equivocal / Indeterminate"


class ClinicalRiskLevel(str, Enum):
    LOW_RISK = "Low Risk (Likely Benign)"
    INTERMEDIATE_RISK = "Intermediate / Equivocal Risk"
    HIGH_RISK = "High Risk (Suspicion of Malignancy)"


class ClinicalAction(str, Enum):
    ROUTINE_SCREENING = "ROUTINE_SCREENING"
    ULTRASOUND_CORE_BIOPSY = "ULTRASOUND_CORE_BIOPSY"
    URGENT_ONCOLOGY_REFERRAL = "URGENT_ONCOLOGY_REFERRAL"


class PatientFNAInput(BaseModel):
    patient_id: Optional[str] = Field(default_factory=lambda: f"PT-{uuid.uuid4().hex[:6].upper()}")
    
    # 10 Mean Nuclear Characteristics
    radius_mean: float = Field(..., ge=0.0, description="Mean radius of cell nuclei")
    texture_mean: float = Field(..., ge=0.0, description="Mean texture (std dev of gray-scale values)")
    perimeter_mean: float = Field(..., ge=0.0, description="Mean perimeter of cell nuclei")
    area_mean: float = Field(..., ge=0.0, description="Mean area of cell nuclei")
    smoothness_mean: float = Field(..., ge=0.0, description="Mean smoothness (local variation in radius)")
    compactness_mean: float = Field(..., ge=0.0, description="Mean compactness (perimeter^2 / area - 1.0)")
    concavity_mean: float = Field(..., ge=0.0, description="Mean concavity (severity of concave portions)")
    concave_points_mean: float = Field(..., ge=0.0, alias="concave points_mean", description="Mean number of concave portions of contour")
    symmetry_mean: float = Field(..., ge=0.0, description="Mean nuclear symmetry")
    fractal_dimension_mean: float = Field(..., ge=0.0, description="Mean fractal dimension (coastline approximation - 1)")

    # 10 Standard Error Features
    radius_se: float = Field(..., ge=0.0)
    texture_se: float = Field(..., ge=0.0)
    perimeter_se: float = Field(..., ge=0.0)
    area_se: float = Field(..., ge=0.0)
    smoothness_se: float = Field(..., ge=0.0)
    compactness_se: float = Field(..., ge=0.0)
    concavity_se: float = Field(..., ge=0.0)
    concave_points_se: float = Field(..., ge=0.0, alias="concave points_se")
    symmetry_se: float = Field(..., ge=0.0)
    fractal_dimension_se: float = Field(..., ge=0.0)

    # 10 "Worst" / Largest Nuclear Features
    radius_worst: float = Field(..., ge=0.0)
    texture_worst: float = Field(..., ge=0.0)
    perimeter_worst: float = Field(..., ge=0.0)
    area_worst: float = Field(..., ge=0.0)
    smoothness_worst: float = Field(..., ge=0.0)
    compactness_worst: float = Field(..., ge=0.0)
    concavity_worst: float = Field(..., ge=0.0)
    concave_points_worst: float = Field(..., ge=0.0, alias="concave points_worst")
    symmetry_worst: float = Field(..., ge=0.0)
    fractal_dimension_worst: float = Field(..., ge=0.0)

    model_config = ConfigDict(populate_by_name=True)


class FeatureDeviation(BaseModel):
    feature_name: str
    patient_value: float
    benign_mean: float
    z_score: float
    percentile: float
    impact: str  # NORMAL, ELEVATED, HIGHLY_ELEVATED


class PredictionResult(BaseModel):
    patient_id: str
    diagnosis: DiagnosisEnum
    malignancy_probability: float = Field(ge=0.0, le=1.0)
    benign_probability: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=100.0, description="Calibrated risk index 0 to 100")
    risk_level: ClinicalRiskLevel
    clinical_action: ClinicalAction
    recommendation: str
    top_biomarker_drivers: List[FeatureDeviation]
    radar_values: Dict[str, float]
    latency_ms: float
    evaluated_at: float = Field(default_factory=time.time)


class BatchPatientInput(BaseModel):
    patients: List[PatientFNAInput]


class BatchPredictionResult(BaseModel):
    total_patients: int
    malignant_count: int
    equivocal_count: int
    benign_count: int
    avg_latency_ms: float
    results: List[PredictionResult]


class ModelMetricsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    sensitivity_recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    clinical_threshold: float
    confusion_matrix: List[List[int]]


class SystemHealth(BaseModel):
    status: str
    version: str
    model_loaded: bool
    total_inferences: int
    avg_latency_ms: float
    uptime_seconds: float
