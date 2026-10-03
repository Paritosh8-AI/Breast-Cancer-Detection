"""
Configuration Module for Breast Cancer Sentinel 2.0.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CANCER_AI_",
        extra="ignore"
    )

    APP_NAME: str = "Breast Cancer Sentinel 2.0 - Clinical AI Decision Support"
    VERSION: str = "2.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DASHBOARD_PORT: int = 8501

    BASE_DIR: Path = PROJECT_ROOT
    DATA_PATH: Path = DATA_DIR / "data.csv"
    ARTIFACTS_PATH: Path = ARTIFACTS_DIR
    MODEL_PATH: Path = ARTIFACTS_DIR / "ensemble_model.joblib"
    SCALER_PATH: Path = ARTIFACTS_DIR / "scaler.joblib"
    PREPROCESSOR_PARAMS_PATH: Path = ARTIFACTS_DIR / "preprocessor_params.json"
    FEATURE_STATS_PATH: Path = ARTIFACTS_DIR / "feature_stats.json"
    METADATA_PATH: Path = ARTIFACTS_DIR / "model_metadata.json"

    # Clinical Decision Cutoffs
    BENIGN_CUTOFF: float = 0.20
    MALIGNANT_CUTOFF: float = 0.55
    OPTIMAL_CLINICAL_THRESHOLD: float = 0.40


settings = Settings()
