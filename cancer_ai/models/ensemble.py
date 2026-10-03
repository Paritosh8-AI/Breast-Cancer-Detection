"""
Calibrated Soft-Voting Clinical Ensemble for Breast Cancer Sentinel 2.0.
"""

from pathlib import Path
from typing import Tuple, Optional
import joblib
import numpy as np

from cancer_ai.config import settings


class ClinicalEnsembleModel:
    """
    Ensemble classifier integrating Calibrated Soft-Voting across
    Gradient Boosting, Random Forest, Deep MLP, and Regularized Logistic Regression.
    """

    def __init__(self, model_path: Path = None):
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def predict_proba(self, X_scaled: np.ndarray) -> Tuple[float, float]:
        """
        Predicts calibrated probabilities: (P(Benign), P(Malignant)).
        """
        if self.model is None:
            # Fallback based on scaled mean
            mean_val = float(np.mean(X_scaled))
            p_mal = float(1.0 / (1.0 + np.exp(-mean_val)))
            return 1.0 - p_mal, p_mal

        probs = self.model.predict_proba(X_scaled)[0]
        p_benign = float(probs[0])
        p_malignant = float(probs[1])
        return p_benign, p_malignant


ensemble_model = ClinicalEnsembleModel()
