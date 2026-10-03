"""
Robust Clinical FNA Cytology Preprocessor.
Applies standardization based on stored parameters with zero third-party pickle issues.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

from cancer_ai.config import settings

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]


class CancerDataPreprocessor:
    """
    Standardizes FNA cytology measurements for machine learning inference.
    """

    def __init__(self, params_path: Union[str, Path] = None):
        self.params_path = Path(params_path or settings.PREPROCESSOR_PARAMS_PATH)
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.features: List[str] = FEATURE_NAMES
        self._load_parameters()

    def _load_parameters(self):
        if self.params_path.exists():
            with open(self.params_path, "r") as f:
                data = json.load(f)
                self.mean = np.array(data["mean"], dtype=np.float32)
                self.scale = np.array(data["scale"], dtype=np.float32)
                self.features = data.get("features", self.features)
        else:
            # Fallback zero mean, unit scale
            self.mean = np.zeros(len(self.features), dtype=np.float32)
            self.scale = np.ones(len(self.features), dtype=np.float32)

    def extract_vector(self, data: Dict[str, Any]) -> np.ndarray:
        """Extracts ordered raw feature vector from patient dictionary."""
        row = []
        for feat in self.features:
            # Check direct name or underscore variant for concave points
            val = data.get(feat)
            if val is None and "concave points" in feat:
                alt = feat.replace("concave points", "concave_points")
                val = data.get(alt, 0.0)
            elif val is None:
                val = 0.0
            row.append(float(val))
        return np.array(row, dtype=np.float32)

    def transform(self, data: Dict[str, Any]) -> np.ndarray:
        """Transforms patient dictionary into normalized (1, 30) array."""
        raw_vec = self.extract_vector(data)
        scaled_vec = (raw_vec - self.mean) / np.maximum(self.scale, 1e-7)
        return np.expand_dims(scaled_vec, axis=0)

    def transform_batch(self, batch: List[Dict[str, Any]]) -> np.ndarray:
        """Transforms a batch of patient dicts into (N, 30) array."""
        raw_mat = np.stack([self.extract_vector(d) for d in batch])
        scaled_mat = (raw_mat - self.mean) / np.maximum(self.scale, 1e-7)
        return scaled_mat
