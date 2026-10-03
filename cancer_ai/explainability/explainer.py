"""
Clinical Explainability Engine (XAI) for Breast Cancer Sentinel 2.0.
Calculates statistical deviations against benign population baselines,
percentile ranks, and radar chart coordinates.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats as sp_stats

from cancer_ai.config import settings
from cancer_ai.schemas import FeatureDeviation

CATEGORIES = [
    ("Radius", "radius"),
    ("Texture", "texture"),
    ("Perimeter", "perimeter"),
    ("Area", "area"),
    ("Smoothness", "smoothness"),
    ("Compactness", "compactness"),
    ("Concavity", "concavity"),
    ("Concave Points", "concave points"),
    ("Symmetry", "symmetry"),
    ("Fractal Dim", "fractal_dimension")
]


class ClinicalExplainer:
    """
    Translates mathematical model inferences into transparent clinical biomarkers.
    """

    def __init__(self, stats_path: Path = None):
        self.stats_path = Path(stats_path or settings.FEATURE_STATS_PATH)
        self.stats: Dict[str, Any] = {}
        self._load_stats()

    def _load_stats(self):
        if self.stats_path.exists():
            with open(self.stats_path, "r") as f:
                self.stats = json.load(f)

    def analyze_deviations(self, patient_dict: Dict[str, Any]) -> List[FeatureDeviation]:
        """
        Compares patient features against benign population distribution.
        Returns top 5 features with highest positive z-score deviations toward malignancy.
        """
        deviations = []
        benign_mean = self.stats.get("benign_mean", {})
        benign_std = self.stats.get("benign_std", {})

        for feat, b_mean in benign_mean.items():
            b_std = max(benign_std.get(feat, 1.0), 1e-5)
            # Find patient val
            val = patient_dict.get(feat)
            if val is None and "concave points" in feat:
                alt = feat.replace("concave points", "concave_points")
                val = patient_dict.get(alt, b_mean)
            val = float(val if val is not None else b_mean)

            z = (val - b_mean) / b_std
            pct = float(sp_stats.norm.cdf(z) * 100.0)

            if z >= 3.0:
                impact = "CRITICALLY_ELEVATED (>+3.0σ Above Benign Baseline)"
            elif z >= 1.8:
                impact = "ELEVATED (Atypical Nuclear Morphometry)"
            else:
                impact = "WITHIN_NORMAL_BENIGN_RANGE"

            deviations.append(
                FeatureDeviation(
                    feature_name=feat,
                    patient_value=round(val, 4),
                    benign_mean=round(b_mean, 4),
                    z_score=round(z, 2),
                    percentile=round(pct, 1),
                    impact=impact
                )
            )

        # Sort by highest positive Z-score (greatest divergence toward malignant morphometry)
        deviations.sort(key=lambda d: d.z_score, reverse=True)
        return deviations[:5]

    def compute_radar_values(self, patient_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes min-max normalized values in [0, 1] across the 10 morphological categories
        for radar / polar visualization.
        """
        min_vals = self.stats.get("global_min", {})
        max_vals = self.stats.get("global_max", {})
        radar = {}

        for label, prefix in CATEGORIES:
            mean_key = f"{prefix}_mean"
            val = patient_dict.get(mean_key)
            if val is None and "concave points" in mean_key:
                val = patient_dict.get("concave_points_mean")
            val = float(val or 0.0)

            g_min = min_vals.get(mean_key, 0.0)
            g_max = max_vals.get(mean_key, 1.0)
            norm = (val - g_min) / max(g_max - g_min, 1e-5)
            radar[label] = round(float(np.clip(norm, 0.0, 1.0)), 3)

        return radar
