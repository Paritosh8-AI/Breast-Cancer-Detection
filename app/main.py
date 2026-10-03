"""
Backwards-compatible Streamlit interface for Breast Cancer Predictor.
Updated with dynamic relative paths and modern model support.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cancer_ai.service import service
from cancer_ai.schemas import PatientFNAInput

# Forward directly to the modern Next-Gen dashboard
import cancer_ai.dashboard.app as modern_dashboard
