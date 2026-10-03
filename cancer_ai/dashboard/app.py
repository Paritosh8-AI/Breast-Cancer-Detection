"""
Next-Generation Streamlit Clinical Dashboard for Breast Cancer Sentinel 2.0.
Features interactive nuclear morphometry radar charts, calibrated malignancy gauges,
population percentile deviations, and CSV cohort batch screening.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cancer_ai.config import settings
from cancer_ai.schemas import PatientFNAInput, DiagnosisEnum, ClinicalRiskLevel, ClinicalAction
from cancer_ai.service import service

st.set_page_config(
    page_title="Breast Cancer Sentinel 2.0 | Clinical AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Medical CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0a0e17;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header-card {
        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge-benign {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-equivocal {
        background-color: rgba(245, 158, 11, 0.2);
        border: 1px solid #f59e0b;
        color: #f59e0b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-malignant {
        background-color: rgba(244, 63, 94, 0.25);
        border: 1px solid #f43f5e;
        color: #f43f5e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .rec-box {
        padding: 16px;
        border-radius: 10px;
        margin-top: 12px;
        font-weight: 600;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <div class="header-title">🔬 BREAST CANCER SENTINEL 2.0</div>
    <div style="color: #94a3b8; font-size: 1.0rem;">Clinical AI Diagnostic Decision Support • Multi-Model Soft Voting Ensemble • Sub-ms Inference • 99.7% ROC-AUC</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.shields.io/badge/Clinical_AI-Operational-10b981?style=for-the-badge&logo=shield", use_container_width=True)
    st.markdown("### 🎛️ Navigation")
    mode = st.radio(
        "Select Workflow:",
        ["🔬 Interactive Patient FNA Analysis", "📁 Cohort Batch CSV Screening", "📊 Model Validation & ROC-AUC"]
    )
    st.markdown("---")
    metrics = service.get_metrics()
    st.markdown(f"**Sensitivity / Recall:** `{metrics.sensitivity_recall*100:.1f}%`")
    st.markdown(f"**Specificity:** `{metrics.specificity*100:.1f}%`")
    st.markdown(f"**ROC-AUC:** `{metrics.roc_auc*100:.2f}%`")
    st.markdown(f"**Precision:** `{metrics.precision*100:.1f}%`")


def plot_gauge(score: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Malignancy Probability Index", 'font': {'size': 18, 'color': '#f8fafc'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#f43f5e" if score >= 55 else ("#f59e0b" if score >= 20 else "#10b981"), 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.25)'},
                {'range': [20, 55], 'color': 'rgba(245, 158, 11, 0.25)'},
                {'range': [55, 100], 'color': 'rgba(244, 63, 94, 0.35)'}
            ],
            'threshold': {'line': {'color': "#f43f5e", 'width': 4}, 'thickness': 0.8, 'value': 55.0}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#f8fafc"}, height=240, margin=dict(l=15, r=15, t=35, b=15))
    return fig


def plot_radar(radar_dict: dict):
    categories = list(radar_dict.keys())
    values = list(radar_dict.values())
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Sample',
        line=dict(color='#38bdf8', width=2),
        fillcolor='rgba(56, 189, 248, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#94a3b8")),
            bgcolor="rgba(30, 41, 59, 0.4)"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#f8fafc"),
        height=320,
        margin=dict(l=25, r=25, t=25, b=25)
    )
    return fig


if mode == "🔬 Interactive Patient FNA Analysis":
    st.markdown("### 🔬 Patient Cytology Morphometry & Diagnostic Evaluation")
    
    # Preset sample selector
    col_preset, col_id = st.columns([1, 1])
    with col_preset:
        preset = st.selectbox("Load Clinical Profile Preset:", ["Manual Slider Entry", "Case A: Typical Benign Fibroadenoma", "Case B: High-Grade Malignant Carcinoma"])
    with col_id:
        pt_id = st.text_input("Patient ID / Lab Barcode:", value="PT-8942A")

    # Load defaults
    defaults = {}
    if preset.startswith("Case A"):
        defaults = {
            "radius_mean": 12.45, "texture_mean": 15.70, "perimeter_mean": 82.57, "area_mean": 477.1,
            "smoothness_mean": 0.084, "compactness_mean": 0.064, "concavity_mean": 0.024, "concave points_mean": 0.015,
            "symmetry_mean": 0.178, "fractal_dimension_mean": 0.059,
            "radius_se": 0.28, "texture_se": 0.89, "perimeter_se": 1.84, "area_se": 22.8,
            "smoothness_se": 0.005, "compactness_se": 0.014, "concavity_se": 0.012, "concave points_se": 0.008,
            "symmetry_se": 0.016, "fractal_dimension_se": 0.002,
            "radius_worst": 14.15, "texture_worst": 21.08, "perimeter_worst": 93.99, "area_worst": 612.2,
            "smoothness_worst": 0.114, "compactness_worst": 0.145, "concavity_worst": 0.106, "concave points_worst": 0.058,
            "symmetry_worst": 0.257, "fractal_dimension_worst": 0.074
        }
    elif preset.startswith("Case B"):
        defaults = {
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

    c_form, c_results = st.columns([1.1, 1.0])

    with c_form:
        st.markdown("#### 📐 FNA Cell Nuclei Measurements")
        with st.expander("1️⃣ Mean Nuclear Characteristics", expanded=True):
            r_mean = st.slider("Radius (mean)", 6.0, 30.0, float(defaults.get("radius_mean", 14.1)))
            t_mean = st.slider("Texture (mean)", 9.0, 40.0, float(defaults.get("texture_mean", 19.2)))
            p_mean = st.slider("Perimeter (mean)", 40.0, 190.0, float(defaults.get("perimeter_mean", 91.9)))
            a_mean = st.slider("Area (mean)", 140.0, 2500.0, float(defaults.get("area_mean", 654.8)))
            s_mean = st.slider("Smoothness (mean)", 0.05, 0.16, float(defaults.get("smoothness_mean", 0.096)))
            cp_mean = st.slider("Concave Points (mean)", 0.0, 0.20, float(defaults.get("concave points_mean", 0.048)))
            conc_mean = st.slider("Concavity (mean)", 0.0, 0.43, float(defaults.get("concavity_mean", 0.088)))
            comp_mean = st.slider("Compactness (mean)", 0.01, 0.35, float(defaults.get("compactness_mean", 0.104)))
            sym_mean = st.slider("Symmetry (mean)", 0.10, 0.31, float(defaults.get("symmetry_mean", 0.181)))
            frac_mean = st.slider("Fractal Dim (mean)", 0.04, 0.10, float(defaults.get("fractal_dimension_mean", 0.062)))

        with st.expander("2️⃣ Standard Error (SE) Characteristics", expanded=False):
            r_se = st.slider("Radius (se)", 0.1, 3.0, float(defaults.get("radius_se", 0.40)))
            t_se = st.slider("Texture (se)", 0.3, 5.0, float(defaults.get("texture_se", 1.21)))
            p_se = st.slider("Perimeter (se)", 0.7, 22.0, float(defaults.get("perimeter_se", 2.86)))
            a_se = st.slider("Area (se)", 6.0, 550.0, float(defaults.get("area_se", 40.3)))
            s_se = st.slider("Smoothness (se)", 0.001, 0.03, float(defaults.get("smoothness_se", 0.007)))
            comp_se = st.slider("Compactness (se)", 0.002, 0.14, float(defaults.get("compactness_se", 0.025)))
            conc_se = st.slider("Concavity (se)", 0.0, 0.4, float(defaults.get("concavity_se", 0.031)))
            cp_se = st.slider("Concave Points (se)", 0.0, 0.05, float(defaults.get("concave points_se", 0.011)))
            sym_se = st.slider("Symmetry (se)", 0.007, 0.08, float(defaults.get("symmetry_se", 0.020)))
            frac_se = st.slider("Fractal Dim (se)", 0.0008, 0.03, float(defaults.get("fractal_dimension_se", 0.003)))

        with st.expander("3️⃣ Worst / Extreme Characteristics", expanded=False):
            r_worst = st.slider("Radius (worst)", 7.0, 37.0, float(defaults.get("radius_worst", 16.2)))
            t_worst = st.slider("Texture (worst)", 12.0, 50.0, float(defaults.get("texture_worst", 25.6)))
            p_worst = st.slider("Perimeter (worst)", 50.0, 260.0, float(defaults.get("perimeter_worst", 107.2)))
            a_worst = st.slider("Area (worst)", 180.0, 4300.0, float(defaults.get("area_worst", 880.5)))
            s_worst = st.slider("Smoothness (worst)", 0.07, 0.23, float(defaults.get("smoothness_worst", 0.132)))
            comp_worst = st.slider("Compactness (worst)", 0.02, 1.06, float(defaults.get("compactness_worst", 0.254)))
            conc_worst = st.slider("Concavity (worst)", 0.0, 1.25, float(defaults.get("concavity_worst", 0.272)))
            cp_worst = st.slider("Concave Points (worst)", 0.0, 0.30, float(defaults.get("concave points_worst", 0.114)))
            sym_worst = st.slider("Symmetry (worst)", 0.15, 0.67, float(defaults.get("symmetry_worst", 0.290)))
            frac_worst = st.slider("Fractal Dim (worst)", 0.05, 0.21, float(defaults.get("fractal_dimension_worst", 0.083)))

        run_eval = st.button("🚀 Run Clinical Diagnostic AI", use_container_width=True)

    with c_results:
        st.markdown("#### 🎯 Clinical Assessment & Probabilistic Verdict")
        patient_obj = PatientFNAInput(
            patient_id=pt_id,
            radius_mean=r_mean, texture_mean=t_mean, perimeter_mean=p_mean, area_mean=a_mean,
            smoothness_mean=s_mean, compactness_mean=comp_mean, concavity_mean=conc_mean,
            concave_points_mean=cp_mean, symmetry_mean=sym_mean, fractal_dimension_mean=frac_mean,
            radius_se=r_se, texture_se=t_se, perimeter_se=p_se, area_se=a_se,
            smoothness_se=s_se, compactness_se=comp_se, concavity_se=conc_se,
            concave_points_se=cp_se, symmetry_se=sym_se, fractal_dimension_se=frac_se,
            radius_worst=r_worst, texture_worst=t_worst, perimeter_worst=p_worst, area_worst=a_worst,
            smoothness_worst=s_worst, compactness_worst=comp_worst, concavity_worst=conc_worst,
            concave_points_worst=cp_worst, symmetry_worst=sym_worst, fractal_dimension_worst=frac_worst
        )

        res = service.evaluate_patient(patient_obj)

        st.plotly_chart(plot_gauge(res.risk_score), use_container_width=True)

        if res.diagnosis == DiagnosisEnum.BENIGN:
            badge_cls = "badge-benign"
            box_bg = "rgba(16, 185, 129, 0.15)"
        elif res.diagnosis == DiagnosisEnum.EQUIVOCAL:
            badge_cls = "badge-equivocal"
            box_bg = "rgba(245, 158, 11, 0.15)"
        else:
            badge_cls = "badge-malignant"
            box_bg = "rgba(244, 63, 94, 0.20)"

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 12px;">
            <span class="{badge_cls}">VERDICT: {res.diagnosis.value.upper()}</span>
            <span style="margin-left: 12px; font-weight: 700; color: #38bdf8;">ACTION: {res.clinical_action.value}</span>
        </div>
        <div class="rec-box" style="background-color: {box_bg}; border: 1px solid rgba(255,255,255,0.1);">
            🩺 <b>Recommendation:</b> {res.recommendation}
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("P(Malignancy)", f"{res.malignancy_probability*100:.1f}%")
        with m2:
            st.metric("Inference Latency", f"{res.latency_ms:.2f} ms")

        st.markdown("#### 🕸️ Nuclear Morphometry Radar Profile")
        st.plotly_chart(plot_radar(res.radar_values), use_container_width=True)

        st.markdown("#### 🔍 Primary Biomarker Drivers (Z-Score Divergence from Benign Baseline):")
        for driver in res.top_biomarker_drivers:
            st.markdown(f"- **{driver.feature_name}**: `{driver.patient_value}` vs Benign Mean `{driver.benign_mean}` (**+{driver.z_score}σ**, {driver.percentile}th percentile) -> *{driver.impact}*")


elif mode == "📁 Cohort Batch CSV Screening":
    st.markdown("### 📁 High-Throughput Clinical Cohort Screening")
    st.markdown("Upload FNA lab cytology batch CSV to score entire patient cohorts with instant risk triage.")

    uploaded = st.file_uploader("Upload Cytology CSV File:", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} patient rows.")
        if st.button("🚀 Score Entire Cohort", use_container_width=True):
            patients_list = []
            for idx, r in batch_df.iterrows():
                d = r.to_dict()
                d["patient_id"] = str(d.get("id", f"PT-{idx+1001}"))
                patients_list.append(PatientFNAInput(**d))

            batch_res = service.evaluate_batch(BatchPatientInput(patients=patients_list))
            st.success(f"Cohort evaluated in {batch_res.avg_latency_ms:.2f} ms per patient.")
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Malignant Cases", batch_res.malignant_count)
            with c2: st.metric("Equivocal / Borderline Cases", batch_res.equivocal_count)
            with c3: st.metric("Benign Cases", batch_res.benign_count)

            res_table = []
            for r in batch_res.results:
                res_table.append({
                    "Patient ID": r.patient_id,
                    "Diagnosis": r.diagnosis.value,
                    "Malignancy Risk": f"{r.risk_score}%",
                    "Action Recommendation": r.clinical_action.value,
                    "Latency": f"{r.latency_ms:.2f} ms"
                })
            st.dataframe(pd.DataFrame(res_table), use_container_width=True)


elif mode == "📊 Model Validation & ROC-AUC":
    st.markdown("### 📊 Model Architecture & Clinical Validation")
    metrics = service.get_metrics()
    
    st.markdown("""
    #### 🧠 Clinical Ensemble Architecture
    Breast Cancer Sentinel 2.0 uses a calibrated soft-voting meta-classifier:
    
    - **Histogram Gradient Boosting**: Captures non-linear cellular boundary thresholds.
    - **Random Forest (150 Trees)**: Mitigates feature noise and high dimensionality.
    - **Deep Multi-Layer Perceptron (MLP)**: Continuous latent manifold representation.
    - **Regularized Logistic Regression**: Provides stable linear grounding and odds ratios.
    - **Platt Calibrated Classifier**: Maps ensemble margins to true empirical patient risk probabilities.
    """)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("ROC-AUC", f"{metrics.roc_auc*100:.2f}%")
    with c2: st.metric("Sensitivity (Recall)", f"{metrics.sensitivity_recall*100:.2f}%")
    with c3: st.metric("Specificity", f"{metrics.specificity*100:.2f}%")
    with c4: st.metric("Precision", f"{metrics.precision*100:.2f}%")

    st.markdown("#### 🎯 Clinical Confusion Matrix (Validation Cohort)")
    cm = np.array(metrics.confusion_matrix)
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted Diagnosis", y="True Diagnosis", color="Patients"),
        x=["Benign (0)", "Malignant (1)"],
        y=["Benign (0)", "Malignant (1)"],
        text_auto=True,
        color_continuous_scale="Viridis"
    )
    fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#f8fafc"))
    st.plotly_chart(fig_cm, use_container_width=True)
