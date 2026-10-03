"""
Unit tests for pipeline, preprocessor, and model inference.
"""

from cancer_ai.pipeline.preprocessor import CancerDataPreprocessor
from cancer_ai.service import service
from cancer_ai.schemas import DiagnosisEnum, ClinicalAction


def test_preprocessor_shape(sample_benign_patient):
    prep = CancerDataPreprocessor()
    vec = prep.transform(sample_benign_patient.model_dump(by_alias=True))
    assert vec.shape == (1, 30)


def test_benign_evaluation(sample_benign_patient):
    result = service.evaluate_patient(sample_benign_patient)
    assert result.diagnosis == DiagnosisEnum.BENIGN
    assert result.malignancy_probability < 0.25
    assert result.clinical_action == ClinicalAction.ROUTINE_SCREENING
    assert len(result.top_biomarker_drivers) > 0


def test_malignant_evaluation(sample_malignant_patient):
    result = service.evaluate_patient(sample_malignant_patient)
    assert result.diagnosis == DiagnosisEnum.MALIGNANT
    assert result.malignancy_probability > 0.70
    assert result.clinical_action == ClinicalAction.URGENT_ONCOLOGY_REFERRAL
