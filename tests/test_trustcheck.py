import os
import pytest
import numpy as np
import cv2

from engine.perturbation import (
    apply_gaussian_blur,
    apply_illumination_attenuation,
    apply_corneal_glare,
    apply_resolution_scaling,
)
from engine.calibration import compute_ece, compute_brier_score
from engine.sanity_check import verify_lesion_classification_consensus
from engine.auditor import CandidateModelEvaluator, run_full_model_audit
from backend.certificate_gen import generate_deployment_certificate


@pytest.fixture
def dummy_image():
    # 384x384 BGR image
    return np.full((384, 384, 3), 128, dtype=np.uint8)


def test_perturbation_shapes(dummy_image):
    blurred = apply_gaussian_blur(dummy_image, 2.5)
    assert blurred.shape == dummy_image.shape
    assert blurred.dtype == np.uint8

    dimmed = apply_illumination_attenuation(dummy_image, 0.5)
    assert dimmed.shape == dummy_image.shape
    assert np.mean(dimmed) < np.mean(dummy_image)

    glared = apply_corneal_glare(dummy_image, 0.6)
    assert glared.shape == dummy_image.shape
    assert np.mean(glared) > np.mean(dummy_image)

    scaled = apply_resolution_scaling(dummy_image, 160)
    assert scaled.shape == dummy_image.shape


def test_calibration_computation():
    probs = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0, 0.0],
        [0.7, 0.1, 0.1, 0.1, 0.0],
        [0.6, 0.4, 0.0, 0.0, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])
    res = compute_ece(probs, labels, n_bins=4)

    assert "ece" in res
    assert 0.0 <= res["ece"] <= 1.0
    assert len(res["reliability_bins"]) == 4

    brier = compute_brier_score(probs, labels)
    assert 0.0 <= brier <= 2.0


def test_silent_failure_detection():
    # Model predicts Grade 0 (Normal) with high confidence, but 12 MAs exist
    biomarkers = {
        "microaneurysms": 12,
        "exudate_area_pct": 0.25,
        "hemorrhage_quadrants": 2,
    }
    check = verify_lesion_classification_consensus(
        predicted_grade=0, confidence=92.5, biomarkers=biomarkers
    )
    assert not check["is_safe"]
    assert check["violation_type"] == "CRITICAL_SILENT_FALSE_NEGATIVE"


def test_end_to_end_audit_and_pdf(dummy_image, tmp_path):
    model_path = "assets/models/dr_retinal_lcnet_edge.onnx"
    if not os.path.exists(model_path):
        pytest.skip("Model not found in assets.")

    evaluator = CandidateModelEvaluator(model_path, "Test-Candidate-Model")
    samples = {"sample_test": dummy_image}

    audit = run_full_model_audit(evaluator, samples)
    assert "readiness_score" in audit
    assert "verdict" in audit
    assert "calibration" in audit
    assert "stress_tests" in audit

    pdf_out = str(tmp_path / "test_cert.pdf")
    cert_path = generate_deployment_certificate(audit, pdf_out)
    assert os.path.exists(cert_path)
    assert os.path.getsize(cert_path) > 1000
