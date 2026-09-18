import os
import pytest
import numpy as np
import cv2

from engine.perturbation import (
    apply_gaussian_blur,
    apply_illumination_attenuation,
    apply_corneal_glare,
    apply_resolution_scaling,
    apply_adversarial_noise,
    apply_synthetic_artifact,
    apply_spatial_occlusion,
)
from engine.calibration import compute_ece, compute_brier_score
from engine.sanity_check import verify_lesion_classification_consensus, compute_subgroup_fairness
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

def test_adversarial_and_artifact_perturbations(dummy_image):
    adv = apply_adversarial_noise(dummy_image, epsilon=0.05)
    assert adv.shape == dummy_image.shape
    assert adv.dtype == np.uint8
    
    artifact = apply_synthetic_artifact(dummy_image, intensity=0.5)
    assert artifact.shape == dummy_image.shape
    assert artifact.dtype == np.uint8

def test_spatial_occlusion(dummy_image):
    occluded = apply_spatial_occlusion(dummy_image, occlusion_fraction=0.3)
    assert occluded.shape == dummy_image.shape
    assert occluded.dtype == np.uint8
    # Ensure there is a significant amount of black pixels (0)
    assert np.mean(occluded) < np.mean(dummy_image)
    black_pixels = np.sum(np.all(occluded == [0, 0, 0], axis=-1))
    total_pixels = dummy_image.shape[0] * dummy_image.shape[1]
    # Check that at least 30% of pixels are black (some overlap might happen, so check > 20%)
    assert black_pixels / total_pixels > 0.2


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


def test_subgroup_fairness_computation():
    preds = [1, 1, 0, 0, 1, 0, 1, 0]
    gts = [1, 0, 0, 1, 1, 0, 0, 0]
    demographics = ["Alpha", "Beta", "Alpha", "Beta", "Beta", "Alpha", "Alpha", "Beta"]
    
    res = compute_subgroup_fairness(preds, gts, demographics)
    assert "average_fpr_disparity" in res
    assert "average_fnr_disparity" in res
    assert "passes_equalized_odds" in res
    assert "subgroups" in res
    assert "Alpha" in res["subgroups"]
    assert "Beta" in res["subgroups"]


def test_end_to_end_audit_and_pdf(dummy_image, tmp_path):
    model_path = "assets/models/retinal_dr/dr_retinal_lcnet_edge.onnx"
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
