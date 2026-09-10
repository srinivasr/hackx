"""Unit tests validating TrustCheck engines and HLT-08 compliance assertions."""

import os
import pytest
import numpy as np
import pandas as pd

from engines.robustness_engine import (
    apply_contrast_attenuation,
    apply_poisson_noise,
    apply_motion_blur,
    inject_anatomical_distractor,
    inject_hospital_text_stamp,
    inject_metallic_hardware_artifact,
    inject_collimator_aperture_border,
    RobustnessEngine,
)
from engines.fairness_engine import (
    FairnessEngine,
    compute_rbf_mmd,
    bootstrap_disparity_ci,
)
from engines.safety_engine import (
    compute_ece,
    compute_adaptive_calibration_error,
    fit_temperature_scaling,
    compute_shannon_entropy,
    compute_free_energy,
    detect_mahalanobis_ood,
    SafetyEngine,
)
from audit_runner import run_evaluation_suite


@pytest.fixture
def dummy_scan():
    # 224x224 grayscale radiograph simulation
    img = np.full((224, 224), 120, dtype=np.uint8)
    img[50:150, 50:150] = 60
    return img


@pytest.fixture
def dummy_cohort_df():
    return pd.DataFrame({
        "patient_id": [f"P_{i:03d}" for i in range(20)],
        "age": [8, 12, 25, 34, 45, 52, 68, 72, 15, 30, 42, 55, 60, 70, 80, 22, 33, 44, 16, 50],
        "sex": ["M", "F"] * 10,
        "site_id": ["AIIMS_DELHI"] * 10 + ["DIST_HOSP_JAIPUR"] * 10,
        "scanner_type": ["DIGITAL_RAD"] * 10 + ["COMPUTED_RAD"] * 10,
        "ground_truth": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })


def test_robustness_perturbations(dummy_scan):
    # 1. Contrast Attenuation
    attenuated = apply_contrast_attenuation(dummy_scan, severity=5)
    assert attenuated.shape == dummy_scan.shape
    assert attenuated.dtype == np.uint8
    assert np.std(attenuated) < np.std(dummy_scan)

    # 2. Poisson Noise (Deterministic reproducible)
    noisy1 = apply_poisson_noise(dummy_scan, severity=3, seed=100)
    noisy2 = apply_poisson_noise(dummy_scan, severity=3, seed=100)
    assert noisy1.shape == dummy_scan.shape
    assert np.array_equal(noisy1, noisy2)

    # 3. Motion Blur
    blurred = apply_motion_blur(dummy_scan, severity=3)
    assert blurred.shape == dummy_scan.shape


def test_multi_archetype_distractors(dummy_scan):
    # Test all 4 non-clinical shortcut archetypes (DeGrave et al. Nature MI 2021)
    d_marker = inject_anatomical_distractor(dummy_scan, marker="L")
    assert d_marker.shape == dummy_scan.shape
    assert np.max(d_marker[:30, -30:]) >= 200

    d_text = inject_hospital_text_stamp(dummy_scan, text="PORTABLE CHEST 08:30")
    assert d_text.shape == dummy_scan.shape
    assert np.mean(d_text[-14:, :]) < np.mean(dummy_scan)

    d_metal = inject_metallic_hardware_artifact(dummy_scan)
    assert d_metal.shape == dummy_scan.shape
    assert np.max(d_metal) >= 240

    d_border = inject_collimator_aperture_border(dummy_scan, border_fraction=0.10)
    assert d_border.shape == dummy_scan.shape
    assert np.sum(d_border[:10, :]) == 0


def test_fairness_subgroup_disparity(dummy_cohort_df):
    engine = FairnessEngine()
    # Simulate predictions with pediatric under-diagnosis (FNR spike on <18)
    preds = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    confs = np.array([0.4, 0.1, 0.8, 0.2, 0.9, 0.1, 0.7, 0.2, 0.3, 0.1, 0.85, 0.15, 0.9, 0.2, 0.75, 0.1, 0.88, 0.22, 0.35, 0.12])
    labels = dummy_cohort_df["ground_truth"].values

    report = engine.slice_cohort(dummy_cohort_df, preds, confs, labels)
    assert "slices" in report
    assert "fairness_score" in report
    assert "underdiagnosis_ratio_pediatric_vs_adult" in report
    assert report["underdiagnosis_ratio_pediatric_vs_adult"] > 1.0
    assert "pediatric_disparity_ci" in report
    assert "intersectional_disparities" in report
    assert "equalized_odds_difference" in report


def test_bootstrap_disparity_ci(dummy_cohort_df):
    dummy_cohort_df["pred"] = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
    dummy_cohort_df["age_bracket"] = pd.cut(dummy_cohort_df["age"], bins=[-1, 17, 65, 120], labels=["<18", "18-65", ">65"])
    point, low, high = bootstrap_disparity_ci(dummy_cohort_df, "age_bracket", "<18", "18-65", n_bootstraps=100)
    assert point > 0.0
    assert low <= high


def test_latent_drift_and_mmd():
    engine = FairnessEngine()
    np.random.seed(42)
    # 20 samples from Site A, 20 samples from Site B with slight mean shift
    emb_a = np.random.normal(0.0, 1.0, size=(20, 64))
    emb_b = np.random.normal(1.2, 1.0, size=(20, 64))
    embeddings = np.vstack([emb_a, emb_b])
    site_labels = ["AIIMS_DELHI"] * 20 + ["DIST_HOSP_JAIPUR"] * 20

    drift_report = engine.detect_latent_distribution_drift(embeddings, site_labels)
    assert "domain_shift_classifier_auroc" in drift_report
    assert "ks_drift_p_value" in drift_report
    assert "latent_mmd_distance" in drift_report
    assert drift_report["latent_mmd_distance"] > 0.0
    assert drift_report["domain_shift_classifier_auroc"] > 0.65
    assert drift_report["site_divergence_detected"] is True


def test_calibration_and_silent_failures():
    engine = SafetyEngine(n_bins=10)
    
    # 10 samples, with 1 critical silent failure: conf=0.92, label=1, pred=0
    confs = np.array([0.92, 0.88, 0.75, 0.12, 0.05, 0.80, 0.65, 0.20, 0.10, 0.95])
    preds = np.array([0,    1,    1,    0,    0,    1,    1,    0,    0,    1])
    labels = np.array([1,   1,    1,    0,    0,    1,    0,    0,    0,    1])
    probs = np.stack([1 - confs, confs], axis=1)

    cal_res = engine.evaluate(probs, preds, labels)
    assert "expected_calibration_error" in cal_res
    assert "adaptive_calibration_error" in cal_res
    assert "temperature_scaling_diagnostic" in cal_res
    assert 0.0 <= cal_res["expected_calibration_error"] <= 1.0
    assert cal_res["critical_silent_failures_count"] == 1
    assert cal_res["critical_silent_failures"][0]["confidence"] == 0.92

    # Shannon Entropy and Free Energy
    entropies = compute_shannon_entropy(probs)
    assert len(entropies) == 10
    assert np.all(entropies >= 0.0)

    energy = compute_free_energy(confs)
    assert len(energy) == 10


def test_adaptive_calibration_and_temperature_scaling():
    np.random.seed(42)
    # Simulated overconfident model
    logits = np.random.normal(0, 2.0, size=50)
    probs = 1.0 / (1.0 + np.exp(-logits * 3.0))  # Exaggerated overconfidence
    labels = (np.random.uniform(0, 1, size=50) < probs).astype(int)
    preds = (probs >= 0.50).astype(int)

    ace = compute_adaptive_calibration_error(probs, preds, labels, n_bins=5)
    assert 0.0 <= ace <= 1.0

    t_best, recal_ece = fit_temperature_scaling(probs, labels)
    assert t_best > 0.0
    assert 0.0 <= recal_ece <= 1.0


def test_composite_trust_score_math():
    score = SafetyEngine.compute_composite_trust_score(
        robustness_score=80.0,
        fairness_score=70.0,
        calibration_score=90.0,
        shift_score=60.0,
    )
    # 0.35*80 + 0.30*70 + 0.20*90 + 0.15*60 = 28 + 21 + 18 + 9 = 76.0
    assert score == 76.0

    verdict, contraindications = SafetyEngine.evaluate_gating(
        trust_score=76.0,
        max_subgroup_disparity=1.5,
        critical_silent_failures_count=0,
        total_samples=50,
    )
    assert verdict == "CAUTION_RESTRICTED_DEPLOYMENT"


def test_end_to_end_audit_runner(tmp_path):
    metadata_csv = "data/sample_metadata.csv"
    if not os.path.exists(metadata_csv):
        pytest.skip("data/sample_metadata.csv not found.")

    out_dir = str(tmp_path / "audit_test_out")
    telemetry = run_evaluation_suite(
        model_target="benchmark:chexnet-densenet121",
        metadata_path=metadata_csv,
        output_dir=out_dir,
    )

    assert "audit_run_id" in telemetry
    assert "trust_score" in telemetry
    assert "verdict" in telemetry
    assert "metrics" in telemetry
    assert "shortcut_diagnostics" in telemetry["metrics"]["robustness"]
    assert "adaptive_calibration_error" in telemetry["metrics"]["calibration_and_uncertainty"]
    assert "temperature_scaling_diagnostic" in telemetry["metrics"]["calibration_and_uncertainty"]
    assert os.path.exists(os.path.join(out_dir, "telemetry.json"))
    assert os.path.exists(os.path.join(out_dir, "audit_certificate.pdf"))
    assert os.path.getsize(os.path.join(out_dir, "audit_certificate.pdf")) > 1000
