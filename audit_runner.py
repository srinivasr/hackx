#!/usr/bin/env python3
"""TrustCheck: Independent Pre-Deployment Stress-Testing & Safety Harness for Clinical AI.
HLT-08 Compliance CLI Orchestrator.

Usage:
  python audit_runner.py --model benchmark:chexnet-densenet121 --metadata data/sample_metadata.csv --output-dir output/audit_run_01
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional
import cv2
import yaml
import numpy as np
import pandas as pd

from engines.model_interface import ClinicalModelWrapper
from engines.robustness_engine import RobustnessEngine
from engines.fairness_engine import FairnessEngine
from engines.safety_engine import SafetyEngine
from engines.clinical_utility import (
    compute_decision_curve_analysis,
    compute_prevalence_shift_ladder,
    find_clinical_operating_points,
)
from engines.gaudit_engine import (
    run_gaudit_analysis,
    compute_selective_suppression_policy,
    compute_clinical_mce,
    compute_worst_group_metrics,
)
from reporting.pdf_generator import generate_dossier_pdf


def load_config(config_path: str = "config/audit_thresholds.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def run_evaluation_suite(
    model_target: str,
    metadata_path: str,
    output_dir: str = "output/audit_run_01",
    config_path: str = "config/audit_thresholds.yaml",
    modality: str = "chest_xray",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)

    print(f"==================================================================")
    print(f"  🛡️  TrustCheck: Clinical AI Pre-Deployment Audit Battery")
    print(f"==================================================================")
    print(f"Target Model:   {model_name or model_target}")
    print(f"Cohort Dataset: {metadata_path}")
    print(f"Modality Suite: {modality.upper()}")
    print(f"Output Path:    {output_dir}")

    # 1. Ingestion & Registration Layer
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cohort metadata CSV not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    print(f"Loaded validation cohort: {len(meta_df)} patients across {meta_df['site_id'].nunique()} clinical site(s).")

    samples = []
    labels = []
    patient_ids = []
    
    is_text_cohort = "clinical_note" in meta_df.columns or "text" in meta_df.columns or modality in ["clinical_nlp", "clinical_text", "nlp", "text"]

    for _, row in meta_df.iterrows():
        if is_text_cohort:
            text_val = str(row.get("clinical_note", row.get("text", "")))
            samples.append(text_val)
        else:
            img_p = row.get("image_path", "")
            if not os.path.isabs(img_p):
                if not os.path.exists(img_p):
                    alt_p = os.path.join(os.path.dirname(metadata_path), "..", img_p)
                    if os.path.exists(alt_p):
                        img_p = alt_p
            img = cv2.imread(img_p)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            samples.append(img)

        labels.append(int(row["ground_truth"]))
        patient_ids.append(str(row["patient_id"]))

    labels_arr = np.array(labels)

    # 2. Target Model Ingestion
    wrapper = ClinicalModelWrapper(model_target, model_name=model_name)
    print("Running baseline inference & feature extraction...")
    clean_confs, clean_preds, embeddings = wrapper.infer_batch(samples, patient_ids)

    # Callable closure for batch inference
    def predict_batch_fn(batch_samples):
        c, p, _ = wrapper.infer_batch(batch_samples)
        return c, p

    # 3. STEP 1: ENGINE A - Physics & Hardware Shift (Robustness)
    print(f"Executing Step 1: Engine A (Corruptions for {modality})...")
    rob_engine = RobustnessEngine(
        intensity_tiers=cfg.get("engine_a_robustness", {}).get("intensity_tiers"),
        modality=modality,
    )
    robustness_results = rob_engine.evaluate_cohort(samples, labels, predict_batch_fn)

    # 4. STEP 2: ENGINE B - Subgroup Bias & Covariate Shift Auditor
    print(f"Executing Step 2: Engine B (Subgroup fairness & shift, profile={wrapper.tier_profile})...")
    fair_engine = FairnessEngine(
        alarm_classifier_auc=cfg.get("engine_b_fairness", {}).get("adversarial_classifier_alarm_auc", 0.65)
    )
    fairness_results = fair_engine.slice_cohort(meta_df, clean_preds, clean_confs, labels_arr)
    drift_results = fair_engine.detect_latent_distribution_drift(
        embeddings, meta_df["site_id"].tolist(), tier_profile=wrapper.tier_profile
    )

    fairness_and_shift = {
        **fairness_results,
        **drift_results,
    }

    # 5. STEP 3: ENGINE C - Uncertainty, Calibration & Silent Failure Probing
    print("Executing Step 3: Engine C (Uncertainty, ECE calibration & silent failure probing)...")
    safety_engine = SafetyEngine(
        n_bins=cfg.get("engine_c_calibration", {}).get("n_bins", 10),
        high_conf_threshold=cfg.get("engine_c_calibration", {}).get("high_confidence_threshold", 0.85),
        quarantine_conf_threshold=cfg.get("engine_c_calibration", {}).get("quarantine_confidence_threshold", 0.90),
    )
    cal_results = safety_engine.evaluate(
        clean_confs, clean_preds, labels_arr, embeddings=embeddings, patient_ids=patient_ids
    )

    # 6. Clinical Utility, Decision Curve Analysis (DCA) & Prevalence Shift Simulator
    print("Executing Decision Curve Analysis (DCA) & Prevalence Shift Stress Test...")
    dca_results = compute_decision_curve_analysis(labels_arr, clean_confs)
    operating_points = find_clinical_operating_points(labels_arr, clean_confs)
    prevalence_simulation = compute_prevalence_shift_ladder(
        sensitivity=operating_points.get("screening_triage_sensitivity", 0.90),
        specificity=operating_points.get("screening_triage_specificity", 0.82),
    )
    demographic_leakage = fair_engine.audit_demographic_latent_leakage(
        embeddings, meta_df["sex"].tolist(), tier_profile=wrapper.tier_profile
    )

    # 7. G-AUDIT, Selective Triage Suppression & Worst-Group Benchmarks
    print("Executing G-AUDIT (FDA/JHU 2025) & Selective Suppression Policy (JAMIA 2023)...")
    gaudit_report = run_gaudit_analysis(embeddings, meta_df, labels_arr)
    suppression_policy = compute_selective_suppression_policy(clean_confs, clean_preds, labels_arr)
    cmce_report = compute_clinical_mce(robustness_results["decay_slopes"])
    worst_group = compute_worst_group_metrics(fairness_results["slices"])

    # 8. Deterministic Composite Scorer & Gating Logic
    weights = cfg.get("scoring_weights", {"robustness": 0.35, "fairness": 0.30, "calibration": 0.20, "shift": 0.15})
    trust_score = safety_engine.compute_composite_trust_score(
        robustness_score=robustness_results["robustness_score"],
        fairness_score=fairness_and_shift["fairness_score"],
        calibration_score=cal_results["calibration_score"],
        shift_score=fairness_and_shift["shift_score"],
        weights=weights,
    )

    contrast_decay_fail = robustness_results.get("contrast_decay_failure", False)

    verdict, contraindications = safety_engine.evaluate_gating(
        trust_score=trust_score,
        max_subgroup_disparity=fairness_and_shift["max_disparity"],
        critical_silent_failures_count=cal_results["critical_silent_failures_count"],
        total_samples=len(samples),
        contrast_decay_failure=contrast_decay_fail,
    )

    if prevalence_simulation.get("collapse_detected", False):
        contraindications.append(
            f"CAUTION: High alert fatigue risk in outpatient triage. Positive Predictive Value collapses below 50% when disease prevalence drops < {prevalence_simulation.get('prevalence_collapse_point', 0.05)*100:.0f}%."
        )

    if demographic_leakage.get("leakage_detected", False):
        contraindications.append(
            f"ETHICS RISK: Demographic identity leakage detected in latent representations (Sex decoding AUROC: {demographic_leakage.get('leakage_auroc', 0.5):.2f} >= 0.75). Model may exploit demographic prevalence shortcuts."
        )

    if gaudit_report.get("has_shortcut_hazard", False):
        for sc in gaudit_report.get("high_risk_shortcuts", []):
            contraindications.append(
                f"G-AUDIT SHORTCUT HAZARD: Attribute '{sc['attribute']}' is both highly detectable (AUC {sc['detectability_auc']:.2f}) and correlated with diagnosis (AUC {sc['utility_auc']:.2f}). High spurious shortcut risk."
            )

    # 9. Standardized Telemetry Compilation
    run_id = f"tc_run_{int(time.time())}_{wrapper.model_name.lower().replace(' ', '_')[:16]}"
    telemetry = {
        "audit_run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": wrapper.model_name,
        "modality": modality,
        "tier_profile": wrapper.tier_profile,
        "metrics": {
            "robustness": robustness_results,
            "fairness_and_shift": fairness_and_shift,
            "calibration_and_uncertainty": cal_results,
            "clinical_utility_dca": dca_results,
            "clinical_operating_points": operating_points,
            "prevalence_shift_simulation": prevalence_simulation,
            "demographic_latent_leakage": demographic_leakage,
            "gaudit_shortcut_risk": gaudit_report,
            "selective_suppression_policy": suppression_policy,
            "clinical_mce": cmce_report,
            "worst_group_benchmarks": worst_group,
        },
        "trust_score": trust_score,
        "composite_trust_score": trust_score,
        "verdict": verdict,
        "clinical_contraindications": contraindications,
    }

    # Save telemetry JSON
    telemetry_path = os.path.join(output_dir, "telemetry.json")
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved standardized telemetry JSON -> {telemetry_path}")

    # 8. Compile Exportable Regulatory PDF Dossier
    pdf_path = os.path.join(output_dir, "audit_certificate.pdf")
    generate_dossier_pdf(telemetry, pdf_path)
    print(f"Compiled FDA/CDSCO SaMD Safety Dossier -> {pdf_path}")

    # Print Summary Terminal Telemetry
    elapsed = round(time.time() - start_time, 2)
    print(f"------------------------------------------------------------------")
    print(f"AUDIT VERDICT:     {verdict}")
    print(f"COMPOSITE SCORE:   {trust_score} / 100")
    print(f"  • Robustness:    {robustness_results['robustness_score']} / 100 (Clean AUROC: {robustness_results['clean_auroc']})")
    print(f"  • Fairness:      {fairness_and_shift['fairness_score']} / 100 (Max Disparity: {fairness_and_shift['max_disparity']}x)")
    print(f"  • Calibration:   {cal_results['calibration_score']} / 100 (ECE: {cal_results['ece_percent']}%)")
    print(f"  • Shift:         {fairness_and_shift['shift_score']} / 100 (Domain AUC: {fairness_and_shift['domain_shift_classifier_auroc']})")
    print(f"Silent Failures:   {cal_results['critical_silent_failures_count']} high-confidence false negative(s)")
    if contraindications:
        print(f"Contraindications:")
        for c in contraindications:
            print(f"  [!] {c}")
    print(f"Audit completed in {elapsed}s.")
    print(f"==================================================================")

    return telemetry


def main():
    parser = argparse.ArgumentParser(description="TrustCheck Clinical AI Stress-Testing & Safety Harness")
    parser.add_argument("--model", type=str, default="benchmark:chexnet-densenet121", help="Target model (.onnx, .pt, benchmark, or REST URL)")
    parser.add_argument("--metadata", type=str, default="data/sample_metadata.csv", help="Cohort metadata CSV path")
    parser.add_argument("--output-dir", type=str, default="output/audit_run_01", help="Directory for telemetry JSON and PDF dossier")
    parser.add_argument("--config", type=str, default="config/audit_thresholds.yaml", help="Audit thresholds configuration YAML")
    parser.add_argument("--modality", type=str, default="chest_xray", choices=["chest_xray", "retinal_fundus", "tabular_ehr", "clinical_nlp", "clinical_text", "dermatology_dermoscopy", "dermatology", "digital_pathology", "histopathology"], help="Clinical Modality Perturbation Suite")

    args = parser.parse_args()
    try:
        run_evaluation_suite(
            model_target=args.model,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            config_path=args.config,
            modality=args.modality,
        )
    except Exception as e:
        print(f"Error during audit run: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
