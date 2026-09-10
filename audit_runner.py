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
from typing import Dict, Any
import cv2
import yaml
import numpy as np
import pandas as pd

from engines.model_interface import ClinicalModelWrapper
from engines.robustness_engine import RobustnessEngine
from engines.fairness_engine import FairnessEngine
from engines.safety_engine import SafetyEngine
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
) -> Dict[str, Any]:
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)

    print(f"==================================================================")
    print(f"  🛡️  TrustCheck: Clinical AI Pre-Deployment Audit Battery")
    print(f"==================================================================")
    print(f"Target Model:   {model_target}")
    print(f"Cohort Dataset: {metadata_path}")
    print(f"Output Path:    {output_dir}")

    # 1. Ingestion & Registration Layer
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cohort metadata CSV not found: {metadata_path}")

    meta_df = pd.read_csv(metadata_path)
    print(f"Loaded validation cohort: {len(meta_df)} patients across {meta_df['site_id'].nunique()} clinical site(s).")

    images = []
    labels = []
    patient_ids = []
    
    for _, row in meta_df.iterrows():
        img_p = row["image_path"]
        if not os.path.isabs(img_p):
            # Look relative to current working directory or metadata directory
            if not os.path.exists(img_p):
                alt_p = os.path.join(os.path.dirname(metadata_path), "..", img_p)
                if os.path.exists(alt_p):
                    img_p = alt_p
        img = cv2.imread(img_p)
        if img is None:
            # Generate simulated scan if file not found
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        images.append(img)
        labels.append(int(row["ground_truth"]))
        patient_ids.append(str(row["patient_id"]))

    labels_arr = np.array(labels)

    # 2. Target Model Ingestion
    wrapper = ClinicalModelWrapper(model_target)
    print("Running baseline inference & feature extraction...")
    clean_confs, clean_preds, embeddings = wrapper.infer_batch(images, patient_ids)

    # Callable closure for image batch inference
    def predict_batch_fn(batch_imgs):
        c, p, _ = wrapper.infer_batch(batch_imgs)
        return c, p

    # 3. STEP 1: ENGINE A - Physics & Hardware Shift (Robustness)
    print("Executing Step 1: Engine A (Physics corruptions & decay curves)...")
    rob_engine = RobustnessEngine(intensity_tiers=cfg.get("engine_a_robustness", {}).get("intensity_tiers"))
    robustness_results = rob_engine.evaluate_cohort(images, labels, predict_batch_fn)

    # 4. STEP 2: ENGINE B - Subgroup Bias & Covariate Shift Auditor
    print("Executing Step 2: Engine B (Subgroup fairness slicing & site drift)...")
    fair_engine = FairnessEngine(
        alarm_classifier_auc=cfg.get("engine_b_fairness", {}).get("adversarial_classifier_alarm_auc", 0.65)
    )
    fairness_results = fair_engine.slice_cohort(meta_df, clean_preds, clean_confs, labels_arr)
    drift_results = fair_engine.detect_latent_distribution_drift(embeddings, meta_df["site_id"].tolist())

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

    # 6. Deterministic Composite Scorer & Gating Logic
    weights = cfg.get("scoring_weights", {"robustness": 0.35, "fairness": 0.30, "calibration": 0.20, "shift": 0.15})
    trust_score = safety_engine.compute_composite_trust_score(
        robustness_score=robustness_results["robustness_score"],
        fairness_score=fairness_and_shift["fairness_score"],
        calibration_score=cal_results["calibration_score"],
        shift_score=fairness_and_shift["shift_score"],
        weights=weights,
    )

    contrast_decay_fail = (
        len(robustness_results["decay_slopes"]["contrast_attenuation"]) >= 3
        and robustness_results["decay_slopes"]["contrast_attenuation"][2] < 0.70
    )

    verdict, contraindications = safety_engine.evaluate_gating(
        trust_score=trust_score,
        max_subgroup_disparity=fairness_and_shift["max_disparity"],
        critical_silent_failures_count=cal_results["critical_silent_failures_count"],
        total_samples=len(images),
        contrast_decay_failure=contrast_decay_fail,
    )

    # 7. Standardized Telemetry Compilation
    run_id = f"tc_run_{int(time.time())}_{wrapper.model_name.lower().replace(' ', '_')[:16]}"
    telemetry = {
        "audit_run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_model": wrapper.model_name,
        "metrics": {
            "robustness": robustness_results,
            "fairness_and_shift": fairness_and_shift,
            "calibration_and_uncertainty": cal_results,
        },
        "trust_score": trust_score,
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

    args = parser.parse_args()
    try:
        run_evaluation_suite(
            model_target=args.model,
            metadata_path=args.metadata,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    except Exception as e:
        print(f"Error during audit run: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
