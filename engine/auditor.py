import os
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import onnxruntime as ort

from engine.perturbation import (
    apply_gaussian_blur,
    apply_illumination_attenuation,
    apply_corneal_glare,
    apply_resolution_scaling,
)
from engine.calibration import compute_ece, compute_brier_score
from engine.sanity_check import verify_lesion_classification_consensus


class CandidateModelEvaluator:
    """Independent evaluation runner for healthcare AI models."""

    def __init__(self, model_path: str, model_name: str = "Candidate-Model-A"):
        self.model_path = model_path
        self.model_name = model_name
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        
        try:
            self.session = ort.InferenceSession(model_path, providers=self.providers)
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            if len(inp.shape) >= 4 and isinstance(inp.shape[-2], int) and isinstance(inp.shape[-1], int):
                self.target_size = (inp.shape[-1], inp.shape[-2])
            else:
                self.target_size = (384, 384)
            self.has_masks = len(self.session.get_outputs()) > 1
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX model at {model_path}: {e}")

    def preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """Preprocesses raw BGR fundus or X-ray scan into normalized float32 tensor."""
        resized = cv2.resize(img_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        elif resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = resized.astype(np.float32) / 255.0

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        norm = (rgb - mean) / std

        # (H, W, C) -> (1, C, H, W)
        tensor = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def infer(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Runs inference and returns softmax probabilities and segmented lesion counts."""
        tensor_in = self.preprocess_image(img_bgr)
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor_in})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        logits = outputs[0][0]
        # Numerical stability softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100.0

        biomarkers = {"microaneurysms": 0, "exudate_area_pct": 0.0, "hemorrhage_quadrants": 0, "soft_exudate_area_pct": 0.0}
        if self.has_masks and len(outputs) > 1:
            raw_masks = outputs[1][0]
            # Sigmoid activation on 4-channel lesion masks
            mask_probs = 1.0 / (1.0 + np.exp(-np.clip(raw_masks, -15.0, 15.0)))
            
            h, w = raw_masks.shape[1], raw_masks.shape[2]

            # Channel 0: Microaneurysms
            ma_bin = (mask_probs[0] >= 0.50).astype(np.uint8)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ma_bin, connectivity=8)
            mas = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 3)
            
            # Channel 1: Hard Exudates
            ex_bin = (mask_probs[1] >= 0.50).astype(np.uint8)
            ex_pct = float(np.sum(ex_bin) / (h * w)) * 100.0

            # Channel 2: Hemorrhages
            he_bin = (mask_probs[2] >= 0.50).astype(np.uint8)
            quads = [he_bin[0:h//2, 0:w//2], he_bin[0:h//2, w//2:w], he_bin[h//2:h, 0:w//2], he_bin[h//2:h, w//2:w]]
            he_quads = sum(1 for q in quads if np.sum(q) > 10)

            # Channel 3: Soft Exudates
            se_bin = (mask_probs[3] >= 0.50).astype(np.uint8)
            se_pct = float(np.sum(se_bin) / (h * w)) * 100.0

            biomarkers = {
                "microaneurysms": mas,
                "exudate_area_pct": round(ex_pct, 2),
                "hemorrhage_quadrants": he_quads,
                "soft_exudate_area_pct": round(se_pct, 2),
            }

        return {
            "predicted_grade": pred_class,
            "confidence": round(confidence, 1),
            "probs": probs.tolist(),
            "latency_ms": round(latency_ms, 2),
            "biomarkers": biomarkers,
        }


def run_full_model_audit(
    evaluator: CandidateModelEvaluator,
    sample_images: Dict[str, np.ndarray],
    ground_truth_labels: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Runs end-to-end clinical stress and safety audit on a candidate healthcare model."""
    start_time = time.time()
    
    # 1. Baseline In-Domain Inference
    baseline_predictions = {}
    all_probs = []
    all_labels = []

    for name, img in sample_images.items():
        res = evaluator.infer(img)
        baseline_predictions[name] = res
        all_probs.append(res["probs"])
        true_label = ground_truth_labels.get(name, res["predicted_grade"]) if ground_truth_labels else res["predicted_grade"]
        all_labels.append(true_label)

    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    # 2. Calibration Analysis
    calibration_metrics = compute_ece(all_probs_arr, all_labels_arr, n_bins=5)
    brier = compute_brier_score(all_probs_arr, all_labels_arr)
    calibration_metrics["brier_score"] = round(brier, 4)

    # 3. Optical Stress Ladders
    # Use a primary reference sample for progressive degradation curves
    ref_key = "sample_clinical_pass" if "sample_clinical_pass" in sample_images else list(sample_images.keys())[0]
    ref_img = sample_images[ref_key]
    base_ref = baseline_predictions[ref_key]

    def compute_retention_stability(base_res: Dict[str, Any], p_res: Dict[str, Any], severity_pct: float) -> float:
        """Calculates model-specific retained empirical stability under perturbation."""
        base_p = np.array(base_res.get("probs", []), dtype=np.float32)
        p_p = np.array(p_res.get("probs", []), dtype=np.float32)
        
        if len(base_p) > 0 and len(base_p) == len(p_p):
            tvd = 0.5 * float(np.sum(np.abs(base_p - p_p)))
            fidelity = max(0.0, 1.0 - tvd)
        else:
            fidelity = 0.5

        if p_res.get("predicted_grade") == base_res.get("predicted_grade"):
            class_cons = 1.0
        else:
            diff = abs(int(p_res.get("predicted_grade", 0)) - int(base_res.get("predicted_grade", 0)))
            class_cons = max(0.15, 1.0 - 0.25 * diff)

        base_conf = max(1.0, float(base_res.get("confidence", 50.0)))
        conf_ratio = min(1.0, float(p_res.get("confidence", 50.0)) / base_conf)

        raw_resilience = 0.50 * fidelity + 0.35 * class_cons + 0.15 * conf_ratio
        attenuation = 1.0 - (severity_pct / 100.0) * (1.1 - 0.5 * raw_resilience)
        stability = max(2.0, min(100.0, 100.0 * raw_resilience * max(0.05, attenuation)))
        return round(float(stability), 1)

    stress_tests = {
        "blur_ladder": [],
        "illumination_ladder": [],
        "glare_ladder": [],
        "resolution_ladder": [],
    }

    # Defocus Blur Stress
    for sigma in [0.0, 1.5, 3.0, 4.5, 6.0]:
        perturbed = apply_gaussian_blur(ref_img, sigma)
        p_res = evaluator.infer(perturbed)
        severity = round(sigma / 6.0 * 100, 1)
        stress_tests["blur_ladder"].append({
            "param": f"σ={sigma:.1f}",
            "severity": severity,
            "predicted_grade": p_res["predicted_grade"],
            "confidence": p_res["confidence"],
            "retained_stability": compute_retention_stability(base_ref, p_res, severity),
        })

    # Illumination Drop Stress
    for factor in [1.0, 0.8, 0.6, 0.4, 0.2]:
        perturbed = apply_illumination_attenuation(ref_img, factor)
        p_res = evaluator.infer(perturbed)
        drop_pct = round((1.0 - factor) * 100, 1)
        stress_tests["illumination_ladder"].append({
            "param": f"-{drop_pct:.0f}%",
            "severity": drop_pct,
            "predicted_grade": p_res["predicted_grade"],
            "confidence": p_res["confidence"],
            "retained_stability": compute_retention_stability(base_ref, p_res, drop_pct),
        })

    # Corneal Glare Stress
    for intensity in [0.0, 0.25, 0.50, 0.75, 0.95]:
        perturbed = apply_corneal_glare(ref_img, intensity)
        p_res = evaluator.infer(perturbed)
        severity = round(intensity * 100, 1)
        stress_tests["glare_ladder"].append({
            "param": f"int={intensity:.2f}",
            "severity": severity,
            "predicted_grade": p_res["predicted_grade"],
            "confidence": p_res["confidence"],
            "retained_stability": compute_retention_stability(base_ref, p_res, severity),
        })

    # Resolution Scaling Stress
    for dim in [384, 288, 224, 160, 96]:
        perturbed = apply_resolution_scaling(ref_img, dim)
        p_res = evaluator.infer(perturbed)
        res_loss_pct = round((1.0 - (dim / 384.0)) * 100, 1)
        stress_tests["resolution_ladder"].append({
            "param": f"{dim}x{dim}",
            "severity": res_loss_pct,
            "predicted_grade": p_res["predicted_grade"],
            "confidence": p_res["confidence"],
            "retained_stability": compute_retention_stability(base_ref, p_res, res_loss_pct),
        })

    # 4. Clinical Sanity & Shortcut Learning Discrepancy Checks
    discrepancy_cases = []
    for name, res in baseline_predictions.items():
        if evaluator.has_masks:
            check = verify_lesion_classification_consensus(
                res["predicted_grade"], res["confidence"], res["biomarkers"]
            )
        else:
            # Classification-only models: compare prediction against confirmed clinical ground-truth
            gt_grade = ground_truth_labels.get(name) if ground_truth_labels else None
            pred_g = res["predicted_grade"]
            if gt_grade is not None:
                if pred_g == 0 and gt_grade >= 2:
                    check = {
                        "is_safe": False,
                        "predicted_grade": pred_g,
                        "physical_biomarker_grade": gt_grade,
                        "violation_type": "CRITICAL_SILENT_FALSE_NEGATIVE",
                        "verdict": f"DANGER: Candidate model predicted Normal with {res['confidence']:.1f}% confidence on confirmed Grade {gt_grade} pathology.",
                        "evidence_summary": f"Confirmed clinical ground-truth Grade {gt_grade} missed by model.",
                    }
                elif pred_g >= 3 and gt_grade == 0:
                    check = {
                        "is_safe": False,
                        "predicted_grade": pred_g,
                        "physical_biomarker_grade": gt_grade,
                        "violation_type": "FALSE_POSITIVE_PANIC",
                        "verdict": f"OVERCLASSIFICATION: Model predicted severe disease (Grade {pred_g}) on confirmed normal tissue.",
                        "evidence_summary": "Confirmed normal clinical ground-truth.",
                    }
                else:
                    check = {"is_safe": True}
            else:
                check = {"is_safe": True}

        if not check.get("is_safe", True):
            discrepancy_cases.append({
                "sample_id": name,
                "predicted_grade": check["predicted_grade"],
                "biomarker_grade": check.get("physical_biomarker_grade", 0),
                "violation_type": check["violation_type"],
                "verdict": check["verdict"],
                "evidence": check["evidence_summary"],
            })

    # 5. Composite Hospital Readiness Score & Certification
    # Score out of 100
    stability_avg = np.mean([
        np.mean([s["retained_stability"] for s in stress_tests["blur_ladder"]]),
        np.mean([s["retained_stability"] for s in stress_tests["illumination_ladder"]]),
        np.mean([s["retained_stability"] for s in stress_tests["glare_ladder"]]),
    ])
    calibration_penalty = min(35.0, calibration_metrics["ece"] * 100.0 * 2.5)
    discrepancy_penalty = len(discrepancy_cases) * 15.0

    readiness_score = max(0.0, min(100.0, stability_avg - calibration_penalty - discrepancy_penalty + 30.0))
    readiness_score = round(readiness_score, 1)

    if readiness_score >= 80.0 and len(discrepancy_cases) == 0:
        verdict = "APPROVED_HOSPITAL_READY"
        verdict_title = "APPROVED: Hospital Deployment Ready"
        guardrail_policy = "Model clears safety gates. Routine quarterly recalibration recommended."
    elif readiness_score >= 60.0:
        verdict = "CONDITIONAL_PASS_GUARDRAILS_REQUIRED"
        verdict_title = "CONDITIONAL PASS: Edge IQA Guardrails Mandated"
        guardrail_policy = "Deployable ONLY with upstream hardware IQA gate rejecting scans with focus score Q < 0.65 or flash drop > 30%."
    else:
        verdict = "REJECTED_UNSAFE_FOR_PATIENT_CARE"
        verdict_title = "REJECTED: Unsafe for Independent Clinical Use"
        guardrail_policy = "High vulnerability to optical noise and silent false negatives. Requires retraining with multi-source regularization."

    optical_robustness = round(float(stability_avg), 1)
    calibration_precision = round(max(0.0, 100.0 - float(calibration_metrics["ece"] * 100.0)), 1)
    utility_margin = round(max(0.0, 100.0 - (len(discrepancy_cases) * 25.0)), 1)

    audit_summary = {
        "audit_id": f"TC-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_name": evaluator.model_name,
        "readiness_score": readiness_score,
        "optical_robustness": optical_robustness,
        "calibration_precision": calibration_precision,
        "utility_margin": utility_margin,
        "verdict": verdict,
        "verdict_title": verdict_title,
        "guardrail_policy": guardrail_policy,
        "calibration": calibration_metrics,
        "discrepancies": discrepancy_cases,
        "stress_tests": stress_tests,
        "samples_audited": len(sample_images),
        "audit_duration_sec": round(time.time() - start_time, 2),
    }

    return audit_summary
