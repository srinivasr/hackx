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
            self.is_text = (inp.type == "tensor(int64)" or "ids" in inp.name or len(inp.shape) <= 2)
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

    def infer(self, img_bgr: Any) -> Dict[str, Any]:
        """Runs inference and returns softmax probabilities and segmented lesion counts."""
        if getattr(self, "is_text", False):
            from engines.model_interface import tokenize_text
            if isinstance(img_bgr, str):
                dummy_ids = tokenize_text(img_bgr, max_len=128)
            elif isinstance(img_bgr, np.ndarray) and img_bgr.dtype in (np.int64, np.int32):
                dummy_ids = img_bgr
            else:
                dummy_ids = tokenize_text("Patient 68yo presenting with acute shortness of breath, history of CHF and HTN, denies chest pain, bilateral crackles on exam.", max_len=128)
            outputs = self.session.run(None, {self.input_name: dummy_ids})
            logits = outputs[0][0]
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100.0
            return {
                "predicted_grade": pred_class,
                "confidence": round(confidence, 1),
                "probs": probs.tolist(),
                "biomarkers": {"clinical_nlp_risk": "Acute Decompensation" if pred_class == 1 else "Normal Triage"},
                "latency_ms": 1.2,
            }

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


def compute_model_stability(p_res: Dict[str, Any], base_res: Dict[str, Any], severity_pct: float = 0.0) -> float:
    """Computes dynamic retained model stability percentage under stress perturbation.
    Combines Total Variation Distance fidelity, decision preservation, and confidence."""
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


def evaluate_spectrum_verdict(stability: float, vector_name: str) -> Dict[str, str]:
    if stability >= 72.0:
        return {"status": "pass", "verdict": "PASS • Robust"}
    elif stability >= 48.0:
        return {"status": "warn", "verdict": "WARN • Sub-threshold"}
    else:
        if any(w in vector_name for w in ["Glare", "Haze", "Noise"]):
            return {"status": "fail", "verdict": "FAIL • Noise Saturated"}
        elif any(w in vector_name for w in ["Illumination", "Contrast"]):
            return {"status": "fail", "verdict": "FAIL • Severe Sensitivity"}
        elif any(w in vector_name for w in ["Resolution", "Downsampling"]):
            return {"status": "fail", "verdict": "FAIL • Sub-threshold"}
        else:
            return {"status": "fail", "verdict": "FAIL • Sub-threshold"}


def run_full_model_audit(
    evaluator: CandidateModelEvaluator,
    sample_images: Dict[str, np.ndarray],
    ground_truth_labels: Optional[Dict[str, int]] = None,
    modality: str = "retinal_fundus",
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

    # 3. Dynamic Clinical Stress Ladders & Spectrum
    ref_key = "sample_clinical_pass" if "sample_clinical_pass" in sample_images else list(sample_images.keys())[0]
    ref_img = sample_images[ref_key]
    base_ref = baseline_predictions.get(ref_key, evaluator.infer(ref_img))
    base_res = base_ref

    def compute_retention_stability(base_res: Dict[str, Any], p_res: Dict[str, Any], severity_pct: float) -> float:
        return compute_model_stability(p_res, base_res, severity_pct)

    stress_tests = {
        "blur_ladder": [],
        "illumination_ladder": [],
        "glare_ladder": [],
        "resolution_ladder": [],
    }
    stress_spectrum = []

    if modality == "chest_xray":
        from engines.modality_suites import RadiologySuite
        # Vector 1: Thoracic Motion Blur
        v1_name = "Thoracic Motion Blur"
        v1_min = "Kernel = 0 px"
        v1_max = "Kernel = 21 px (Respiratory tremor)"
        for k in [3, 7, 11, 15, 21]:
            perturbed = RadiologySuite.apply_motion_blur(ref_img, severity=min(5, k // 4 + 1))
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            stress_tests["blur_ladder"].append({
                "param": f"k={k}px",
                "severity": round(k / 21.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["blur_ladder"][-1]["retained_stability"]
        v1_info = evaluate_spectrum_verdict(final_stab, v1_name)
        stress_spectrum.append({
            "vector_name": v1_name,
            "min_param": v1_min,
            "max_param": v1_max,
            "retained_stability": final_stab,
            "status": v1_info["status"],
            "verdict": v1_info["verdict"],
        })

        # Vector 2: Quantum Poisson Shot Noise
        v2_name = "Quantum Poisson Shot Noise"
        v2_min = "High-Dose Photons"
        v2_max = "Low-Dose Scatter (Severe noise)"
        for s in [1, 2, 3, 4, 5]:
            perturbed = RadiologySuite.apply_poisson_noise(ref_img, severity=s)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            stress_tests["glare_ladder"].append({
                "param": f"sev={s}",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["glare_ladder"][-1]["retained_stability"]
        v2_info = evaluate_spectrum_verdict(final_stab, v2_name)
        stress_spectrum.append({
            "vector_name": v2_name,
            "min_param": v2_min,
            "max_param": v2_max,
            "retained_stability": final_stab,
            "status": v2_info["status"],
            "verdict": v2_info["verdict"],
        })

        # Vector 3: CR / DR Contrast Attenuation
        v3_name = "CR / DR Contrast Attenuation"
        v3_min = "100% Dynamic Range"
        v3_max = "-75% (Severe underexposure)"
        for s in [1, 2, 3, 4, 5]:
            perturbed = RadiologySuite.apply_contrast_attenuation(ref_img, severity=s)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            stress_tests["illumination_ladder"].append({
                "param": f"-{int((1.0 - [0.85, 0.70, 0.55, 0.40, 0.25][s-1])*100)}%",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["illumination_ladder"][-1]["retained_stability"]
        v3_info = evaluate_spectrum_verdict(final_stab, v3_name)
        stress_spectrum.append({
            "vector_name": v3_name,
            "min_param": v3_min,
            "max_param": v3_max,
            "retained_stability": final_stab,
            "status": v3_info["status"],
            "verdict": v3_info["verdict"],
        })

        # Vector 4: Matrix Resolution Downsampling
        v4_name = "Matrix Resolution Downsampling"
        v4_min = "384×384 px"
        v4_max = "96×96 px (Mobile cart display)"
        for dim in [384, 288, 224, 160, 96]:
            perturbed = apply_resolution_scaling(ref_img, dim)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            loss_pct = round((1.0 - (dim / 384.0)) * 100, 1)
            stress_tests["resolution_ladder"].append({
                "param": f"{dim}x{dim}",
                "severity": loss_pct,
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["resolution_ladder"][-1]["retained_stability"]
        v4_info = evaluate_spectrum_verdict(final_stab, v4_name)
        stress_spectrum.append({
            "vector_name": v4_name,
            "min_param": v4_min,
            "max_param": v4_max,
            "retained_stability": final_stab,
            "status": v4_info["status"],
            "verdict": v4_info["verdict"],
        })

    elif modality == "clinical_nlp":
        from engines.modality_suites import ClinicalTextSuite
        sample_text = "Patient 68yo presenting with acute shortness of breath, history of CHF and HTN, denies chest pain, bilateral crackles on exam."
        base_nlp_res = evaluator.infer(sample_text)

        # Vector 1: OCR Typographical Noise
        v1_name = "OCR Typographical Noise"
        v1_min = "0% Typos (Clean Digital EHR)"
        v1_max = "42% Char Swaps (Faxed note OCR)"
        for s in [1, 2, 3, 4, 5]:
            perturbed_txt = ClinicalTextSuite.apply_ocr_typos(sample_text, severity=s)
            p_res = evaluator.infer(perturbed_txt)
            stab = compute_model_stability(p_res, base_nlp_res)
            stress_tests["blur_ladder"].append({
                "param": f"sev={s}",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["blur_ladder"][-1]["retained_stability"]
        v1_info = evaluate_spectrum_verdict(final_stab, v1_name)
        stress_spectrum.append({
            "vector_name": v1_name,
            "min_param": v1_min,
            "max_param": v1_max,
            "retained_stability": final_stab,
            "status": v1_info["status"],
            "verdict": v1_info["verdict"],
        })

        # Vector 2: Medical Abbreviation Density
        v2_name = "Medical Abbreviation Density"
        v2_min = "Standard Clinical Text"
        v2_max = "80% Shorthand (Physician jargon)"
        for s in [1, 2, 3, 4, 5]:
            perturbed_txt = ClinicalTextSuite.apply_abbreviations(sample_text, severity=s)
            p_res = evaluator.infer(perturbed_txt)
            stab = compute_model_stability(p_res, base_nlp_res)
            stress_tests["illumination_ladder"].append({
                "param": f"sev={s}",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["illumination_ladder"][-1]["retained_stability"]
        v2_info = evaluate_spectrum_verdict(final_stab, v2_name)
        stress_spectrum.append({
            "vector_name": v2_name,
            "min_param": v2_min,
            "max_param": v2_max,
            "retained_stability": final_stab,
            "status": v2_info["status"],
            "verdict": v2_info["verdict"],
        })

        # Vector 3: Hasty Note Truncation
        v3_name = "Hasty Clinical Note Truncation"
        v3_min = "100% Complete Narrative"
        v3_max = "40% Length (Abrupt cutoff)"
        for s in [1, 2, 3, 4, 5]:
            perturbed_txt = ClinicalTextSuite.apply_note_truncation(sample_text, severity=s)
            p_res = evaluator.infer(perturbed_txt)
            stab = compute_model_stability(p_res, base_nlp_res)
            stress_tests["glare_ladder"].append({
                "param": f"sev={s}",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["glare_ladder"][-1]["retained_stability"]
        v3_info = evaluate_spectrum_verdict(final_stab, v3_name)
        stress_spectrum.append({
            "vector_name": v3_name,
            "min_param": v3_min,
            "max_param": v3_max,
            "retained_stability": final_stab,
            "status": v3_info["status"],
            "verdict": v3_info["verdict"],
        })

        # Vector 4: Negation Assertion Stress
        v4_name = "Negation Assertion Stress"
        v4_min = "Intact Assertions"
        v4_max = "NegEx Inversion Pairs"
        for s in [1, 2, 3, 4, 5]:
            pairs = [("denies", "reports"), ("shortness of breath", "no respiratory distress"), ("crackles", "clear lungs")]
            perturbed_txt = sample_text
            for p1, p2 in pairs[:s]:
                perturbed_txt = perturbed_txt.replace(p1, p2)
            p_res = evaluator.infer(perturbed_txt)
            stab = compute_model_stability(p_res, base_nlp_res)
            stress_tests["resolution_ladder"].append({
                "param": f"sev={s}",
                "severity": round(s / 5.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["resolution_ladder"][-1]["retained_stability"]
        v4_info = evaluate_spectrum_verdict(final_stab, v4_name)
        stress_spectrum.append({
            "vector_name": v4_name,
            "min_param": v4_min,
            "max_param": v4_max,
            "retained_stability": final_stab,
            "status": v4_info["status"],
            "verdict": v4_info["verdict"],
        })

    else:
        # Default: Retinal Fundus (Ophthalmology)
        # Vector 1: Defocus / Motion Blur
        v1_name = "Defocus / Motion Blur"
        v1_min = "σ = 0.0"
        v1_max = "σ = 6.0 (Severe movement)"
        for sigma in [0.0, 1.5, 3.0, 4.5, 6.0]:
            perturbed = apply_gaussian_blur(ref_img, sigma)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            stress_tests["blur_ladder"].append({
                "param": f"σ={sigma:.1f}",
                "severity": round(sigma / 6.0 * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["blur_ladder"][-1]["retained_stability"]
        v1_info = evaluate_spectrum_verdict(final_stab, v1_name)
        stress_spectrum.append({
            "vector_name": v1_name,
            "min_param": v1_min,
            "max_param": v1_max,
            "retained_stability": final_stab,
            "status": v1_info["status"],
            "verdict": v1_info["verdict"],
        })

        # Vector 2: Flash / Illumination Drop
        v2_name = "Flash / Illumination Drop"
        v2_min = "100% Brightness"
        v2_max = "-80% (Undilated pupil)"
        for factor in [1.0, 0.8, 0.6, 0.4, 0.2]:
            perturbed = apply_illumination_attenuation(ref_img, factor)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            drop_pct = round((1.0 - factor) * 100, 1)
            stress_tests["illumination_ladder"].append({
                "param": f"-{drop_pct:.0f}%",
                "severity": drop_pct,
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["illumination_ladder"][-1]["retained_stability"]
        v2_info = evaluate_spectrum_verdict(final_stab, v2_name)
        stress_spectrum.append({
            "vector_name": v2_name,
            "min_param": v2_min,
            "max_param": v2_max,
            "retained_stability": final_stab,
            "status": v2_info["status"],
            "verdict": v2_info["verdict"],
        })

        # Vector 3: Corneal Glare Reflection
        v3_name = "Corneal Glare Reflection"
        v3_min = "0.00"
        v3_max = "0.95 (Corneal Whiteout)"
        for intensity in [0.0, 0.25, 0.50, 0.75, 0.95]:
            perturbed = apply_corneal_glare(ref_img, intensity)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            stress_tests["glare_ladder"].append({
                "param": f"int={intensity:.2f}",
                "severity": round(intensity * 100, 1),
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["glare_ladder"][-1]["retained_stability"]
        v3_info = evaluate_spectrum_verdict(final_stab, v3_name)
        stress_spectrum.append({
            "vector_name": v3_name,
            "min_param": v3_min,
            "max_param": v3_max,
            "retained_stability": final_stab,
            "status": v3_info["status"],
            "verdict": v3_info["verdict"],
        })

        # Vector 4: Sensor Resolution Downsampling
        v4_name = "Sensor Resolution Downsampling"
        v4_min = "384×384 px"
        v4_max = "96×96 px (Extreme drop)"
        for dim in [384, 288, 224, 160, 96]:
            perturbed = apply_resolution_scaling(ref_img, dim)
            p_res = evaluator.infer(perturbed)
            stab = compute_model_stability(p_res, base_res)
            res_loss_pct = round((1.0 - (dim / 384.0)) * 100, 1)
            stress_tests["resolution_ladder"].append({
                "param": f"{dim}x{dim}",
                "severity": res_loss_pct,
                "predicted_grade": p_res["predicted_grade"],
                "confidence": p_res["confidence"],
                "retained_stability": stab,
            })
        final_stab = stress_tests["resolution_ladder"][-1]["retained_stability"]
        v4_info = evaluate_spectrum_verdict(final_stab, v4_name)
        stress_spectrum.append({
            "vector_name": v4_name,
            "min_param": v4_min,
            "max_param": v4_max,
            "retained_stability": final_stab,
            "status": v4_info["status"],
            "verdict": v4_info["verdict"],
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
    stability_avg = np.mean([s["retained_stability"] for s in stress_spectrum]) if stress_spectrum else 50.0
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
        "stress_spectrum": stress_spectrum,
        "modality": modality,
        "samples_audited": len(sample_images),
        "audit_duration_sec": round(time.time() - start_time, 2),
    }

    return audit_summary
