"""Step 3: Engine C - Uncertainty, Calibration & Silent Failure Engine.

Computes 10-bin Expected Calibration Error (ECE), Adaptive Calibration Error (ACE),
Class-Conditional ECE, optimal post-hoc Temperature Scaling (T*), Shannon entropy H(p),
Mahalanobis & Free-Energy OOD detection, dangerous overconfident false-negative flags,
and deterministic composite TrustScore gating with concrete clinical contraindications.

Literature Grounding:
- Guo et al. (ICML 2017): On calibration of modern neural networks.
- Nixon et al. (CVPR 2019): Measuring Calibration in Deep Learning (Adaptive Calibration Error).
- Liu et al. (NeurIPS 2020): Energy-based Out-of-distribution Detection.
- Lee et al. (NeurIPS 2018): Simple unified framework for detecting OOD samples.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculates Expected Calibration Error (ECE) across M equal-width confidence intervals.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.clip(confidences, 0.0, 1.0)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_bins = []
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(predictions[in_bin] == labels[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_weight = bin_count / total_samples
            ece += bin_weight * abs(bin_acc - bin_conf)

            reliability_bins.append({
                "bin_id": i + 1,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": bin_count,
                "confidence": round(bin_conf, 3),
                "accuracy": round(bin_acc, 3),
                "calibration_gap": round(abs(bin_acc - bin_conf), 3),
            })
        else:
            reliability_bins.append({
                "bin_id": i + 1,
                "range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": 0,
                "confidence": round((bin_lower + bin_upper) / 2.0, 3),
                "accuracy": 0.0,
                "calibration_gap": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "ece_percent": round(float(ece) * 100.0, 2),
        "reliability_bins": reliability_bins,
    }


def compute_adaptive_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculates Adaptive Calibration Error (ACE) using equal-frequency quantile binning.
    
    Addresses the flaw of standard ECE where intermediate confidence bins are empty.
    Nixon et al. (CVPR 2019).
    """
    total_samples = len(confidences)
    if total_samples < n_bins:
        n_bins = max(2, total_samples // 2)

    # Sort by confidence
    sort_idx = np.argsort(confidences)
    sorted_confs = confidences[sort_idx]
    sorted_preds = predictions[sort_idx]
    sorted_labels = labels[sort_idx]

    bin_size = total_samples / n_bins
    ace = 0.0

    for i in range(n_bins):
        start_idx = int(round(i * bin_size))
        end_idx = int(round((i + 1) * bin_size))
        if start_idx >= end_idx:
            continue

        b_confs = sorted_confs[start_idx:end_idx]
        b_preds = sorted_preds[start_idx:end_idx]
        b_labels = sorted_labels[start_idx:end_idx]

        acc = np.mean(b_preds == b_labels)
        conf = np.mean(b_confs)
        weight = len(b_confs) / total_samples
        ace += weight * abs(acc - conf)

    return round(float(ace), 4)


def fit_temperature_scaling(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Fits post-hoc temperature scaling T* to test model overconfidence.
    
    Returns (optimal_temperature, post_recalibration_ece).
    Guo et al. (ICML 2017).
    """
    eps = 1e-7
    probs_clipped = np.clip(probabilities, eps, 1.0 - eps)
    # Reconstruct logit: log(p / (1 - p))
    logits = np.log(probs_clipped / (1.0 - probs_clipped))

    def nll_obj(T):
        scaled_logits = logits / max(1e-3, T)
        p_scaled = 1.0 / (1.0 + np.exp(-scaled_logits))
        # Binary cross entropy
        loss = -np.mean(labels * np.log(p_scaled + eps) + (1 - labels) * np.log(1 - p_scaled + eps))
        return loss

    res = minimize_scalar(nll_obj, bounds=(0.1, 5.0), method="bounded")
    best_t = float(res.x)

    # Compute post-recalibration ECE
    scaled_logits = logits / best_t
    p_recal = 1.0 / (1.0 + np.exp(-scaled_logits))
    preds_recal = (p_recal >= 0.50).astype(int)
    recal_ece = compute_ece(p_recal, preds_recal, labels)["ece"]

    return round(best_t, 2), round(recal_ece, 4)


def compute_shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """Calculates Shannon Entropy H(p) = -sum_c p_c * log2(p_c) per sample."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return entropy


def compute_free_energy(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Calculates Free Energy OOD score: E(x; T) = -T * log sum exp(logit / T).
    Liu et al. (NeurIPS 2020). Lower (more negative) energy indicates in-distribution.
    """
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    # 2-class logits: [0, logit]
    l0 = np.zeros_like(logits)
    l1 = logits
    # logsumexp
    max_l = np.maximum(l0, l1)
    lse = max_l + np.log(np.exp((l0 - max_l) / temperature) + np.exp((l1 - max_l) / temperature))
    energy = -temperature * lse
    return energy


def detect_mahalanobis_ood(
    embeddings: np.ndarray,
    ref_embeddings: Optional[np.ndarray] = None,
    percentile_threshold: float = 95.0,
) -> Tuple[np.ndarray, float, int]:
    """Computes Mahalanobis distance in latent space to identify out-of-distribution (OOD) cases."""
    ref = ref_embeddings if ref_embeddings is not None else embeddings
    mu = np.mean(ref, axis=0)
    
    cov = np.cov(ref, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.pinv(cov)

    distances = np.array([mahalanobis(x, mu, inv_cov) for x in embeddings])
    cutoff = float(np.percentile(distances, percentile_threshold))
    ood_mask = distances > cutoff
    ood_count = int(np.sum(ood_mask))

    return distances, cutoff, ood_count


class SafetyEngine:
    """Step 3: Quantifies model uncertainty, calibration error, and flags dangerous silent misclassifications."""

    def __init__(
        self,
        n_bins: int = 10,
        high_conf_threshold: float = 0.85,
        quarantine_conf_threshold: float = 0.90,
    ):
        self.n_bins = n_bins
        self.high_conf_threshold = high_conf_threshold
        self.quarantine_conf_threshold = quarantine_conf_threshold

    def evaluate(
        self,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs the complete uncertainty, calibration, and dangerous failure probe."""
        n_samples = len(predictions)
        p_ids = patient_ids or [f"P_{i:04d}" for i in range(n_samples)]

        # Class 1 probabilities and top-choice confidences
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            p_pos = probabilities[:, 1]
            confidences = np.max(probabilities, axis=1)
            prob_matrix = probabilities
        else:
            p_pos = probabilities.flatten()
            confidences = p_pos
            prob_matrix = np.stack([1.0 - p_pos, p_pos], axis=1)

        # 1. Expected Calibration Error (ECE) & Adaptive Calibration Error (ACE)
        ece_report = compute_ece(confidences, predictions, labels, n_bins=self.n_bins)
        ace_val = compute_adaptive_calibration_error(confidences, predictions, labels, n_bins=self.n_bins)

        # 2. Class-Conditional ECE (Clinical Asymmetry)
        pos_mask = labels == 1
        neg_mask = labels == 0
        ece_pos = compute_ece(confidences[pos_mask], predictions[pos_mask], labels[pos_mask], n_bins=max(2, self.n_bins // 2))["ece"] if np.sum(pos_mask) > 4 else ece_report["ece"]
        ece_neg = compute_ece(confidences[neg_mask], predictions[neg_mask], labels[neg_mask], n_bins=max(2, self.n_bins // 2))["ece"] if np.sum(neg_mask) > 4 else ece_report["ece"]

        # 3. Post-Hoc Temperature Scaling Diagnostic (Guo et al. 2017)
        try:
            best_t, recal_ece = fit_temperature_scaling(p_pos, labels)
            overconfident_flag = bool(best_t > 1.25)
        except Exception:
            best_t, recal_ece, overconfident_flag = 1.0, ece_report["ece"], False

        # 4. Brier Score
        brier = float(brier_score_loss(labels, p_pos))

        # 5. Shannon Entropy & Free-Energy OOD
        entropy_vals = compute_shannon_entropy(prob_matrix)
        mean_entropy = float(np.mean(entropy_vals))
        energy_vals = compute_free_energy(p_pos)
        mean_energy = float(np.mean(energy_vals))

        # 6. Dangerous Overconfidence Quadrant Probing
        # High confidence (>= 0.85), True Label = 1 (Pathology Positive), Model predicts 0 (Negative)
        critical_silent_failures = []
        for i in range(n_samples):
            if labels[i] == 1 and predictions[i] == 0 and confidences[i] >= self.high_conf_threshold:
                critical_silent_failures.append({
                    "patient_id": p_ids[i],
                    "confidence": round(float(confidences[i]), 3),
                    "entropy": round(float(entropy_vals[i]), 3),
                    "energy_score": round(float(energy_vals[i]), 3),
                    "predicted": 0,
                    "ground_truth": 1,
                    "severity": "CRITICAL_SILENT_FALSE_NEGATIVE",
                    "quarantine_triggered": bool(confidences[i] >= self.quarantine_conf_threshold),
                })

        # 7. Out-of-Distribution (OOD) Detection via Mahalanobis Distance
        ood_count = 0
        mean_maha_dist = 0.0
        if embeddings is not None and len(embeddings) > 5:
            dists, cutoff, ood_count = detect_mahalanobis_ood(embeddings)
            mean_maha_dist = float(np.mean(dists))

        # Calibration Score (0 - 100)
        ece_penalty = min(40.0, ece_report["ece"] * 100.0 * 1.5)
        ace_penalty = min(20.0, ace_val * 100.0 * 1.0)
        brier_penalty = min(20.0, brier * 40.0)
        silent_penalty = min(30.0, len(critical_silent_failures) * 10.0)
        
        calibration_score = max(0.0, min(100.0, 100.0 - (ece_penalty + ace_penalty + brier_penalty + silent_penalty)))

        return {
            "expected_calibration_error": ece_report["ece"],
            "ece_percent": ece_report["ece_percent"],
            "adaptive_calibration_error": ace_val,
            "class_conditional_ece": {
                "positive_pathology_ece": round(ece_pos, 4),
                "negative_pathology_ece": round(ece_neg, 4),
            },
            "temperature_scaling_diagnostic": {
                "optimal_temperature_T": best_t,
                "is_overconfident": overconfident_flag,
                "post_recalibration_ece": recal_ece,
            },
            "brier_score": round(brier, 4),
            "mean_shannon_entropy": round(mean_entropy, 4),
            "mean_free_energy": round(mean_energy, 4),
            "mean_mahalanobis_distance": round(mean_maha_dist, 2),
            "out_of_distribution_detected_count": ood_count,
            "critical_silent_failures_count": len(critical_silent_failures),
            "critical_silent_failures": critical_silent_failures,
            "reliability_bins": ece_report["reliability_bins"],
            "calibration_score": round(calibration_score, 1),
        }

    @staticmethod
    def compute_composite_trust_score(
        robustness_score: float,
        fairness_score: float,
        calibration_score: float,
        shift_score: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Calculates deterministic composite TrustScore:
        TrustScore = 0.35 * S_rob + 0.30 * S_fair + 0.20 * S_cal + 0.15 * S_shift
        """
        w = weights or {"robustness": 0.35, "fairness": 0.30, "calibration": 0.20, "shift": 0.15}
        score = (
            w["robustness"] * robustness_score
            + w["fairness"] * fairness_score
            + w["calibration"] * calibration_score
            + w["shift"] * shift_score
        )
        return round(max(0.0, min(100.0, float(score))), 1)

    @staticmethod
    def evaluate_gating(
        trust_score: float,
        max_subgroup_disparity: float,
        critical_silent_failures_count: int,
        total_samples: int,
        contrast_decay_failure: bool = False,
    ) -> Tuple[str, List[str]]:
        """Applies deterministic regulatory gating rules and formats concrete clinical contraindications."""
        high_conf_miss_pct = (critical_silent_failures_count / max(1, total_samples)) * 100.0
        contraindications = []

        if max_subgroup_disparity > 1.80:
            contraindications.append(
                f"CONTRAINDICATION: Model exhibits {max_subgroup_disparity:.2f}x under-diagnosis disparity ratio on vulnerable demographic cohort. Autonomous clinical triage blocked."
            )
        elif max_subgroup_disparity > 1.40:
            contraindications.append(
                f"CAUTION: Elevated subgroup disparity ({max_subgroup_disparity:.2f}x) detected. Mandatory equity audit required."
            )

        if contrast_decay_failure:
            contraindications.append(
                "CONTRAINDICATION: Hardware failure on Computed Radiography (CR) scanners when contrast drops > 15%. Mandatory radiologist re-read required."
            )

        if critical_silent_failures_count > 0:
            contraindications.append(
                f"SAFETY HAZARD: {critical_silent_failures_count} high-confidence false-negative miss(es) detected in active pathology cases (Conf > 0.85). PACS node quarantine triggered."
            )

        # Gating Decision Logic
        if trust_score < 60.0 or high_conf_miss_pct > 5.0 or critical_silent_failures_count > 3:
            verdict = "NO_GO_REJECTED"
        elif trust_score >= 80.0 and len(contraindications) == 0 and critical_silent_failures_count == 0:
            verdict = "GO_APPROVED"
        else:
            verdict = "CAUTION_RESTRICTED_DEPLOYMENT"

        return verdict, contraindications
