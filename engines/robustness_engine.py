"""Step 1: Engine A - Physics & Hardware Shift Robustness Engine.

Stress-tests candidate clinical models across 5-tier intensity corruptions
(contrast attenuation, Poisson shot noise, motion blur) and non-clinical
adversarial visual shortcuts (anatomical L/R markers, hospital text annotations,
metallic hardware density, and collimator border cropping).

Literature Grounding:
- DeGrave et al. (Nature Machine Intelligence 2021): AI for radiographic detection
  frequently relies on superficial shortcuts (tokens, markers, borders).
- Zech et al. (PLoS Medicine 2018): Variable generalization across hospital systems.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score


def apply_contrast_attenuation(img: np.ndarray, severity: int) -> np.ndarray:
    """Simulates low-grade computed radiography (CR) plates via contrast reduction.
    
    severity: 1 (slight attenuation) to 5 (severe wash-out).
    """
    factors = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25}
    factor = factors.get(severity, 0.55)
    img_f = img.astype(np.float32)
    mean_val = np.mean(img_f)
    attenuated = mean_val + factor * (img_f - mean_val)
    return np.clip(attenuated, 0, 255).astype(np.uint8)


def apply_poisson_noise(img: np.ndarray, severity: int, seed: Optional[int] = 42) -> np.ndarray:
    """Simulates reduced radiation exposure protocols (e.g. pediatric beds) using Poisson shot noise.
    
    severity: 1 (subtle quantum mottle) to 5 (heavy photon starvation).
    Uses deterministic RNG for audit reproducibility.
    """
    scale_factors = {1: 150.0, 2: 80.0, 3: 40.0, 4: 20.0, 5: 10.0}
    scale = scale_factors.get(severity, 40.0)
    
    img_norm = np.maximum(img.astype(np.float32), 1e-3) / 255.0
    photons = img_norm * scale
    
    rng = np.random.default_rng(seed)
    noisy_photons = rng.poisson(photons).astype(np.float32)
    noisy = (noisy_photons / scale) * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Simulates patient movement in emergency room or pediatric radiography.
    
    severity: 1 (3px kernel) to 5 (21px kernel).
    """
    kernel_sizes = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}
    k = kernel_sizes.get(severity, 11)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred.astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-Archetype Non-Clinical Distractor / Shortcut Injection Suite
# ---------------------------------------------------------------------------

def inject_anatomical_distractor(img: np.ndarray, marker: str = "R") -> np.ndarray:
    """Injects high-contrast radiological orientation marker ('L' / 'R' lead stamp)
    with backing plaque into the top-right corner.
    """
    out = img.copy()
    h, w = out.shape[:2]
    stamp_w, stamp_h = max(16, int(w * 0.12)), max(16, int(h * 0.12))
    x1, y1 = max(0, w - stamp_w - 10), 10
    x2, y2 = min(w, w - 10), 10 + stamp_h
    
    # Lead marker backing plaque
    out[y1:y2, x1:x2] = 20
    # Draw high-contrast letter
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, stamp_h / 45.0)
    thickness = max(2, int(font_scale * 2.5))
    text_size = cv2.getTextSize(marker, font, font_scale, thickness)[0]
    text_x = x1 + (stamp_w - text_size[0]) // 2
    text_y = y1 + (stamp_h + text_size[1]) // 2
    cv2.putText(out, marker, (text_x, text_y), font, font_scale, 250, thickness, cv2.LINE_AA)
    return out


def inject_hospital_text_stamp(img: np.ndarray, text: str = "PORTABLE CHEST 08:30") -> np.ndarray:
    """Simulates burned-in clinical acquisition text banners across the bottom border.
    Documented in DeGrave et al. (2021) as a major source of spurious hospital shortcut bias.
    """
    out = img.copy()
    h, w = out.shape[:2]
    banner_h = max(14, int(h * 0.08))
    # Dark banner bar at bottom
    out[h - banner_h : h, :] = 10
    # Text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, banner_h / 50.0)
    thickness = 1
    cv2.putText(out, text, (8, h - int(banner_h * 0.3)), font, font_scale, 240, thickness, cv2.LINE_AA)
    return out


def inject_metallic_hardware_artifact(img: np.ndarray) -> np.ndarray:
    """Simulates radio-opaque metallic foreign body (pacemaker / surgical clips).
    Models frequently associate metallic radio-opacities with severe pathology shortcuts.
    """
    out = img.copy()
    h, w = out.shape[:2]
    # Pacemaker generator shadow in upper left quadrant
    center = (int(w * 0.25), int(h * 0.25))
    axes = (max(8, int(w * 0.07)), max(6, int(h * 0.05)))
    cv2.ellipse(out, center, axes, 15, 0, 360, 250, -1)
    # Surgical wire / lead trajectory
    pt1 = (center[0] + axes[0], center[1])
    pt2 = (int(w * 0.5), int(h * 0.55))
    cv2.line(out, pt1, pt2, 240, max(1, int(w * 0.01)))
    return out


def inject_collimator_aperture_border(img: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    """Simulates physical X-ray collimator crop shutters / unexposed black border edges."""
    out = img.copy()
    h, w = out.shape[:2]
    bx = int(w * border_fraction)
    by = int(h * border_fraction)
    out[:by, :] = 0
    out[h - by :, :] = 0
    out[:, :bx] = 0
    out[:, w - bx :] = 0
    return out


class RobustnessEngine:
    """Step 1: Executes physics corruptions, multi-archetype shortcut evaluations,
    and calculates AUROC/recall degradation decay slopes.
    """

    def __init__(self, intensity_tiers: Optional[List[int]] = None):
        self.tiers = intensity_tiers or [1, 2, 3, 4, 5]

    def evaluate_cohort(
        self,
        images: List[np.ndarray],
        labels: List[int],
        predict_fn: Callable[[List[np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Any]:
        """Runs the complete robustness & shortcut evaluation battery over a cohort.
        
        predict_fn signature: images -> (confidences: np.ndarray, predictions: np.ndarray)
        where confidences are in [0.0, 1.0].
        """
        clean_confs, clean_preds = predict_fn(images)
        clean_auroc = self._safe_auroc(labels, clean_confs)
        clean_recall = recall_score(labels, clean_preds, zero_division=0)

        decay_slopes = {
            "contrast_attenuation": [],
            "poisson_shot_noise": [],
            "motion_blur": [],
        }
        recall_slopes = {
            "contrast_attenuation": [],
            "poisson_shot_noise": [],
            "motion_blur": [],
        }

        # 1. Contrast Attenuation Ladder
        for tier in self.tiers:
            corrupted = [apply_contrast_attenuation(img, tier) for img in images]
            confs, preds = predict_fn(corrupted)
            decay_slopes["contrast_attenuation"].append(round(self._safe_auroc(labels, confs), 4))
            recall_slopes["contrast_attenuation"].append(round(recall_score(labels, preds, zero_division=0), 4))

        # 2. Poisson Shot Noise Ladder
        for tier in self.tiers:
            corrupted = [apply_poisson_noise(img, tier, seed=42 + tier) for img in images]
            confs, preds = predict_fn(corrupted)
            decay_slopes["poisson_shot_noise"].append(round(self._safe_auroc(labels, confs), 4))
            recall_slopes["poisson_shot_noise"].append(round(recall_score(labels, preds, zero_division=0), 4))

        # 3. Motion Blur Ladder
        for tier in self.tiers:
            corrupted = [apply_motion_blur(img, tier) for img in images]
            confs, preds = predict_fn(corrupted)
            decay_slopes["motion_blur"].append(round(self._safe_auroc(labels, confs), 4))
            recall_slopes["motion_blur"].append(round(recall_score(labels, preds, zero_division=0), 4))

        # 4. Multi-Archetype Shortcut / Non-Clinical Distractor Battery
        # Evaluated on clean true-negative cases (labels == 0) to measure false positive triggers
        neg_indices = [i for i, y in enumerate(labels) if y == 0]
        shortcut_diagnostics: Dict[str, Any] = {}
        
        if neg_indices:
            neg_images = [images[i] for i in neg_indices]
            neg_orig_confs, orig_neg_preds = predict_fn(neg_images)
            n_neg = len(neg_indices)

            distractor_types = {
                "laterality_stamp_R": [inject_anatomical_distractor(img, "R") for img in neg_images],
                "laterality_stamp_L": [inject_anatomical_distractor(img, "L") for img in neg_images],
                "hospital_text_stamp": [inject_hospital_text_stamp(img) for img in neg_images],
                "metallic_hardware": [inject_metallic_hardware_artifact(img) for img in neg_images],
                "collimator_aperture": [inject_collimator_aperture_border(img) for img in neg_images],
            }

            fp_rates = {}
            conf_deltas = {}
            for name, d_imgs in distractor_types.items():
                d_confs, d_preds = predict_fn(d_imgs)
                # Cases flipped from 0 to 1
                flips = sum(1 for o, d in zip(orig_neg_preds, d_preds) if o == 0 and d == 1)
                fp_rates[name] = round(flips / n_neg, 4)
                conf_deltas[name] = round(float(np.mean(np.abs(d_confs - neg_orig_confs))), 4)

            primary_fp_rate = fp_rates["laterality_stamp_R"]
            mean_conf_delta = float(np.mean(list(conf_deltas.values())))
            shortcut_diagnostics = {
                "distractor_fp_rates": fp_rates,
                "distractor_conf_deltas": conf_deltas,
                "shortcut_vulnerability_index": round(mean_conf_delta, 4),
            }
        else:
            primary_fp_rate = 0.0
            shortcut_diagnostics = {
                "distractor_fp_rates": {},
                "distractor_conf_deltas": {},
                "shortcut_vulnerability_index": 0.0,
            }

        # Calculate composite robustness score (0 - 100)
        mean_tier3_auroc = np.mean([slopes[2] for slopes in decay_slopes.values()])
        mean_tier5_auroc = np.mean([slopes[4] for slopes in decay_slopes.values()])
        retention = (0.6 * mean_tier3_auroc + 0.4 * mean_tier5_auroc) / max(0.5, clean_auroc)
        
        # Distractor & shortcut penalty
        distractor_penalty = min(25.0, primary_fp_rate * 100.0 * 2.0)
        shortcut_penalty = min(15.0, shortcut_diagnostics["shortcut_vulnerability_index"] * 100.0 * 0.5)
        
        robustness_score = max(0.0, min(100.0, (retention * 100.0) - distractor_penalty - shortcut_penalty))

        # Check for catastrophic contrast collapse
        contrast_t3 = decay_slopes["contrast_attenuation"][2] if len(decay_slopes["contrast_attenuation"]) >= 3 else 0.8
        contrast_failure = bool(contrast_t3 < 0.70)

        return {
            "clean_auroc": round(clean_auroc, 4),
            "clean_recall": round(clean_recall, 4),
            "decay_slopes": decay_slopes,
            "recall_slopes": recall_slopes,
            "distractor_false_positive_rate": primary_fp_rate,
            "shortcut_diagnostics": shortcut_diagnostics,
            "contrast_decay_failure": contrast_failure,
            "robustness_score": round(robustness_score, 1),
        }

    def _safe_auroc(self, y_true: List[int], y_score: np.ndarray) -> float:
        if len(set(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return 0.5
