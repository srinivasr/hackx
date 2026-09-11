"""Step 1: Engine A - Physics & Hardware Shift Robustness Engine.

Stress-tests candidate clinical models across 5-tier intensity corruptions
and non-clinical adversarial visual shortcuts across pluggable modalities:
- Chest Radiography (Poisson shot noise, CR contrast attenuation, motion blur, lead stamps, collimators)
- Retinal Fundus (Cataract media haze, non-mydriatic illumination falloff, handheld tremor blur, lens dust)
- Tabular EHR (Missing value MCAR injection, sensor jitter, outlier spikes)

Literature Grounding:
- DeGrave et al. (Nature Machine Intelligence 2021): Radiology visual shortcuts.
- Zech et al. (PLoS Medicine 2018): Variable generalization across hospital systems.
- Gulshan et al. (JAMA 2016): Retinal fundus quality degradation.
"""

from typing import Dict, Any, List, Tuple, Callable, Optional
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score
from engines.modality_suites import get_modality_suite, RadiologySuite

# Backward-compatible functional aliases
apply_contrast_attenuation = RadiologySuite.apply_contrast_attenuation
apply_poisson_noise = RadiologySuite.apply_poisson_noise
apply_motion_blur = RadiologySuite.apply_motion_blur
inject_anatomical_distractor = lambda img, marker="R": RadiologySuite._stamp(img, marker)
inject_hospital_text_stamp = RadiologySuite._text_banner
inject_metallic_hardware_artifact = RadiologySuite._pacemaker
inject_collimator_aperture_border = RadiologySuite._collimator


class RobustnessEngine:
    """Step 1: Executes physics corruptions, multi-archetype shortcut evaluations,
    and calculates AUROC/recall degradation decay slopes across modalities.
    """

    def __init__(self, intensity_tiers: Optional[List[int]] = None, modality: str = "chest_xray"):
        self.tiers = intensity_tiers or [1, 2, 3, 4, 5]
        self.modality = modality
        self.suite = get_modality_suite(modality)

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

        decay_slopes: Dict[str, List[float]] = {}
        recall_slopes: Dict[str, List[float]] = {}

        # 1. Dynamic Corruption Ladders from active Modality Suite
        ladder_fns = self.suite.get_corruption_ladder()
        for corr_name, corr_fn in ladder_fns.items():
            decay_slopes[corr_name] = []
            recall_slopes[corr_name] = []
            for tier in self.tiers:
                corrupted = [corr_fn(img, tier) for img in images]
                confs, preds = predict_fn(corrupted)
                decay_slopes[corr_name].append(round(self._safe_auroc(labels, confs), 4))
                recall_slopes[corr_name].append(round(recall_score(labels, preds, zero_division=0), 4))

        # 2. Multi-Archetype Non-Clinical Shortcut Suite
        # Evaluated on clean true-negative cases (labels == 0) to measure false positive triggers
        neg_indices = [i for i, y in enumerate(labels) if y == 0]
        shortcut_diagnostics: Dict[str, Any] = {}
        
        if neg_indices:
            neg_images = [images[i] for i in neg_indices]
            neg_orig_confs, orig_neg_preds = predict_fn(neg_images)
            n_neg = len(neg_indices)

            shortcut_fns = self.suite.get_shortcuts()
            fp_rates = {}
            conf_deltas = {}

            for name, sc_fn in shortcut_fns.items():
                d_imgs = [sc_fn(img) for img in neg_images]
                d_confs, d_preds = predict_fn(d_imgs)
                # Cases flipped from 0 to 1
                flips = sum(1 for o, d in zip(orig_neg_preds, d_preds) if o == 0 and d == 1)
                fp_rates[name] = round(flips / n_neg, 4)
                conf_deltas[name] = round(float(np.mean(np.abs(d_confs - neg_orig_confs))), 4)

            primary_fp_rate = float(np.mean(list(fp_rates.values()))) if fp_rates else 0.0
            mean_conf_delta = float(np.mean(list(conf_deltas.values()))) if conf_deltas else 0.0
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
        mean_tier3_auroc = np.mean([slopes[2] for slopes in decay_slopes.values() if len(slopes) >= 3])
        mean_tier5_auroc = np.mean([slopes[4] for slopes in decay_slopes.values() if len(slopes) >= 5])
        retention = (0.6 * mean_tier3_auroc + 0.4 * mean_tier5_auroc) / max(0.5, clean_auroc)
        
        # Distractor & shortcut penalty
        distractor_penalty = min(25.0, primary_fp_rate * 100.0 * 2.0)
        shortcut_penalty = min(15.0, shortcut_diagnostics["shortcut_vulnerability_index"] * 100.0 * 0.5)
        
        robustness_score = max(0.0, min(100.0, (retention * 100.0) - distractor_penalty - shortcut_penalty))

        # Check for catastrophic decay collapse (any tier 3 dropping below AUROC 0.70)
        contrast_failure = any(slopes[2] < 0.70 for slopes in decay_slopes.values() if len(slopes) >= 3)

        return {
            "modality": self.modality,
            "clean_auroc": round(clean_auroc, 4),
            "clean_recall": round(clean_recall, 4),
            "decay_slopes": decay_slopes,
            "recall_slopes": recall_slopes,
            "distractor_false_positive_rate": round(primary_fp_rate, 4),
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
