"""Pluggable Clinical Modality Perturbation & Shortcut Suites.

Supports:
1. RadiologySuite (Chest X-Ray / CT): Poisson quantum noise, CR contrast attenuation, motion blur, laterality lead stamps, collimators.
2. OphthalmologySuite (Retinal Fundus): Cataract media opacity haze, non-mydriatic illumination falloff, handheld motion blur, lens dust artifact.
3. TabularEHRSuite (Clinical Labs & Vitals): Missing value MCAR/MAR injection, Gaussian sensor noise, outlier spikes.

Literature Grounding:
- DeGrave et al. (Nature Machine Intelligence 2021): Radiology visual shortcuts.
- Gulshan et al. (JAMA 2016) & DRISHYA V2: Fundus image quality degradation.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Radiology Suite (Chest Radiographs)
# ---------------------------------------------------------------------------

class RadiologySuite:
    name = "chest_xray"
    description = "Thoracic Radiography & CT Hardware Shift Battery"

    @staticmethod
    def get_corruption_ladder() -> Dict[str, Any]:
        return {
            "contrast_attenuation": RadiologySuite.apply_contrast_attenuation,
            "poisson_shot_noise": RadiologySuite.apply_poisson_noise,
            "motion_blur": RadiologySuite.apply_motion_blur,
        }

    @staticmethod
    def apply_contrast_attenuation(img: np.ndarray, severity: int) -> np.ndarray:
        factors = {1: 0.85, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25}
        factor = factors.get(severity, 0.55)
        img_f = img.astype(np.float32)
        mean_val = np.mean(img_f)
        attenuated = mean_val + factor * (img_f - mean_val)
        return np.clip(attenuated, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_poisson_noise(img: np.ndarray, severity: int, seed: Optional[int] = 42) -> np.ndarray:
        scale_factors = {1: 150.0, 2: 80.0, 3: 40.0, 4: 20.0, 5: 10.0}
        scale = scale_factors.get(severity, 40.0)
        img_norm = np.maximum(img.astype(np.float32), 1e-3) / 255.0
        photons = img_norm * scale
        rng = np.random.default_rng(seed)
        noisy_photons = rng.poisson(photons).astype(np.float32)
        noisy = (noisy_photons / scale) * 255.0
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
        kernel_sizes = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}
        k = kernel_sizes.get(severity, 11)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel).astype(np.uint8)

    @staticmethod
    def get_shortcuts() -> Dict[str, Any]:
        return {
            "laterality_stamp_R": lambda img: RadiologySuite._stamp(img, "R"),
            "laterality_stamp_L": lambda img: RadiologySuite._stamp(img, "L"),
            "hospital_text_stamp": RadiologySuite._text_banner,
            "metallic_hardware": RadiologySuite._pacemaker,
            "collimator_aperture": RadiologySuite._collimator,
        }

    @staticmethod
    def _stamp(img: np.ndarray, marker: str) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        sw, sh = max(16, int(w * 0.12)), max(16, int(h * 0.12))
        x1, y1 = max(0, w - sw - 10), 10
        x2, y2 = min(w, w - 10), 10 + sh
        out[y1:y2, x1:x2] = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, sh / 45.0)
        thick = max(2, int(font_scale * 2.5))
        ts = cv2.getTextSize(marker, font, font_scale, thick)[0]
        tx, ty = x1 + (sw - ts[0]) // 2, y1 + (sh + ts[1]) // 2
        cv2.putText(out, marker, (tx, ty), font, font_scale, 250, thick, cv2.LINE_AA)
        return out

    @staticmethod
    def _text_banner(img: np.ndarray, text: str = "PORTABLE CHEST 08:30") -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        bh = max(14, int(h * 0.08))
        out[h - bh : h, :] = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, text, (8, h - int(bh * 0.3)), font, max(0.35, bh / 50.0), 240, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _pacemaker(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.25), int(h * 0.25))
        axes = (max(8, int(w * 0.07)), max(6, int(h * 0.05)))
        cv2.ellipse(out, center, axes, 15, 0, 360, 250, -1)
        pt1 = (center[0] + axes[0], center[1])
        pt2 = (int(w * 0.5), int(h * 0.55))
        cv2.line(out, pt1, pt2, 240, max(1, int(w * 0.01)))
        return out

    @staticmethod
    def _collimator(img: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        bx, by = int(w * border_fraction), int(h * border_fraction)
        out[:by, :] = 0
        out[h - by :, :] = 0
        out[:, :bx] = 0
        out[:, w - bx :] = 0
        return out


# ---------------------------------------------------------------------------
# 2. Ophthalmology Suite (Retinal Fundus)
# ---------------------------------------------------------------------------

class OphthalmologySuite:
    name = "retinal_fundus"
    description = "Retinal Fundus Photography & Diabetic Retinopathy Stress Battery"

    @staticmethod
    def get_corruption_ladder() -> Dict[str, Any]:
        return {
            "cataract_media_haze": OphthalmologySuite.apply_cataract_haze,
            "illumination_falloff": OphthalmologySuite.apply_illumination_vignette,
            "motion_blur": OphthalmologySuite.apply_tremor_blur,
        }

    @staticmethod
    def apply_cataract_haze(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates crystalline lens opacification / nuclear cataract causing scattering and contrast loss."""
        blur_sigmas = {1: 1.0, 2: 2.5, 3: 4.5, 4: 7.0, 5: 10.0}
        sigma = blur_sigmas.get(severity, 4.5)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Haze milkiness blend
        haze_weights = {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.50, 5: 0.65}
        alpha = haze_weights.get(severity, 0.35)
        haze_overlay = np.full_like(img, 180)
        blended = cv2.addWeighted(blurred, 1.0 - alpha, haze_overlay, alpha, 0)
        return blended.astype(np.uint8)

    @staticmethod
    def apply_illumination_vignette(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates poor pupil dilation (non-mydriatic) or unaligned condenser lens peripheral light loss."""
        h, w = img.shape[:2]
        # Radial vignette mask centered on macula/fovea
        cx, cy = w / 2.0, h / 2.0
        max_r = np.sqrt(cx**2 + cy**2)
        radius_factors = {1: 0.90, 2: 0.75, 3: 0.60, 4: 0.45, 5: 0.30}
        r_cutoff = max_r * radius_factors.get(severity, 0.60)
        
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
        mask = np.clip(1.0 - (dist_from_center / max(1.0, r_cutoff)) ** 2, 0.0, 1.0)
        if img.ndim == 3:
            mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
        vignetted = (img.astype(np.float32) * mask).astype(np.uint8)
        return vignetted

    @staticmethod
    def apply_tremor_blur(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates handheld fundus camera micro-saccade tremor blur."""
        k_sizes = {1: 3, 2: 5, 3: 9, 4: 13, 5: 19}
        k = k_sizes.get(severity, 9)
        # Diagonal motion trajectory simulating hand tremor
        kernel = np.eye(k, dtype=np.float32) / k
        return cv2.filter2D(img, -1, kernel).astype(np.uint8)

    @staticmethod
    def get_shortcuts() -> Dict[str, Any]:
        return {
            "camera_dust_ring": OphthalmologySuite._lens_dust_ring,
            "macula_overflash": OphthalmologySuite._flash_artifact,
            "aperture_shutter_crop": OphthalmologySuite._circular_mask_crop,
        }

    @staticmethod
    def _lens_dust_ring(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.7), int(h * 0.3))
        cv2.circle(out, center, max(6, int(w * 0.04)), 15, 2)
        return out

    @staticmethod
    def _flash_artifact(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.45), int(h * 0.5))
        cv2.circle(out, center, max(12, int(w * 0.08)), 255, -1)
        return out

    @staticmethod
    def _circular_mask_crop(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        r = int(min(cx, cy) * 0.85)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        if img.ndim == 3:
            mask = mask[:, :, np.newaxis]
        return np.where(mask > 0, img, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Tabular & EHR Suite (Clinical Laboratory & Patient Vitals)
# ---------------------------------------------------------------------------

class TabularEHRSuite:
    name = "tabular_ehr"
    description = "Electronic Health Record & Diagnostic Laboratory Stress Battery"

    @staticmethod
    def inject_mcar_missingness(features: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
        """Injects Missing Completely at Random (MCAR) null values into feature vectors."""
        rates = {1: 0.05, 2: 0.15, 3: 0.25, 4: 0.40, 5: 0.60}
        rate = rates.get(severity, 0.25)
        rng = np.random.default_rng(seed)
        mask = rng.uniform(0, 1, size=features.shape) < rate
        out = features.copy().astype(float)
        out[mask] = np.nan
        return out

    @staticmethod
    def inject_sensor_noise(features: np.ndarray, severity: int, seed: int = 42) -> np.ndarray:
        """Simulates physiological monitor sensor jitter and drift."""
        sigmas = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50}
        sigma = sigmas.get(severity, 0.20)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, sigma, size=features.shape)
        return features + noise


def get_modality_suite(modality_name: str):
    mapping = {
        "chest_xray": RadiologySuite,
        "radiology": RadiologySuite,
        "retinal_fundus": OphthalmologySuite,
        "ophthalmology": OphthalmologySuite,
        "fundus": OphthalmologySuite,
        "tabular_ehr": TabularEHRSuite,
        "ehr": TabularEHRSuite,
    }
    suite_cls = mapping.get(modality_name.lower().strip(), RadiologySuite)
    return suite_cls()
