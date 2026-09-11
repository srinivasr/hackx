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



# ---------------------------------------------------------------------------
# 4. Clinical NLP Suite (EHR Clinical Notes, Discharge Summaries & MedNLI)
# ---------------------------------------------------------------------------

class ClinicalTextSuite:
    name = "clinical_nlp"
    description = "EHR Clinical Notes, Discharge Summaries & Clinical Diagnostic NLP Stress Battery"

    ABBREVIATIONS = {
        "shortness of breath": "SOB",
        "chest pain": "CP",
        "hypertension": "HTN",
        "type 2 diabetes": "T2DM",
        "diabetes mellitus": "DM",
        "congestive heart failure": "CHF",
        "myocardial infarction": "MI",
        "chronic obstructive pulmonary disease": "COPD",
        "history of": "h/o",
        "complains of": "c/o",
        "intensive care unit": "ICU",
        "emergency department": "ED",
        "vital signs stable": "VSS",
        "blood pressure": "BP",
        "heart rate": "HR",
        "respiratory rate": "RR",
        "treatment": "tx",
        "diagnosis": "dx",
        "prescription": "rx",
        "patient": "pt",
    }

    OCR_CHAR_MAP = {
        "l": "1", "1": "l",
        "O": "0", "0": "O",
        "m": "rn",
        "d": "cl",
        "w": "vv",
        "e": "c",
        "s": "5", "5": "s",
    }

    @staticmethod
    def get_corruption_ladder() -> Dict[str, Any]:
        return {
            "ocr_typographical_noise": ClinicalTextSuite.apply_ocr_typos,
            "abbreviation_density": ClinicalTextSuite.apply_abbreviations,
            "note_truncation": ClinicalTextSuite.apply_note_truncation,
        }

    @staticmethod
    def apply_ocr_typos(text: str, severity: int, seed: Optional[int] = 42) -> str:
        """Simulates OCR scanner character degradation and typographical noise in EHR physician notes."""
        if not isinstance(text, str) or len(text) == 0:
            return text
        rates = {1: 0.04, 2: 0.09, 3: 0.16, 4: 0.28, 5: 0.42}
        rate = rates.get(severity, 0.16)
        rng = np.random.default_rng(seed)
        chars = list(text)
        for i in range(len(chars)):
            if rng.uniform(0, 1) < rate:
                c = chars[i]
                if c in ClinicalTextSuite.OCR_CHAR_MAP:
                    chars[i] = ClinicalTextSuite.OCR_CHAR_MAP[c]
                elif c.isalpha() and rng.uniform(0, 1) < 0.4:
                    chars[i] = chr(ord('a') + rng.integers(0, 26)) if c.islower() else chr(ord('A') + rng.integers(0, 26))
        return "".join(chars)

    @staticmethod
    def apply_abbreviations(text: str, severity: int) -> str:
        """Contracts clinical terminology into dense physician shorthand."""
        if not isinstance(text, str):
            return text
        out = text
        import re
        for term, abbr in ClinicalTextSuite.ABBREVIATIONS.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(out):
                if (severity / 5.0) >= 0.3:
                    out = pattern.sub(abbr, out)
        return out

    @staticmethod
    def apply_note_truncation(text: str, severity: int) -> str:
        """Simulates incomplete or hasty clinical documentation cut-off."""
        if not isinstance(text, str):
            return text
        retention_factors = {1: 0.90, 2: 0.75, 3: 0.55, 4: 0.35, 5: 0.20}
        factor = retention_factors.get(severity, 0.55)
        words = text.split()
        if len(words) <= 3:
            return text
        cut = max(3, int(len(words) * factor))
        return " ".join(words[:cut])

    @staticmethod
    def get_shortcuts() -> Dict[str, Any]:
        return {
            "negation_inversion": ClinicalTextSuite.inject_negation_inversion,
            "demographic_pronoun_swap": ClinicalTextSuite.swap_demographic_markers,
            "clerical_banner_stamp": ClinicalTextSuite.inject_clerical_banner,
        }

    @staticmethod
    def inject_negation_inversion(text: str) -> str:
        """Inverts medical negation phrases to stress-test clinical NegEx comprehension."""
        if not isinstance(text, str):
            return text
        import re
        neg_pairs = [
            (r"\bdenies\b", "reports"),
            (r"\bdenied\b", "reported"),
            (r"\bno acute\b", "acute"),
            (r"\bnegative for\b", "positive for"),
            (r"\babsent\b", "present"),
            (r"\bwithout\b", "with"),
            (r"\bcleared for\b", "critical risk for"),
        ]
        out = text
        for pat, rep in neg_pairs:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        return out

    @staticmethod
    def swap_demographic_markers(text: str) -> str:
        """Swaps gender pronouns and demographic mentions to test for social bias leakage."""
        if not isinstance(text, str):
            return text
        import re
        swaps = [
            (r"\bhe\b", "she"), (r"\bHe\b", "She"),
            (r"\bhis\b", "her"), (r"\bHis\b", "Her"),
            (r"\bhim\b", "her"), (r"\bhimself\b", "herself"),
            (r"\bmale\b", "female"), (r"\bMale\b", "Female"),
            (r"\bman\b", "woman"), (r"\bgentleman\b", "lady"),
        ]
        out = text
        for pat, rep in swaps:
            out = re.sub(pat, rep, out)
        return out

    @staticmethod
    def inject_clerical_banner(text: str) -> str:
        """Appends clerical administrative metadata to test for false-positive administrative shortcut triggers."""
        if not isinstance(text, str):
            return text
        return f"[EHR CLERICAL RECORD #9824 - BETH ISRAEL ICUR TRANSCRIPTION COMPLETE] {text}"


# ---------------------------------------------------------------------------
# 4. Dermatology Suite (Dermoscopy & Melanoma Screening)
# ---------------------------------------------------------------------------

class DermatologySuite:
    name = "dermatology_dermoscopy"
    description = "Dermoscopy & Point-of-Care Melanoma Screening Stress Battery"

    @staticmethod
    def get_corruption_ladder() -> Dict[str, Any]:
        return {
            "specular_glare": DermatologySuite.apply_specular_glare,
            "peripheral_vignetting": DermatologySuite.apply_peripheral_vignetting,
            "handheld_defocus": DermatologySuite.apply_handheld_defocus,
        }

    @staticmethod
    def apply_specular_glare(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates immersion fluid flash glare reflections on skin surface."""
        glare_intensities = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.75, 5: 0.90}
        intensity = glare_intensities.get(severity, 0.60)
        out = img.copy().astype(np.float32)
        h, w = out.shape[:2]
        center = (int(w * 0.45), int(h * 0.45))
        radius = int(min(h, w) * (0.12 + 0.05 * severity))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, center, radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        glare = np.ones_like(out) * 255.0
        if out.ndim == 3:
            out = out * (1.0 - mask[:, :, None] * intensity) + glare * (mask[:, :, None] * intensity)
        else:
            out = out * (1.0 - mask * intensity) + glare * (mask * intensity)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_peripheral_vignetting(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates dermatoscope optical barrel illumination falloff."""
        vignette_factors = {1: 0.80, 2: 0.65, 3: 0.50, 4: 0.35, 5: 0.20}
        factor = vignette_factors.get(severity, 0.50)
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        falloff = 1.0 - (dist / max_dist) * (1.0 - factor)
        falloff = np.clip(falloff, factor, 1.0)
        if img.ndim == 3:
            out = img.astype(np.float32) * falloff[:, :, None]
        else:
            out = img.astype(np.float32) * falloff
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_handheld_defocus(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates operator hand motion and patient tremor blur."""
        k_sizes = {1: 3, 2: 7, 3: 11, 4: 17, 5: 23}
        k = k_sizes.get(severity, 11)
        return cv2.GaussianBlur(img, (k, k), k / 3.0)

    @staticmethod
    def get_shortcuts() -> Dict[str, Any]:
        return {
            "surgical_skin_marker": DermatologySuite._surgical_marker,
            "measurement_ruler": DermatologySuite._measurement_ruler,
            "gel_air_bubble": DermatologySuite._gel_bubble,
        }

    @staticmethod
    def _surgical_marker(img: np.ndarray) -> np.ndarray:
        """Injects surgical gentian violet pen line near lesion (Nature Med 2020 shortcut)."""
        out = img.copy()
        h, w = out.shape[:2]
        # Draw blue/violet surgical incision line
        violet_bgr = (180, 40, 90)
        pt1 = (int(w * 0.15), int(h * 0.80))
        pt2 = (int(w * 0.70), int(h * 0.90))
        cv2.line(out, pt1, pt2, violet_bgr, max(2, int(w * 0.015)), cv2.LINE_AA)
        return out

    @staticmethod
    def _measurement_ruler(img: np.ndarray) -> np.ndarray:
        """Injects millimeter calibration ruler ticks along image edge."""
        out = img.copy()
        h, w = out.shape[:2]
        ruler_y = h - 20
        cv2.line(out, (20, ruler_y), (w - 20, ruler_y), (240, 240, 240), 2)
        for x in range(20, w - 20, 15):
            tick_h = 8 if (x % 30 == 0) else 4
            cv2.line(out, (x, ruler_y), (x, ruler_y - tick_h), (240, 240, 240), 1)
        return out

    @staticmethod
    def _gel_bubble(img: np.ndarray) -> np.ndarray:
        """Injects immersion ultrasound/dermatoscope air bubble artifact."""
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.75), int(h * 0.25))
        radius = int(min(h, w) * 0.08)
        cv2.circle(out, center, radius, (20, 20, 20), 2)
        cv2.circle(out, (center[0] - 2, center[1] - 2), int(radius * 0.3), (250, 250, 250), -1)
        return out


# ---------------------------------------------------------------------------
# 5. Digital Histopathology Suite (Whole Slide Imaging WSI / PatchCamelyon)
# ---------------------------------------------------------------------------

class HistopathologySuite:
    name = "digital_pathology"
    description = "Whole-Slide Histopathology & Lymph Node Metastasis Stress Battery"

    @staticmethod
    def get_corruption_ladder() -> Dict[str, Any]:
        return {
            "he_stain_variability": HistopathologySuite.apply_he_stain_variability,
            "wsi_defocus": HistopathologySuite.apply_wsi_defocus,
            "microtome_compression": HistopathologySuite.apply_microtome_compression,
        }

    @staticmethod
    def apply_he_stain_variability(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates Hematoxylin & Eosin chemical pH batch stain shift."""
        stain_shifts = {
            1: (1.05, 0.95, 1.05), # Subtle eosin shift
            2: (1.12, 0.90, 1.10),
            3: (1.22, 0.85, 1.18), # Moderate pH variation
            4: (1.35, 0.75, 1.25),
            5: (1.50, 0.65, 1.35), # Severe over-staining
        }
        b_mul, g_mul, r_mul = stain_shifts.get(severity, (1.22, 0.85, 1.18))
        if img.ndim == 3 and img.shape[2] == 3:
            out = img.astype(np.float32)
            out[:, :, 0] *= b_mul
            out[:, :, 1] *= g_mul
            out[:, :, 2] *= r_mul
            return np.clip(out, 0, 255).astype(np.uint8)
        else:
            avg_mult = float(r_mul * 0.5 + b_mul * 0.5)
            return np.clip(img.astype(np.float32) * avg_mult, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_wsi_defocus(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates whole slide scanner automated stage focus drift."""
        k_sizes = {1: 3, 2: 7, 3: 11, 4: 15, 5: 21}
        k = k_sizes.get(severity, 11)
        return cv2.GaussianBlur(img, (k, k), k / 3.2)

    @staticmethod
    def apply_microtome_compression(img: np.ndarray, severity: int) -> np.ndarray:
        """Simulates microtome tissue section folding and compression wrinkles."""
        out = img.copy()
        h, w = out.shape[:2]
        num_folds = severity
        for i in range(num_folds):
            y = int(h * (0.2 + 0.15 * i))
            fold_h = max(2, int(h * 0.02))
            # Dark compression line
            out[y : y + fold_h, :] = (out[y : y + fold_h, :].astype(np.float32) * 0.6).astype(np.uint8)
        return out

    @staticmethod
    def get_shortcuts() -> Dict[str, Any]:
        return {
            "glass_slide_bubble": HistopathologySuite._slide_bubble,
            "pathologist_grease_pen": HistopathologySuite._grease_pen,
            "coverslip_edge": HistopathologySuite._coverslip_edge,
        }

    @staticmethod
    def _slide_bubble(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.25), int(h * 0.75))
        radius = int(min(h, w) * 0.09)
        cv2.circle(out, center, radius, (40, 40, 40), 2)
        return out

    @staticmethod
    def _grease_pen(img: np.ndarray) -> np.ndarray:
        """Pathologist green ink boundary circle on glass slide."""
        out = img.copy()
        h, w = out.shape[:2]
        center = (int(w * 0.5), int(h * 0.5))
        axes = (int(w * 0.42), int(h * 0.42))
        cv2.ellipse(out, center, axes, 0, 0, 180, (40, 180, 40), max(2, int(w * 0.012)))
        return out

    @staticmethod
    def _coverslip_edge(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        edge_x = int(w * 0.88)
        cv2.line(out, (edge_x, 0), (edge_x, h), (180, 180, 180), 2)
        cv2.line(out, (edge_x + 1, 0), (edge_x + 1, h), (40, 40, 40), 1)
        return out


def get_modality_suite(modality_name: str):
    mapping = {
        "chest_xray": RadiologySuite,
        "radiology": RadiologySuite,
        "retinal_fundus": OphthalmologySuite,
        "ophthalmology": OphthalmologySuite,
        "fundus": OphthalmologySuite,
        "dermatology": DermatologySuite,
        "dermatology_dermoscopy": DermatologySuite,
        "dermoscopy": DermatologySuite,
        "skin": DermatologySuite,
        "digital_pathology": HistopathologySuite,
        "histopathology": HistopathologySuite,
        "pathology": HistopathologySuite,
        "wsi": HistopathologySuite,
        "tabular_ehr": TabularEHRSuite,
        "ehr": TabularEHRSuite,
        "clinical_nlp": ClinicalTextSuite,
        "clinical_text": ClinicalTextSuite,
        "nlp": ClinicalTextSuite,
        "text": ClinicalTextSuite,
    }
    suite_cls = mapping.get(modality_name.lower().strip(), RadiologySuite)
    return suite_cls()
