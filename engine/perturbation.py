from typing import Dict, List, Tuple, Any
import cv2
import numpy as np


def apply_gaussian_blur(image_bgr: np.ndarray, sigma: float) -> np.ndarray:
    """Simulates camera defocus, media opacity, and patient eye micro-movement."""
    if sigma <= 0.0:
        return image_bgr.copy()
    ksize = int(2 * np.ceil(2 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_illumination_attenuation(image_bgr: np.ndarray, factor: float) -> np.ndarray:
    """Simulates un-dilated pupil capture, low LED battery power, or dense cataract haze."""
    factor = float(np.clip(factor, 0.05, 1.0))
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_corneal_glare(image_bgr: np.ndarray, intensity: float, radius: int = 45) -> np.ndarray:
    """Simulates corneal specular flash artifact, whiteout over pupil aperture."""
    out = image_bgr.copy().astype(np.float32)
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    
    y_indices, x_indices = np.ogrid[:h, :w]
    dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
    falloff = np.exp(-dist_sq / (2 * float(radius) ** 2))
    
    glare = (255.0 * intensity * falloff)[:, :, np.newaxis]
    out = np.clip(out + glare, 0.0, 255.0).astype(np.uint8)
    return out


def apply_sensor_noise(image_bgr: np.ndarray, variance: float) -> np.ndarray:
    """Simulates CMOS analog amplifier noise from handheld battery cameras."""
    if variance <= 0.0:
        return image_bgr.copy()
    noise = np.random.normal(0, np.sqrt(variance) * 25.5, image_bgr.shape)
    noisy = image_bgr.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def apply_resolution_scaling(image_bgr: np.ndarray, target_dim: int) -> np.ndarray:
    """Downsamples image to simulate lower resolution field sensors, then restores input size."""
    h, w = image_bgr.shape[:2]
    if target_dim >= min(h, w):
        return image_bgr.copy()
    downscaled = cv2.resize(image_bgr, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_adversarial_noise(image_bgr: np.ndarray, epsilon: float) -> np.ndarray:
    """Simulates FGSM adversarial attack with structured high-frequency imperceptible noise."""
    if epsilon <= 0.0:
        return image_bgr.copy()
    # Simulate gradient noise direction (sign)
    np.random.seed(42) # Deterministic for audit reproducibility
    sign_data = np.sign(np.random.randn(*image_bgr.shape))
    adversarial = image_bgr.astype(np.float32) + epsilon * 255.0 * sign_data
    return np.clip(adversarial, 0.0, 255.0).astype(np.uint8)


def apply_synthetic_artifact(image_bgr: np.ndarray, intensity: float) -> np.ndarray:
    """Generative simulation of sensor dropout (dead pixels) or MRI ghosting artifact."""
    if intensity <= 0.0:
        return image_bgr.copy()
    out = image_bgr.copy()
    h, w = out.shape[:2]
    # Simulate dead sensor lines/dropout
    num_lines = int(intensity * h * 0.1)
    np.random.seed(42)
    for _ in range(num_lines):
        y = np.random.randint(0, h)
        out[y, :] = 0
    return out


def generate_stress_ladder(
    image_bgr: np.ndarray, stress_type: str, steps: int = 5
) -> List[Dict[str, Any]]:
    """Generates a ladder of perturbed test images with increasing clinical severity."""
    ladder = []
    
    if stress_type == "blur":
        sigmas = np.linspace(0.0, 6.0, steps)
        for s in sigmas:
            perturbed = apply_gaussian_blur(image_bgr, float(s))
            ladder.append({
                "type": "blur",
                "param_name": "sigma",
                "param_value": round(float(s), 2),
                "label": f"Defocus Blur (σ={s:.1f})",
                "image": perturbed,
            })
    elif stress_type == "illumination":
        factors = np.linspace(1.0, 0.2, steps)
        for f in factors:
            perturbed = apply_illumination_attenuation(image_bgr, float(f))
            drop_pct = round((1.0 - float(f)) * 100, 1)
            ladder.append({
                "type": "illumination",
                "param_name": "drop_pct",
                "param_value": drop_pct,
                "label": f"Flash Drop (-{drop_pct}%)",
                "image": perturbed,
            })
    elif stress_type == "glare":
        intensities = np.linspace(0.0, 0.9, steps)
        for i in intensities:
            perturbed = apply_corneal_glare(image_bgr, float(i))
            ladder.append({
                "type": "glare",
                "param_name": "intensity",
                "param_value": round(float(i), 2),
                "label": f"Corneal Glare (int={i:.2f})",
                "image": perturbed,
            })
    elif stress_type == "resolution":
        dims = np.linspace(image_bgr.shape[0], 96, steps, dtype=int)
        for d in dims:
            perturbed = apply_resolution_scaling(image_bgr, int(d))
            ladder.append({
                "type": "resolution",
                "param_name": "dim_px",
                "param_value": int(d),
                "label": f"Sensor Resolution ({d}x{d}px)",
                "image": perturbed,
            })
    elif stress_type == "adversarial":
        epsilons = np.linspace(0.0, 0.1, steps)
        for e in epsilons:
            perturbed = apply_adversarial_noise(image_bgr, float(e))
            ladder.append({
                "type": "adversarial",
                "param_name": "epsilon",
                "param_value": round(float(e), 3),
                "label": f"FGSM Noise (eps={e:.3f})",
                "image": perturbed,
            })
    elif stress_type == "artifact":
        intensities = np.linspace(0.0, 1.0, steps)
        for i in intensities:
            perturbed = apply_synthetic_artifact(image_bgr, float(i))
            ladder.append({
                "type": "artifact",
                "param_name": "intensity",
                "param_value": round(float(i), 2),
                "label": f"Sensor Dropout (int={i:.2f})",
                "image": perturbed,
            })
    else:
        raise ValueError(f"Unknown stress type: {stress_type}")
        
    return ladder
