from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


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


def apply_pgd_noise(image_bgr: np.ndarray, epsilon: float, alpha: float, iterations: int = 10) -> np.ndarray:
    """Simulates Projected Gradient Descent (PGD) iterative adversarial attack."""
    if epsilon <= 0.0:
        return image_bgr.copy()
        
    np.random.seed(42)
    perturbed = image_bgr.astype(np.float32)
    clean = image_bgr.astype(np.float32)
    
    for _ in range(iterations):
        # Simulate local gradient sign at current iteration
        sign_data = np.sign(np.random.randn(*image_bgr.shape))
        perturbed = perturbed + alpha * 255.0 * sign_data
        
        # Project back to L_inf epsilon ball
        delta = np.clip(perturbed - clean, -epsilon * 255.0, epsilon * 255.0)
        perturbed = np.clip(clean + delta, 0.0, 255.0)
        
    return perturbed.astype(np.uint8)


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


def apply_spatial_occlusion(image_bgr: np.ndarray, occlusion_fraction: float) -> np.ndarray:
    """Random spatial cutout/occlusion testing for missing input regions."""
    if occlusion_fraction <= 0.0:
        return image_bgr.copy()
    
    out = image_bgr.copy()
    h, w = out.shape[:2]
    area = h * w
    target_occlusion_area = area * occlusion_fraction
    
    np.random.seed(42)
    occluded_area = 0
    
    while occluded_area < target_occlusion_area:
        # Random size between 5% and 20% of image dimensions
        box_w = np.random.randint(max(1, int(w * 0.05)), max(2, int(w * 0.20)))
        box_h = np.random.randint(max(1, int(h * 0.05)), max(2, int(h * 0.20)))
        
        x = np.random.randint(0, max(1, w - box_w))
        y = np.random.randint(0, max(1, h - box_h))
        
        out[y:y+box_h, x:x+box_w] = 0
        occluded_area += (box_w * box_h)
        
    return out


def generate_stress_ladder(
    image_bgr: np.ndarray, stress_type: str, steps: int = 5
) -> List[Dict[str, Any]]:
    """Generates a ladder of perturbed test images with increasing clinical severity."""
    ladder = []
    
    def _add_rung(ladder_list, type_str, p_name, p_value, label_str, perturbed_img):
        # Calculate Structural Similarity Index (SSIM)
        # Convert BGR to Grayscale for standard SSIM calculation if needed, or use multichannel
        score = ssim(image_bgr, perturbed_img, channel_axis=-1, data_range=255)
        
        ladder_list.append({
            "type": type_str,
            "param_name": p_name,
            "param_value": p_value,
            "label": label_str,
            "image": perturbed_img,
            "ssim_score": round(float(score), 3)
        })
    
    if stress_type == "blur":
        sigmas = np.linspace(0.0, 6.0, steps)
        for s in sigmas:
            perturbed = apply_gaussian_blur(image_bgr, float(s))
            _add_rung(ladder, "blur", "sigma", round(float(s), 2), f"Defocus Blur (σ={s:.1f})", perturbed)
    elif stress_type == "illumination":
        factors = np.linspace(1.0, 0.2, steps)
        for f in factors:
            perturbed = apply_illumination_attenuation(image_bgr, float(f))
            drop_pct = round((1.0 - float(f)) * 100, 1)
            _add_rung(ladder, "illumination", "drop_pct", drop_pct, f"Flash Drop (-{drop_pct}%)", perturbed)
    elif stress_type == "glare":
        intensities = np.linspace(0.0, 0.9, steps)
        for i in intensities:
            perturbed = apply_corneal_glare(image_bgr, float(i))
            _add_rung(ladder, "glare", "intensity", round(float(i), 2), f"Corneal Glare (int={i:.2f})", perturbed)
    elif stress_type == "resolution":
        dims = np.linspace(image_bgr.shape[0], 96, steps, dtype=int)
        for d in dims:
            perturbed = apply_resolution_scaling(image_bgr, int(d))
            _add_rung(ladder, "resolution", "dim_px", int(d), f"Sensor Resolution ({d}x{d}px)", perturbed)
    elif stress_type == "adversarial":
        epsilons = np.linspace(0.0, 0.1, steps)
        for e in epsilons:
            perturbed = apply_adversarial_noise(image_bgr, float(e))
            _add_rung(ladder, "adversarial", "epsilon", round(float(e), 3), f"FGSM Noise (eps={e:.3f})", perturbed)
    elif stress_type == "adversarial_pgd":
        epsilons = np.linspace(0.0, 0.1, steps)
        for e in epsilons:
            alpha = float(e) / 4.0 if float(e) > 0 else 0.0
            perturbed = apply_pgd_noise(image_bgr, float(e), alpha=alpha, iterations=10)
            _add_rung(ladder, "adversarial_pgd", "epsilon", round(float(e), 3), f"PGD Noise (eps={e:.3f})", perturbed)
    elif stress_type == "artifact":
        intensities = np.linspace(0.0, 1.0, steps)
        for i in intensities:
            perturbed = apply_synthetic_artifact(image_bgr, float(i))
            _add_rung(ladder, "artifact", "intensity", round(float(i), 2), f"Sensor Dropout (int={i:.2f})", perturbed)
    elif stress_type == "occlusion":
        fractions = np.linspace(0.0, 0.4, steps)
        for f in fractions:
            perturbed = apply_spatial_occlusion(image_bgr, float(f))
            _add_rung(ladder, "occlusion", "occlusion_fraction", round(float(f), 2), f"Occlusion/Cutout (frac={f:.2f})", perturbed)
    else:
        raise ValueError(f"Unknown stress type: {stress_type}")
        
    return ladder
