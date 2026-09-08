"""Generates synthetic validation cohort radiographs and structured cohort_metadata.csv.
Adheres strictly to HLT-08 Cohort Metadata Contract.
"""

import os
import cv2
import numpy as np
import pandas as pd


def generate_synthetic_chest_scan(
    is_positive: bool,
    scanner: str = "DIGITAL_RAD",
    age: int = 45,
    seed: int = 42,
) -> np.ndarray:
    """Generates a realistic 224x224 grayscale radiograph tensor simulation."""
    np.random.seed(seed)
    h, w = 224, 224
    img = np.zeros((h, w), dtype=np.float32)

    # 1. Mediastinum & Lung Fields outline
    y, x = np.ogrid[:h, :w]
    # Center thorax ellipse
    thorax_mask = ((x - w / 2) ** 2) / ((w * 0.42) ** 2) + ((y - h / 2) ** 2) / ((h * 0.45) ** 2) <= 1.0
    img[thorax_mask] = 40.0

    # Left and Right Lung cavities (darker radiolucent areas)
    left_lung = ((x - w * 0.32) ** 2) / ((w * 0.16) ** 2) + ((y - h * 0.48) ** 2) / ((h * 0.30) ** 2) <= 1.0
    right_lung = ((x - w * 0.68) ** 2) / ((w * 0.16) ** 2) + ((y - h * 0.48) ** 2) / ((h * 0.30) ** 2) <= 1.0
    img[left_lung] = 120.0
    img[right_lung] = 120.0

    # Cardiac silhouette (dense radiopaque structure in lower-left-center)
    cardiac = ((x - w * 0.42) ** 2) / ((w * 0.14) ** 2) + ((y - h * 0.58) ** 2) / ((h * 0.18) ** 2) <= 1.0
    img[cardiac] = 60.0

    # Rib cage arc patterns
    for r_idx in range(5):
        rib_y = int(h * 0.30 + r_idx * 24)
        cv2.ellipse(img, (w // 2, rib_y), (int(w * 0.36), 18), 0, 10, 170, 30.0, 3)

    # 2. Pathological Infiltrate / Consolidation (if positive)
    if is_positive:
        # Patchy opacity in right mid-to-lower zone
        cx, cy = int(w * 0.68), int(h * 0.55)
        consolidation = ((x - cx) ** 2 + (y - cy) ** 2) <= (w * 0.12) ** 2
        # Fuzzy radiopaque infiltrate
        img[consolidation] = np.maximum(20.0, img[consolidation] - 65.0)

    # 3. Scanner Protocol Characteristics
    if scanner == "COMPUTED_RAD":
        # CR plates typically exhibit lower dynamic contrast and higher sensor noise
        img = img * 0.82 + 25.0
        cr_noise = np.random.normal(0, 12, (h, w))
        img += cr_noise
    else:
        # DR plates have higher contrast and quantum clarity
        dr_noise = np.random.normal(0, 4, (h, w))
        img += dr_noise

    # Smooth anatomical transitions
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def generate_cohort(output_dir: str = "data", n_samples: int = 60) -> str:
    os.makedirs(os.path.join(output_dir, "sample_scans"), exist_ok=True)
    metadata_rows = []

    np.random.seed(2026)
    sites = ["AIIMS_DELHI", "DIST_HOSP_JAIPUR"]
    scanners = ["DIGITAL_RAD", "COMPUTED_RAD"]
    sexes = ["M", "F"]

    for i in range(1, n_samples + 1):
        patient_id = f"IND-P-{i:05d}"
        site_id = sites[0] if i <= n_samples // 2 else sites[1]
        
        # Demographic distribution
        if i % 6 == 0:
            age = int(np.random.randint(4, 17))  # Pediatric
        elif i % 5 == 0:
            age = int(np.random.randint(66, 85))  # Geriatric
        else:
            age = int(np.random.randint(21, 60))  # Adult

        sex = sexes[i % 2]
        # Scanner correlation with site (Jaipur has higher proportion of older CR units)
        if site_id == "DIST_HOSP_JAIPUR":
            scanner = "COMPUTED_RAD" if i % 3 != 0 else "DIGITAL_RAD"
        else:
            scanner = "DIGITAL_RAD" if i % 4 != 0 else "COMPUTED_RAD"

        # Balanced pathology status with realistic prevalence (~40%)
        is_positive = 1 if (i % 5 in [1, 3]) else 0

        scan_filename = f"scan_{i:04d}.png"
        rel_scan_path = f"data/sample_scans/{scan_filename}"
        abs_scan_path = os.path.join(output_dir, "sample_scans", scan_filename)

        img = generate_synthetic_chest_scan(
            is_positive=bool(is_positive),
            scanner=scanner,
            age=age,
            seed=i,
        )
        cv2.imwrite(abs_scan_path, img)

        metadata_rows.append({
            "patient_id": patient_id,
            "image_path": rel_scan_path,
            "age": age,
            "sex": sex,
            "site_id": site_id,
            "scanner_type": scanner,
            "ground_truth": is_positive,
        })

    df = pd.DataFrame(metadata_rows)
    csv_path = os.path.join(output_dir, "sample_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {n_samples} cohort scans and metadata at {csv_path}")
    return csv_path


if __name__ == "__main__":
    generate_cohort()
