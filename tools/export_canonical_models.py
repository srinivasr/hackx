#!/usr/bin/env python3
"""Export canonical clinical SaMD ONNX models and generate clinical cohorts for:
1. Dermatology / Dermoscopy (Melanoma Screening)
2. Digital Histopathology (PatchCamelyon Lymph Node Metastasis)
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

os.makedirs("assets/models/dermatology", exist_ok=True)
os.makedirs("assets/models/pathology", exist_ok=True)
os.makedirs("assets/test_samples", exist_ok=True)
os.makedirs("data/sample_dermatology_scans", exist_ok=True)
os.makedirs("data/sample_histopathology_scans", exist_ok=True)

print("--- 1. Building & Exporting Dermatology Models ---")

class ClinicalClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.fc(x)

# 1a. EfficientNet-B0 Dermatology Melanoma
eff_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = eff_model.classifier[1].in_features
eff_model.classifier[1] = nn.Linear(in_features, 2)
eff_model.eval()

dummy_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
derm_eff_path = "assets/models/dermatology/derm_efficientnet_melanoma.onnx"
torch.onnx.export(
    eff_model,
    dummy_img,
    derm_eff_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    dynamo=False,
)
print(f"Exported: {derm_eff_path} ({os.path.getsize(derm_eff_path) / 1024 / 1024:.2f} MB)")

# 1b. ResNet-18 Dermatology Reference
resnet_derm = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet_derm.fc = nn.Linear(resnet_derm.fc.in_features, 2)
resnet_derm.eval()
derm_res_path = "assets/models/dermatology/derm_resnet_reference.onnx"
torch.onnx.export(
    resnet_derm,
    dummy_img,
    derm_res_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    dynamo=False,
)
print(f"Exported: {derm_res_path} ({os.path.getsize(derm_res_path) / 1024 / 1024:.2f} MB)")

print("\n--- 2. Building & Exporting Digital Histopathology Models ---")

# 2a. MobileNetV2 PatchCamelyon (WSI Sentinel Node)
mob_path = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mob_path.classifier[1] = nn.Linear(mob_path.classifier[1].in_features, 2)
mob_path.eval()
path_mob_path = "assets/models/pathology/path_mobilenet_pcam.onnx"
torch.onnx.export(
    mob_path,
    dummy_img,
    path_mob_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    dynamo=False,
)
print(f"Exported: {path_mob_path} ({os.path.getsize(path_mob_path) / 1024 / 1024:.2f} MB)")

# 2b. DenseNet-121 Histopathology Reference
dense_path = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
dense_path.classifier = nn.Linear(dense_path.classifier.in_features, 2)
dense_path.eval()
path_dense_path = "assets/models/pathology/path_densenet_wsi.onnx"
torch.onnx.export(
    dense_path,
    dummy_img,
    path_dense_path,
    input_names=["input"],
    output_names=["logits"],
    opset_version=14,
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    dynamo=False,
)
print(f"Exported: {path_dense_path} ({os.path.getsize(path_dense_path) / 1024 / 1024:.2f} MB)")

print("\n--- 3. Generating Realistic Clinical Synthetic Cohorts ---")

# Generator for synthetic Dermoscopy lesion images
def generate_synthetic_dermoscopy(is_malignant: bool, phototype: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 384, 384
    # Skin tone base according to Fitzpatrick phototype
    if "I_II" in phototype: # Fair
        base_color = np.array([190, 210, 240], dtype=np.float32) # BGR
    elif "III_IV" in phototype: # Medium
        base_color = np.array([140, 175, 215], dtype=np.float32)
    else: # Dark V-VI
        base_color = np.array([75, 105, 140], dtype=np.float32)
    
    img = np.ones((h, w, 3), dtype=np.float32) * base_color
    # Subtle skin texture & pores
    noise = rng.normal(0, 7, (h, w, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    # Lesion center and pigment blob
    cx, cy = w // 2 + rng.integers(-20, 20), h // 2 + rng.integers(-20, 20)
    rx, ry = rng.integers(70, 110), rng.integers(60, 100)
    
    # Pigmentation color: Malignant has variegated dark brown / black / reddish hues
    if is_malignant:
        lesion_bgr = np.array([30, 45, 75], dtype=np.float32) # Dark irregular brown
        angle = rng.integers(0, 180)
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, tuple(lesion_bgr.tolist()), -1)
        # Add irregular pigment asymmetry & satellite globules
        for _ in range(rng.integers(4, 8)):
            gx = cx + rng.integers(-rx, rx)
            gy = cy + rng.integers(-ry, ry)
            gr = rng.integers(12, 35)
            dark_tone = np.array([15, 20, 35], dtype=np.float32)
            cv2.circle(img, (gx, gy), gr, tuple(dark_tone.tolist()), -1)
    else:
        # Benign is symmetric, homogeneous light brown
        lesion_bgr = np.array([60, 95, 145], dtype=np.float32)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, tuple(lesion_bgr.tolist()), -1)

    # Smooth the pigment network borders
    img = cv2.GaussianBlur(img, (21, 21), 9)

    # Vignetting (dermatoscope cylinder rim)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
    max_radius = w * 0.48
    vignette = np.clip(1.0 - (dist / max_radius)**3, 0.2, 1.0)
    img = img * vignette[:, :, None]

    return np.clip(img, 0, 255).astype(np.uint8)

# Generator for synthetic Histopathology WSI images
def generate_synthetic_histology(is_metastatic: bool, vendor: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 384, 384

    # Base background: Eosinophilic pink / stroma
    if vendor == "VENTANA_ROCHE":
        # Deeper crimson / magenta tone
        stroma_color = np.array([180, 160, 225], dtype=np.float32) # BGR
        nuclei_color = np.array([140, 50, 90], dtype=np.float32) # Deep violet
    else:
        # Leica: classic lilac / pink
        stroma_color = np.array([195, 175, 235], dtype=np.float32)
        nuclei_color = np.array([120, 40, 80], dtype=np.float32)

    img = np.ones((h, w, 3), dtype=np.float32) * stroma_color
    noise = rng.normal(0, 5, (h, w, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    # Connective tissue fibers
    for _ in range(rng.integers(15, 25)):
        pt1 = (rng.integers(0, w), rng.integers(0, h))
        pt2 = (rng.integers(0, w), rng.integers(0, h))
        fiber_color = stroma_color * 0.9
        cv2.line(img, pt1, pt2, tuple(fiber_color.tolist()), rng.integers(1, 4))

    # Cell nuclei (lymphocytes vs metastatic carcinoma clusters)
    num_nuclei = rng.integers(180, 320) if is_metastatic else rng.integers(90, 150)
    for _ in range(num_nuclei):
        nx = rng.integers(10, w - 10)
        ny = rng.integers(10, h - 10)
        # Metastatic cells have pleomorphic, enlarged, hyperchromatic nuclei
        nr = rng.integers(5, 10) if is_metastatic else rng.integers(2, 5)
        cv2.circle(img, (nx, ny), nr, tuple(nuclei_color.tolist()), -1)

    # Blur slightly for microscope focal plane
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return np.clip(img, 0, 255).astype(np.uint8)

# 3a. Generate 60 Dermatology Scans & CSV
derm_records = []
sites = ["SYDNEY_MELANOMA_CENTRE", "BARCELONA_DERM_CLINIC"]
phototypes = ["TYPE_I_II_FAIR", "TYPE_III_IV_MEDIUM", "TYPE_V_VI_DARK"]
anatomies = ["TORSO", "LOWER_EXTREMITY", "HEAD_NECK"]

for i in range(1, 61):
    pid = f"DERM-MEL-{i:04d}"
    filename = f"lesion_{i:04d}.png"
    rel_path = f"data/sample_dermatology_scans/{filename}"
    abs_path = os.path.join("data/sample_dermatology_scans", filename)
    
    # 40% malignancy prevalence
    is_mal = 1 if (i % 5 in [1, 3]) else 0
    pt = phototypes[i % len(phototypes)]
    site = sites[i % len(sites)]
    anat = anatomies[i % len(anatomies)]
    age = int(22 + (i * 1.05) % 62)
    sex = "F" if (i % 2 == 0) else "M"

    img = generate_synthetic_dermoscopy(bool(is_mal), pt, seed=i * 7)
    cv2.imwrite(abs_path, img)

    derm_records.append({
        "patient_id": pid,
        "image_path": rel_path,
        "age": age,
        "sex": sex,
        "site_id": site,
        "fitzpatrick_skin_type": pt,
        "anatomical_site": anat,
        "ground_truth": is_mal,
    })

df_derm = pd.DataFrame(derm_records)
df_derm.to_csv("data/sample_dermatology_metadata.csv", index=False)
print("Saved: data/sample_dermatology_metadata.csv (60 patients)")

# 3b. Generate 60 Histopathology Scans & CSV
path_records = []
path_sites = ["RADBOUD_PATHOLOGY_LAB", "UTRECHT_BIOBANK"]
vendors = ["VENTANA_ROCHE", "LEICA_BIOSYSTEMS"]
tissues = ["AXILLARY_LYMPH_NODE", "SENTINEL_NODE"]

for i in range(1, 61):
    pid = f"PATH-WSI-{i:04d}"
    filename = f"wsi_{i:04d}.png"
    rel_path = f"data/sample_histopathology_scans/{filename}"
    abs_path = os.path.join("data/sample_histopathology_scans", filename)

    is_meta = 1 if (i % 5 in [0, 2]) else 0
    vendor = vendors[i % len(vendors)]
    site = path_sites[i % len(path_sites)]
    tiss = tissues[i % len(tissues)]
    age = int(35 + (i * 0.95) % 45)
    sex = "F" if (i % 3 != 0) else "M"

    img = generate_synthetic_histology(bool(is_meta), vendor, seed=i * 11)
    cv2.imwrite(abs_path, img)

    path_records.append({
        "patient_id": pid,
        "image_path": rel_path,
        "age": age,
        "sex": sex,
        "site_id": site,
        "stain_vendor": vendor,
        "tissue_source": tiss,
        "ground_truth": is_meta,
    })

df_path = pd.DataFrame(path_records)
df_path.to_csv("data/sample_histopathology_metadata.csv", index=False)
print("Saved: data/sample_histopathology_metadata.csv (60 patients)")

# Save sample images to assets/test_samples for Optical Stress Studio preview
cv2.imwrite("assets/test_samples/derm_melanoma_sample.jpg", generate_synthetic_dermoscopy(True, "TYPE_I_II_FAIR", seed=42))
cv2.imwrite("assets/test_samples/path_wsi_sample.jpg", generate_synthetic_histology(True, "LEICA_BIOSYSTEMS", seed=101))
print("Saved preview test samples in assets/test_samples/")
print("All canonical models and cohorts generated successfully!")
