import os
import glob
import base64
import json
from typing import Dict, Any, Optional
import time
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from audit_runner import load_config
from engine.auditor import CandidateModelEvaluator, run_full_model_audit
from engine.perturbation import (
    apply_gaussian_blur,
    apply_illumination_attenuation,
    apply_corneal_glare,
    apply_resolution_scaling,
)
from backend.certificate_gen import generate_deployment_certificate

app = FastAPI(
    title="TrustCheck AI Safety Evaluation API",
    description="Automated clinical AI stress-testing and pre-deployment certification platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

MODEL_ALIASES = {
    "candidate_a_edge": "dr_lcnet_edge",
    "candidate_b_teacher": "dr_resnet_teacher",
    "chexnet_cxr": "cxr_chexnet",
}

# In-memory registry and evaluator cache
MODEL_REGISTRY = {
    "dr_lcnet_edge": {
        "id": "dr_lcnet_edge",
        "name": "Diabetic Retinopathy - Mobile LCNet (Edge)",
        "path": "assets/models/dr_retinal_lcnet_edge.onnx",
        "architecture": "PP-LCNet + MSAG (7.6M Params)",
        "modality": "retinal_fundus",
        "target_deployment": "Rural PHC Portable Fundus Camera",
    },
    "dr_resnet_teacher": {
        "id": "dr_resnet_teacher",
        "name": "Diabetic Retinopathy - ResNet Teacher (Server)",
        "path": "assets/models/dr_retinal_resnet_teacher.onnx",
        "architecture": "ResNet18-DeepEnsemble (11.2M Params)",
        "modality": "retinal_fundus",
        "target_deployment": "District Hospital GPU Server",
    },
    "cxr_chexnet": {
        "id": "cxr_chexnet",
        "name": "Chest Radiography - CheXNet DenseNet121 (Hospital Grade)",
        "path": "assets/models/cxr_chexnet_densenet121.onnx",
        "architecture": "DenseNet-121 Clinical Benchmark (7.0M Params)",
        "modality": "chest_xray",
        "target_deployment": "Radiology Department Workstation",
    },
    "cxr_mobilenet_edge": {
        "id": "cxr_mobilenet_edge",
        "name": "Chest Radiography - MobileNetV2 (Bedside Cart Edge)",
        "path": "assets/models/cxr_mobilenet_edge.onnx",
        "architecture": "MobileNetV2 Depthwise-Conv (3.5M Params)",
        "modality": "chest_xray",
        "target_deployment": "Portable Bedside ICU X-Ray Cart",
    },
}

DATASET_REGISTRY = {
    "retinal_dr_60": {
        "id": "retinal_dr_60",
        "name": "Retinal Fundus DR Cohort",
        "modality": "retinal_fundus",
        "path": "data/sample_retinal_metadata.csv",
        "sample_count": 60,
        "sites": ["Sankara Nethralaya", "Aravind Eye Hospital"],
        "description": "60 multi-field fundus scans across Topcon TRC-50DX and Remidio FOP portable cameras",
        "compatible_models": ["dr_lcnet_edge", "dr_resnet_teacher"],
        "default_output_dir": "outputs/audit_run_retinal_dr",
    },
    "chest_xray_60": {
        "id": "chest_xray_60",
        "name": "Chest Radiography CXR Cohort",
        "modality": "chest_xray",
        "path": "data/sample_metadata.csv",
        "sample_count": 60,
        "sites": ["AIIMS Delhi", "District Hospital Jaipur"],
        "description": "60 PA chest radiographs across Siemens Digital and GE Computed Radiography scanners",
        "compatible_models": ["cxr_chexnet", "cxr_mobilenet_edge"],
        "default_output_dir": "outputs/audit_run_chexnet",
    },
}

LATEST_AUDIT_CACHE: Dict[str, Any] = {}
evaluator_instances: Dict[str, CandidateModelEvaluator] = {}


def resolve_audit_paths(model_id: str, dataset_id: Optional[str] = None):
    model_id = MODEL_ALIASES.get(model_id, model_id)
    if model_id not in MODEL_REGISTRY:
        model_id = "dr_lcnet_edge"
    model_meta = MODEL_REGISTRY[model_id]

    if not dataset_id or dataset_id not in DATASET_REGISTRY:
        if model_meta.get("modality") == "chest_xray":
            dataset_id = "chest_xray_60"
        else:
            dataset_id = "retinal_dr_60"

    dataset_meta = DATASET_REGISTRY[dataset_id]
    output_dir = os.path.join("outputs", f"audit_{model_id}_{dataset_id}")
    return model_meta, dataset_meta, output_dir


def get_evaluator(model_id: str) -> CandidateModelEvaluator:
    model_id = MODEL_ALIASES.get(model_id, model_id)
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not registered.")
    if model_id not in evaluator_instances:
        entry = MODEL_REGISTRY[model_id]
        evaluator_instances[model_id] = CandidateModelEvaluator(entry["path"], entry["name"])
    return evaluator_instances[model_id]


def load_test_samples(modality: str = "retinal_fundus") -> Dict[str, np.ndarray]:
    samples = {}
    if modality == "chest_xray":
        files = sorted(glob.glob("data/sample_scans/*.png"))[:6]
        for p in files:
            key = os.path.basename(p).split(".")[0]
            img = cv2.imread(p)
            if img is not None:
                samples[key] = img
    else:
        for p in sorted(glob.glob("assets/test_samples/*.jpg")):
            key = os.path.basename(p).split(".")[0]
            img = cv2.imread(p)
            if img is not None:
                samples[key] = img
    return samples


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "TrustCheck Clinical AI Audit Engine",
        "registered_models": list(MODEL_REGISTRY.keys()),
        "registered_datasets": list(DATASET_REGISTRY.keys()),
        "test_samples_count": len(load_test_samples()),
    }


@app.get("/api/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.values())}


@app.get("/api/datasets")
def list_datasets():
    return {"datasets": list(DATASET_REGISTRY.values())}


@app.post("/api/models/register")
def register_model_path(
    model_id: str = Form(...),
    name: str = Form(...),
    path: str = Form(...),
    architecture: str = Form("ONNX Clinical Classifier"),
    modality: str = Form("retinal_fundus"),
    target_deployment: str = Form("Local Tier-2 Server"),
):
    """Registers an existing ONNX model path on the host into the evaluation catalog."""
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {path}")

    MODEL_REGISTRY[model_id] = {
        "id": model_id,
        "name": name,
        "path": path,
        "architecture": architecture,
        "modality": modality,
        "target_deployment": target_deployment,
    }
    if model_id in evaluator_instances:
        del evaluator_instances[model_id]
    return {"status": "registered", "model": MODEL_REGISTRY[model_id]}


@app.post("/api/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    architecture: str = Form("ONNX Clinical Classifier"),
    modality: str = Form("retinal_fundus"),
    target_deployment: str = Form("Tier-1 Hospital Sandbox"),
):
    """Uploads an ONNX model file and registers it into TrustCheck evaluation harness."""
    os.makedirs("assets/models/uploads", exist_ok=True)
    clean_filename = os.path.basename(file.filename or "model.onnx")
    model_id = f"model_{int(time.time())}_{clean_filename.split('.')[0]}"
    target_path = os.path.join("assets/models/uploads", clean_filename)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded model file is empty.")

    with open(target_path, "wb") as f:
        f.write(content)

    MODEL_REGISTRY[model_id] = {
        "id": model_id,
        "name": name,
        "path": target_path,
        "architecture": architecture,
        "modality": modality,
        "target_deployment": target_deployment,
    }
    if model_id in evaluator_instances:
        del evaluator_instances[model_id]
    return {
        "status": "uploaded_and_registered",
        "model_id": model_id,
        "path": target_path,
        "name": name,
        "model": MODEL_REGISTRY[model_id],
    }


@app.post("/api/datasets/upload")
async def upload_dataset_metadata(
    metadata_file: UploadFile = File(...),
    dataset_name: str = Form("Custom Clinical Cohort"),
    modality: str = Form("retinal_fundus"),
):
    """Uploads cohort metadata CSV for clinical evaluation."""
    os.makedirs("data/uploads", exist_ok=True)
    clean_filename = os.path.basename(metadata_file.filename or "cohort.csv")
    dataset_id = f"ds_{int(time.time())}_{clean_filename.split('.')[0]}"
    dest_path = os.path.join("data/uploads", f"{int(time.time())}_{clean_filename}")

    content = await metadata_file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(dest_path)
        required_cols = ["patient_id", "image_path", "ground_truth"]
        missing = [c for c in required_cols if c not in df.columns]

        dataset_entry = {
            "id": dataset_id,
            "name": dataset_name,
            "modality": modality,
            "path": dest_path,
            "sample_count": len(df),
            "sites": list(df["site_id"].unique()) if "site_id" in df.columns else ["Custom Ingestion Site"],
            "description": f"Custom uploaded cohort: {len(df)} patient records",
            "compatible_models": [m for m, meta in MODEL_REGISTRY.items() if meta.get("modality") == modality],
            "default_output_dir": os.path.join("outputs", f"audit_custom_{dataset_id}"),
        }
        if len(missing) == 0:
            DATASET_REGISTRY[dataset_id] = dataset_entry

        return {
            "status": "uploaded",
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "metadata_path": dest_path,
            "n_samples": len(df),
            "columns": list(df.columns),
            "valid_contract": len(missing) == 0,
            "missing_columns": missing,
            "dataset": dataset_entry if len(missing) == 0 else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid cohort CSV format: {e}")


@app.get("/api/modalities")
def get_modalities():
    """Returns available clinical modality evaluation suites, supported stress corruptions, and benchmarks."""
    return {
        "modalities": [
            {
                "id": "retinal_fundus",
                "name": "Ophthalmology Suite (Retinal Fundus Scans)",
                "clinical_targets": ["Diabetic Retinopathy (5-Class)", "Glaucoma Cup/Disc", "Macular Edema"],
                "perturbations": ["Defocus Blur", "Corneal Glare", "Illumination Drop", "Sensor Downsampling"],
                "reference_benchmarks": ["EyePACS-1", "IDRiD", "MESSIDOR-2"],
                "default_model": "assets/models/dr_retinal_lcnet_edge.onnx",
            },
            {
                "id": "chest_xray",
                "name": "Thoracic Radiology Suite (Chest Radiographs)",
                "clinical_targets": ["Pneumonia", "Infiltration", "Cardiomegaly", "Atelectasis"],
                "perturbations": ["Contrast Attenuation", "Poisson Quantum Noise", "Patient Motion Blur", "Lead Markers"],
                "reference_benchmarks": ["NIH ChestX-ray14", "CheXNet", "MIMIC-CXR"],
                "default_model": "assets/models/cxr_chexnet_densenet121.onnx",
            },
            {
                "id": "ehr_tabular",
                "name": "Clinical Informatics Suite (EHR Labs & Vitals)",
                "clinical_targets": ["Sepsis Onset (qSOFA)", "48-Hour ICU Mortality", "Length of Stay"],
                "perturbations": ["Sensor Drift", "Missing Value MCAR/MAR", "Outlier Spike Noise"],
                "reference_benchmarks": ["MIMIC-IV-ED", "PhysioNet 2019 Challenge"],
                "default_model": "benchmark:tabular-xgboost",
            },
        ]
    }


@app.get("/api/pccp/triggers")
def get_pccp_triggers():
    """Returns FDA Predetermined Change Control Plan (PCCP) triggers and clinical actions."""
    cfg = load_config("config/audit_thresholds.yaml")
    return {
        "framework": cfg.get("framework", "TrustCheck-Clinical-AI-Safety"),
        "version": cfg.get("version", "2026.1"),
        "pccp_triggers": cfg.get("pccp_runtime_triggers", {}),
    }


@app.get("/api/pccp/rules")
def get_pccp_rules():
    """Returns deterministic scoring weights and gating thresholds."""
    cfg = load_config("config/audit_thresholds.yaml")
    return {
        "scoring_weights": cfg.get("scoring_weights", {}),
        "gating_rules": cfg.get("gating_rules", {}),
        "robustness_thresholds": cfg.get("engine_a_robustness", {}),
        "fairness_thresholds": cfg.get("engine_b_fairness", {}),
        "calibration_thresholds": cfg.get("engine_c_calibration", {}),
    }


RETINAL_GROUND_TRUTH = {
    "sample_clinical_pass": 0,
    "sample_corneal_glare": 0,
    "sample_low_illumination": 1,
    "sample_motion_blur": 1,
    "sample_silent_failure_candidate": 2,
    "sample_severe_npdr": 3,
}

CXR_GROUND_TRUTH = {
    "scan_0001": 1,
    "scan_0002": 0,
    "scan_0003": 1,
    "scan_0004": 0,
    "scan_0005": 0,
    "scan_0006": 1,
}


@app.post("/api/audit/run")
def run_audit(model_id: str = Query("dr_lcnet_edge")):
    canonical_id = MODEL_ALIASES.get(model_id, model_id)
    evaluator = get_evaluator(canonical_id)
    entry = MODEL_REGISTRY.get(canonical_id, {})
    modality = entry.get("modality", "retinal_fundus")
    samples = load_test_samples(modality=modality)
    if not samples:
        raise HTTPException(status_code=500, detail=f"No test samples found for modality {modality}.")

    gt_labels = CXR_GROUND_TRUTH if modality == "chest_xray" else RETINAL_GROUND_TRUTH
    audit_res = run_full_model_audit(evaluator, samples, ground_truth_labels=gt_labels)

    # Generate deployment certificate
    pdf_filename = f"TrustCheck_Certificate_{audit_res['audit_id']}.pdf"
    pdf_path = os.path.join("outputs", pdf_filename)
    generate_deployment_certificate(audit_res, pdf_path)

    audit_res["certificate_url"] = f"/outputs/{pdf_filename}"
    audit_res["model_id"] = canonical_id
    audit_res["model_name"] = entry.get("name", canonical_id)
    audit_res["modality"] = modality

    LATEST_AUDIT_CACHE[canonical_id] = audit_res
    if model_id != canonical_id:
        LATEST_AUDIT_CACHE[model_id] = audit_res
    LATEST_AUDIT_CACHE["latest"] = audit_res

    return audit_res


@app.get("/api/audit/latest")
def get_latest_audit(model_id: Optional[str] = None):
    canonical_id = MODEL_ALIASES.get(model_id, model_id) if model_id else None
    if canonical_id:
        if canonical_id not in LATEST_AUDIT_CACHE:
            return run_audit(canonical_id)
        return LATEST_AUDIT_CACHE[canonical_id]
    if "latest" in LATEST_AUDIT_CACHE:
        return LATEST_AUDIT_CACHE["latest"]
    return run_audit("dr_lcnet_edge")


@app.get("/api/audit/certificate/{filename}")
def download_certificate(filename: str):
    safe_filename = os.path.basename(filename)
    path = os.path.join("outputs", safe_filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Certificate file not found.")
    return FileResponse(path, media_type="application/pdf", filename=safe_filename)


@app.post("/api/stress/single")
async def test_single_stress(
    file: Optional[UploadFile] = File(None),
    sample_key: Optional[str] = Form(None),
    stress_type: str = Form("blur"),
    intensity: float = Form(3.0),
    model_id: str = Form("dr_lcnet_edge"),
):
    """Interactive stress endpoint for live slider testing."""
    evaluator = get_evaluator(model_id)

    if file:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif sample_key:
        samples = load_test_samples()
        if sample_key not in samples:
            raise HTTPException(status_code=404, detail=f"Sample {sample_key} not found.")
        img = samples[sample_key]
    else:
        samples = load_test_samples()
        if not samples:
            img = np.full((384, 384, 3), 128, dtype=np.uint8)
        else:
            img = list(samples.values())[0]

    # Apply requested perturbation
    if stress_type == "blur":
        perturbed = apply_gaussian_blur(img, float(intensity))
    elif stress_type == "illumination":
        # Intensity in [0, 100]% drop
        factor = max(0.05, 1.0 - (float(intensity) / 100.0))
        perturbed = apply_illumination_attenuation(img, factor)
    elif stress_type == "glare":
        perturbed = apply_corneal_glare(img, float(intensity))
    elif stress_type == "resolution":
        target_d = max(64, int(intensity))
        perturbed = apply_resolution_scaling(img, target_d)
    else:
        perturbed = img.copy()

    # Model inference on perturbed image
    result = evaluator.infer(perturbed)

    # Encode perturbed image to base64 JPEG for live preview
    _, buffer = cv2.imencode(".jpg", perturbed, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "stress_type": stress_type,
        "intensity": intensity,
        "predicted_grade": result["predicted_grade"],
        "confidence": result["confidence"],
        "probs": result["probs"],
        "biomarkers": result["biomarkers"],
        "latency_ms": result["latency_ms"],
        "image_base64": f"data:image/jpeg;base64,{b64_img}",
    }


@app.get("/api/audit/cohort-summary")
def get_cohort_summary(
    model_id: str = Query("dr_lcnet_edge"),
    dataset_id: Optional[str] = Query(None),
):
    """Returns standardized multicenter cohort audit telemetry for selected (model, dataset)."""
    model_meta, dataset_meta, output_dir = resolve_audit_paths(model_id, dataset_id)

    candidates = [
        os.path.join(output_dir, "telemetry.json"),
        os.path.join(dataset_meta.get("default_output_dir", ""), "telemetry.json"),
        "outputs/audit_run_01/telemetry.json",
        "output/audit_run_01/telemetry.json",
    ]

    for c in candidates:
        if os.path.exists(c):
            with open(c, "r") as f:
                data = json.load(f)
                data["requested_model_id"] = model_id
                data["requested_dataset_id"] = dataset_meta["id"]
                data["model_metadata"] = model_meta
                data["dataset_metadata"] = dataset_meta
                return data

    return {
        "status": "not_executed",
        "requested_model_id": model_id,
        "requested_dataset_id": dataset_meta["id"],
        "model_metadata": model_meta,
        "dataset_metadata": dataset_meta,
        "message": f"Cohort audit telemetry not found for {model_meta['name']} on {dataset_meta['name']}. Click 'Run Clinical Audit' to execute.",
    }


@app.post("/api/audit/run-cohort")
def run_cohort_audit(
    model_id: str = Query("dr_lcnet_edge"),
    dataset_id: Optional[str] = Query(None),
):
    """Executes full multi-engine audit battery on selected model + dataset pair."""
    from audit_runner import run_evaluation_suite

    model_meta, dataset_meta, output_dir = resolve_audit_paths(model_id, dataset_id)
    os.makedirs(output_dir, exist_ok=True)

    telemetry = run_evaluation_suite(
        model_target=model_meta["path"],
        metadata_path=dataset_meta["path"],
        output_dir=output_dir,
        modality=dataset_meta["modality"],
    )

    # Cache into default output dir for fast lookup
    default_dir = dataset_meta.get("default_output_dir")
    if default_dir and default_dir != output_dir:
        os.makedirs(default_dir, exist_ok=True)
        with open(os.path.join(default_dir, "telemetry.json"), "w") as f:
            json.dump(telemetry, f, indent=2)
        pdf_src = os.path.join(output_dir, "audit_certificate.pdf")
        if os.path.exists(pdf_src):
            import shutil
            shutil.copy2(pdf_src, os.path.join(default_dir, "audit_certificate.pdf"))

    telemetry["requested_model_id"] = model_id
    telemetry["requested_dataset_id"] = dataset_meta["id"]
    telemetry["model_metadata"] = model_meta
    telemetry["dataset_metadata"] = dataset_meta
    return telemetry


@app.get("/api/audit/cohort-pdf")
def get_cohort_pdf(
    model_id: str = Query("dr_lcnet_edge"),
    dataset_id: Optional[str] = Query(None),
):
    """Downloads FDA/CDSCO SaMD regulatory audit certificate PDF for selected (model, dataset)."""
    model_meta, dataset_meta, output_dir = resolve_audit_paths(model_id, dataset_id)
    candidates = [
        os.path.join(output_dir, "audit_certificate.pdf"),
        os.path.join(dataset_meta.get("default_output_dir", ""), "audit_certificate.pdf"),
        "outputs/audit_run_01/audit_certificate.pdf",
        "output/audit_run_01/audit_certificate.pdf",
    ]

    for c in candidates:
        if os.path.exists(c):
            filename = f"TrustCheck_Certificate_{model_id}_{dataset_meta['id']}.pdf"
            return FileResponse(c, media_type="application/pdf", filename=filename)

    # Fallback to existing sample or latest generated certificate
    fallback_candidates = [
        "outputs/TrustCheck_Certificate_Sample.pdf",
        *sorted(glob.glob("outputs/TrustCheck_Certificate_*.pdf"), reverse=True),
    ]
    for f in fallback_candidates:
        if os.path.exists(f):
            filename = f"TrustCheck_Certificate_{model_id}_{dataset_meta['id']}.pdf"
            return FileResponse(f, media_type="application/pdf", filename=filename)

    raise HTTPException(status_code=404, detail="Cohort PDF certificate not generated yet.")
