# 🛡️ TrustCheck — Clinical AI Deployment Safety & Stress-Testing Platform

[![NixOS Flake](https://img.shields.io/badge/NixOS-Flake%20Reproducible-blue.svg)](flake.nix)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![Vite React](https://img.shields.io/badge/Vite-React%2019-646CFF.svg)](https://vitejs.dev)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Inference-orange.svg)](https://onnxruntime.ai)

An independent clinical evaluation harness and stress-testing platform designed to test, benchmark, and audit healthcare AI models before hospital deployment. Built for **MUJ HACKX 4.0 (Problem Statement HLT-08)**.

---

## 📌 Clinical Motivation

Over 90% of medical deep learning models that report $>95\%$ benchmark accuracy in papers fail catastrophically when deployed in production hospitals. The culprits are well-documented:
1. **Sensor & Environmental Shift**: Models trained on high-end hospital fundus cameras fail when exposed to handheld, low-cost cameras with optical blur, flash drop, or glare.
2. **Silent Shortcut Learning**: Classification heads learn background noise correlations, predicting "Normal Retina" (Grade 0) even when physical microaneurysms and hemorrhages are visibly present.
3. **Uncalibrated Overconfidence**: Models report $>90\%$ confidence on borderline or degraded inputs where they are actually guessing.

**TrustCheck acts as an independent crash-testing facility for medical AI**: it ingests candidate models, runs adversarial stress ladders, computes Expected Calibration Error (ECE), catches silent false negatives against physical lesion segmentations, and issues an **FDA SaMD / EU AI Act-compliant Hospital Deployment Readiness Certificate**.

---

## 🏛️ System Architecture

```plaintext
┌────────────────────────────────────────────────────────────────────────┐
│                          Candidate AI Model                            │
│                 (.onnx / .pth / Deterministic Weights)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        TrustCheck Audit Engine                         │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│ 1. Optical Stress Ladder │ 2. Calibration Engine    │ 3. Sanity Check  │
│ • Defocus Blur (σ=0..6)  │ • Expected Calib. Error  │ • Class Logits   │
│ • Flash Drop (0..-80%)   │ • Brier Score            │   vs. Physical   │
│ • Corneal Glare (0..0.95)│ • Overconfidence Penalty │   Lesion Masks   │
│ • Downsampling (384..96) │ • Reliability Bins       │ • Silent Misses  │
└──────────────────────────┴──────────────────────────┴──────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       Audit Aggregator & Scorer                        │
│               Hospital Readiness Score: 0 - 100 / 100                  │
│       Verdict: APPROVED / CONDITIONAL PASS / REJECTED (HIGH RISK)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌────────────────────────────────┐            ┌────────────────────────────────┐
│   FastAPI + React Dashboard    │            │ Automated Deployment Audit PDF │
│ • Interactive Stress Sliders   │            │ • FDA SaMD Style Certificate   │
│ • Real-time Perturbation Feed  │            │ • Vector Degradation Tables    │
│ • Model-to-Model Arena Table   │            │ • Auditor Committee Sign-off   │
└────────────────────────────────┘            └────────────────────────────────┘
```

---

## 🚀 Quickstart Guide

### Prerequisites
- NixOS or Linux with `nix` installed (or Python 3.11 with `onnxruntime`, `fastapi`, `opencv`, `reportlab`).

### 1. Launch Full Stack (Backend + UI)
```bash
./start.sh 3
```
- **FastAPI Audit Engine**: `http://localhost:8000` (API documentation at `/docs`)
- **TrustCheck Studio UI**: `http://localhost:5173`

### 2. Run Automated Pytest Suite
```bash
./start.sh 1
```

### 3. Generate Deployment Certificate via CLI
```bash
./start.sh 2
```
Output certificate generated in `outputs/TrustCheck_Certificate_CLI.pdf`.

---

## 👥 4-Person Hackathon Role Mapping

| Role | Primary Module | Core Ownership |
| :--- | :--- | :--- |
| **Person 1: ML Stress Harness Lead** | `engine/perturbation.py` | Optical stress transforms (Gaussian defocus, illumination drop, corneal glare, sensor downscaling). |
| **Person 2: Safety & Calibration Lead** | `engine/calibration.py` & `engine/sanity_check.py` | Expected Calibration Error (ECE), Brier scoring, and shortcut learning detection (Class vs Lesion consensus). |
| **Person 3: Backend & PDF Engine Lead** | `backend/main.py` & `backend/certificate_gen.py` | FastAPI audit endpoints, interactive slider API (`/api/stress/single`), and ReportLab FDA certificate generator. |
| **Person 4: Auditor UI & Pitch Lead** | `ui/src/App.jsx` & `ui/src/App.css` | React studio dashboard, live interactive stress slider, silent failure inspector, and 3-minute pitch deck. |

---

## 📁 Repository Structure

```plaintext
trust-check/
├── flake.nix                       # NixOS development shell definition
├── flake.lock                      # Pinned nix dependencies
├── .envrc                          # Direnv auto-loader
├── start.sh                        # Master runner script (tests, audit CLI, full stack)
├── README.md                       # Documentation & clinical specification
│
├── assets/                         # Test Subjects & Datasets (Treated as 3rd-party)
│   ├── models/
│   │   ├── dr_retinal_lcnet_edge.onnx      # Diabetic Retinopathy PP-LCNet edge model (12.7 MB)
│   │   ├── dr_retinal_resnet_teacher.onnx  # Diabetic Retinopathy ResNet teacher model (44.6 MB)
│   │   ├── cxr_chexnet_densenet121.onnx    # Chest X-Ray CheXNet DenseNet121 hospital benchmark (27.7 MB)
│   │   └── cxr_mobilenet_edge.onnx         # Chest X-Ray MobileNetV2 bedside cart edge model (8.4 MB)
│   └── test_samples/               # Multi-cohort clinical scans (IDRiD, DRIMDB, ODIR)
│       ├── sample_clinical_pass.jpg
│       ├── sample_severe_npdr.jpg
│       ├── sample_low_illumination.jpg
│       ├── sample_motion_blur.jpg
│       ├── sample_corneal_glare.jpg
│       └── sample_silent_failure_candidate.jpg
│
├── engine/                         # Core Safety & Stress Evaluation Harness
│   ├── __init__.py
│   ├── perturbation.py             # Physical optical & sensor degradation functions
│   ├── calibration.py              # Expected Calibration Error (ECE) & reliability curves
│   ├── sanity_check.py             # Dual-engine classification vs lesion discrepancy check
│   └── auditor.py                  # End-to-end multi-vector audit engine & scoring
│
├── backend/                        # API & Certificate Reporting
│   ├── __init__.py
│   ├── main.py                     # FastAPI server with live interactive endpoints
│   └── certificate_gen.py          # ReportLab PDF FDA-style deployment certificate
│
├── ui/                             # React / Vite Dark-Mode Studio
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                 # 4 Views: Overview, Stress Studio, Failures, Arena
│       ├── App.css                 # Clean, dark, clinical styling
│       ├── index.css
│       └── main.jsx
│
├── tests/                          # Automated Verification
│   └── test_trustcheck.py          # 4 comprehensive unit tests
│
└── outputs/                        # Output certificates & audit records
    └── TrustCheck_Certificate_*.pdf
```
