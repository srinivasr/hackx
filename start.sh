#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  🛡️  TrustCheck: Healthcare AI Safety & Clinical Audit Harness    "
echo "=================================================================="

ACTION="${1:-3}"

case "$ACTION" in
    1)
        echo "Running Automated Test Suite..."
        nix develop "$SCRIPT_DIR" --command env PYTHONPATH=. pytest -v tests/
        ;;
    2)
        echo "Executing HLT-08 Pre-Deployment Audit Battery (CLI)..."
        nix develop "$SCRIPT_DIR" --command python3 audit_runner.py \
            --model benchmark:chexnet-densenet121 \
            --metadata data/sample_metadata.csv \
            --output-dir output/audit_run_01
        ;;
    3)
        if [ ! -d "$SCRIPT_DIR/ui/node_modules" ]; then
            echo "📦 Installing Web UI dependencies (first run)..."
            nix develop "$SCRIPT_DIR" --command bash -c "cd '$SCRIPT_DIR/ui' && npm install"
        fi

        echo "Starting FastAPI Audit Server (Port 8000)..."
        nix develop "$SCRIPT_DIR" --command python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!

        echo "Starting TrustCheck Web UI (Port 5173)..."
        nix develop "$SCRIPT_DIR" --command npm --prefix "$SCRIPT_DIR/ui" run dev -- --host 0.0.0.0 --port 5173 &
        UI_PID=$!

        trap "kill $BACKEND_PID $UI_PID 2>/dev/null || true" EXIT
        wait
        ;;
    4)
        echo "Generating Deployment Audit Certificate PDF..."
        nix develop "$SCRIPT_DIR" --command python3 -c "
import glob, cv2
from engine.auditor import CandidateModelEvaluator, run_full_model_audit
from backend.certificate_gen import generate_deployment_certificate

evaluator = CandidateModelEvaluator('assets/models/candidate_model_a.onnx', 'Candidate-Model-A')
samples = {f.split('/')[-1].split('.')[0]: cv2.imread(f) for f in glob.glob('assets/test_samples/*.jpg')}
audit = run_full_model_audit(evaluator, samples)
generate_deployment_certificate(audit, 'outputs/TrustCheck_Certificate_Sample.pdf')
print('Certificate generated at outputs/TrustCheck_Certificate_Sample.pdf')
"
;;
5)
echo "Launching Interactive Clinical AI Safety Dashboard (Streamlit)..."
nix develop "$SCRIPT_DIR" --command streamlit run reporting/dashboard.py -- --telemetry-path output/audit_run_01/telemetry.json
;;
6)
echo "Executing Diabetic Retinopathy (Retinal Fundus) Audit Battery..."
nix develop "$SCRIPT_DIR" --command python3 audit_runner.py \
    --model assets/models/candidate_model_a.onnx \
    --metadata data/sample_retinal_metadata.csv \
    --modality retinal_fundus \
    --output-dir output/audit_run_retinal_dr
;;
*)
echo "Usage: ./start.sh [1|2|3|4|5|6]"
echo "  1) Run pytest test suite"
echo "  2) Run HLT-08 clinical audit battery (CheXNet X-Ray)"
echo "  3) Launch full stack (FastAPI + Vite UI)"
echo "  4) Generate retinal demo PDF certificate"
echo "  5) Launch Streamlit Auditor Dashboard"
echo "  6) Run Diabetic Retinopathy (Retinal Fundus ONNX) audit battery"
exit 1
;;
esac
