#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  🛡️  TrustCheck: Healthcare AI Safety & Clinical Audit Harness    "
echo "=================================================================="

ACTION="${1:-3}"

# Intelligent environment runner: Nix develop if available, else local virtual environment
if command -v nix >/dev/null 2>&1; then
    USE_NIX=true
else
    USE_NIX=false
    if [ ! -d "$SCRIPT_DIR/.venv" ]; then
        echo "📦 Setting up Python virtual environment..."
        if command -v uv >/dev/null 2>&1; then
            uv venv "$SCRIPT_DIR/.venv"
            uv pip install -r "$SCRIPT_DIR/requirements.txt"
        elif [ -x "$HOME/.local/bin/uv" ]; then
            "$HOME/.local/bin/uv" venv "$SCRIPT_DIR/.venv"
            "$HOME/.local/bin/uv" pip install -r "$SCRIPT_DIR/requirements.txt"
        else
            python3 -m venv "$SCRIPT_DIR/.venv"
            "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
        fi
    fi

    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
    export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
fi

run_cmd() {
    if [ "$USE_NIX" = true ]; then
        nix develop "$SCRIPT_DIR" --command bash -c "$*"
    else
        bash -c "$*"
    fi
}

case "$ACTION" in
    1)
        echo "Running Automated Test Suite..."
        run_cmd "env PYTHONPATH=. pytest -v tests/"
        ;;
    2)
        echo "Executing HLT-08 Pre-Deployment Audit Battery (CLI)..."
        run_cmd "python3 audit_runner.py --model benchmark:chexnet-densenet121 --metadata data/sample_metadata.csv --output-dir output/audit_run_01"
        ;;
    3)
        if [ ! -d "$SCRIPT_DIR/ui/node_modules" ]; then
            echo "📦 Installing Web UI dependencies (first run)..."
            run_cmd "cd '$SCRIPT_DIR/ui' && npm install"
        fi

        echo "Starting FastAPI Audit Server (Port 8000)..."
        run_cmd "python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000" &
        BACKEND_PID=$!

        echo "Starting TrustCheck Web UI (Port 5173)..."
        run_cmd "npm --prefix '$SCRIPT_DIR/ui' run dev -- --host 0.0.0.0 --port 5173" &
        UI_PID=$!

        trap "kill $BACKEND_PID $UI_PID 2>/dev/null || true" EXIT
        wait
        ;;
    4)
        echo "Generating Deployment Audit Certificate PDF..."
        run_cmd "python3 -c \"
import glob, cv2
from engine.auditor import CandidateModelEvaluator, run_full_model_audit
from backend.certificate_gen import generate_deployment_certificate

evaluator = CandidateModelEvaluator('assets/models/candidate_model_a.onnx', 'Candidate-Model-A')
samples = {f.split('/')[-1].split('.')[0]: cv2.imread(f) for f in glob.glob('assets/test_samples/*.jpg')}
audit = run_full_model_audit(evaluator, samples)
generate_deployment_certificate(audit, 'outputs/TrustCheck_Certificate_Sample.pdf')
print('Certificate generated at outputs/TrustCheck_Certificate_Sample.pdf')
\""
        ;;
    5)
        echo "Launching Interactive Clinical AI Safety Dashboard (Streamlit)..."
        run_cmd "streamlit run reporting/dashboard.py -- --telemetry-path output/audit_run_01/telemetry.json"
        ;;
    6)
        echo "Executing Diabetic Retinopathy (Retinal Fundus) Audit Battery..."
        run_cmd "python3 audit_runner.py --model assets/models/candidate_model_a.onnx --metadata data/sample_retinal_metadata.csv --modality retinal_fundus --output-dir output/audit_run_retinal_dr"
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

