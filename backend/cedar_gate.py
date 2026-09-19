import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any
from cedarpy import is_authorized, Decision

def _scale_to_basis_points(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Scales float/integer percentages to basis points (x100) for strict Cedar 64-bit integer determinism."""
    bp_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in ["silent_false_negatives"]:
                bp_metrics[key] = int(value)  # keep as raw integer count
            else:
                bp_metrics[f"{key}_bp"] = int(value * 100)
        else:
            bp_metrics[key] = value
    return bp_metrics

def evaluate_cedar_gate(metrics: Dict[str, Any], department: str = "EmergencyICU") -> Dict[str, Any]:
    """Evaluates empirical model metrics against AWS Cedar governance policy."""
    policy_path = os.path.join(os.path.dirname(__file__), "..", "cedar", "policies", "hospital_pacs.cedar")
    
    with open(policy_path, "r") as f:
        policy_str = f.read()

    # Convert to basis points to prevent floating-point non-determinism
    context_payload = _scale_to_basis_points(metrics)

    request = {
        "principal": 'Hospital::Role::"ChiefMedicalOfficer"',
        "action": 'Action::"CertifyDeployment"',
        "resource": f'Hospital::Department::"{department}"',
        "context": context_payload
    }

    # Define resource entity thresholds based on department
    # This allows different departments to have different risk tolerances
    thresholds = {
        "required_sensitivity_bp": 9500,
        "max_shortcut_vulnerability_bp": 1000,
        "max_fnr_disparity_bp": 1000,
        "max_ece_bp": 800,
        "max_adversarial_drop_bp": 2000,
    }

    entities = [
        {
            "uid": {"type": "Hospital::Department", "id": department},
            "attrs": thresholds,
            "parents": []
        }
    ]

    try:
        authz_result = is_authorized(
            request,
            policies=policy_str,
            entities=entities
        )
        
        decision_str = "PERMIT" if authz_result.decision == Decision.Allow else "FORBID"
        
        # Generate Cryptographic Tamper-Evident Trace
        trace_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "request": request,
            "decision": decision_str,
            "reasons": list(authz_result.diagnostics.reasons) if authz_result.diagnostics else []
        }
        
        # Create SHA-256 fingerprint
        payload_str = json.dumps(trace_payload, sort_keys=True)
        fingerprint = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        
        return {
            "decision": decision_str,
            "diagnostics": {
                "reason": trace_payload["reasons"],
                "errors": list(authz_result.diagnostics.errors) if authz_result.diagnostics else []
            },
            "tamper_evident_fingerprint": f"sha256:{fingerprint}"
        }
    except Exception as e:
        return {
            "decision": "ERROR",
            "error": str(e)
        }
