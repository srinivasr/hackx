import pytest
from backend.cedar_gate import evaluate_cedar_gate

def test_cedar_gate_permit():
    metrics = {
        "sensitivity": 96,
        "shortcut_vulnerability_index": 5,
        "subgroup_fnr_disparity": 2,
        "expected_calibration_error": 4,
        "silent_false_negatives": 0,
        "adversarial_resilience_drop": 10,
        "underpowered_subgroup": False
    }
    result = evaluate_cedar_gate(metrics, department="EmergencyICU")
    assert result["decision"] == "PERMIT"

def test_cedar_gate_forbid_silent_fn():
    metrics = {
        "sensitivity": 98,
        "shortcut_vulnerability_index": 5,
        "subgroup_fnr_disparity": 2,
        "expected_calibration_error": 4,
        "silent_false_negatives": 2, # Fails here
        "adversarial_resilience_drop": 10,
        "underpowered_subgroup": False
    }
    result = evaluate_cedar_gate(metrics, department="EmergencyICU")
    assert result["decision"] == "FORBID"

def test_cedar_gate_forbid_adversarial():
    metrics = {
        "sensitivity": 96,
        "shortcut_vulnerability_index": 5,
        "subgroup_fnr_disparity": 2,
        "expected_calibration_error": 4,
        "silent_false_negatives": 0,
        "adversarial_resilience_drop": 25, # Fails here (forbid rule triggers)
        "underpowered_subgroup": False
    }
    result = evaluate_cedar_gate(metrics, department="EmergencyICU")
    assert result["decision"] == "FORBID"

def test_cedar_gate_wrong_department():
    metrics = {
        "sensitivity": 96,
        "shortcut_vulnerability_index": 5,
        "subgroup_fnr_disparity": 2,
        "expected_calibration_error": 4,
        "silent_false_negatives": 0,
        "adversarial_resilience_drop": 10,
        "underpowered_subgroup": False
    }
    result = evaluate_cedar_gate(metrics, department="Pediatrics")
    # Because resource doesn't match the permit rule, it defaults to deny/forbid
    assert result["decision"] == "FORBID"
