// AWS Cedar Clinical Model Authorization Policy
// Namespace: Hospital::Governance

permit(
    principal == Hospital::Role::"ChiefMedicalOfficer",
    action == Action::"CertifyDeployment",
    resource in Hospital::Department::"EmergencyICU"
)
when {
    context.sensitivity_bp >= 9500 &&
    context.shortcut_vulnerability_index_bp <= 1000 &&
    context.subgroup_fnr_disparity_bp <= 1000 &&
    context.expected_calibration_error_bp <= 800 &&
    context.silent_false_negatives == 0
};

// Explicit deny for missing schemas or critical metrics (Default-Deny)
forbid(
    principal,
    action == Action::"CertifyDeployment",
    resource
)
when {
    !(context has sensitivity_bp) ||
    !(context has subgroup_fnr_disparity_bp) ||
    !(context has silent_false_negatives) ||
    !(context has adversarial_resilience_drop_bp)
};

forbid(
    principal,
    action == Action::"CertifyDeployment",
    resource
)
when {
    context.adversarial_resilience_drop_bp > 2000 ||
    context.underpowered_subgroup == true
};
