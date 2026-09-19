// AWS Cedar Clinical Model Authorization Policy
// Namespace: Hospital::Governance

permit(
    principal,
    action == Action::"CertifyDeployment",
    resource
)
when {
    context.sensitivity_bp >= resource.required_sensitivity_bp &&
    context.shortcut_vulnerability_index_bp <= resource.max_shortcut_vulnerability_bp &&
    context.subgroup_fnr_disparity_bp <= resource.max_fnr_disparity_bp &&
    context.expected_calibration_error_bp <= resource.max_ece_bp &&
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
    !(context has adversarial_resilience_drop_bp) ||
    !(resource has required_sensitivity_bp)
};

forbid(
    principal,
    action == Action::"CertifyDeployment",
    resource
)
when {
    context.adversarial_resilience_drop_bp > resource.max_adversarial_drop_bp ||
    context.underpowered_subgroup == true
};

