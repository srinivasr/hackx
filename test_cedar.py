from cedarpy import is_authorized, Decision
policy = """
permit(
    principal == Hospital::Role::"ChiefMedicalOfficer",
    action == Action::"CertifyDeployment",
    resource in Hospital::Department::"EmergencyICU"
)
when {
    context.sensitivity >= 95
};
"""
request = {
    "principal": 'Hospital::Role::"ChiefMedicalOfficer"',
    "action": 'Action::"CertifyDeployment"',
    "resource": 'Hospital::Department::"EmergencyICU"',
    "context": {"sensitivity": 96}
}
res = is_authorized(request, policy, [])
print("Decision:", res.decision)
print("Decision == Allow:", res.decision == Decision.Allow)
