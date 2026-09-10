# PRODUCT.md — TrustCheck

## Vision & Purpose
TrustCheck is an independent pre-deployment safety evaluation and stress-testing harness for healthcare AI models. It addresses the clinical reality where medical models scoring $>95\%$ benchmark accuracy on clean academic datasets silently fail in production due to camera differences, optical noise, and shortcut learning.

## Target Audience & Usage Scene
- **Primary Users**: Hospital Chief Medical Officers (CMOs), Clinical AI Safety Auditors, Regulatory Compliance Committees (FDA SaMD / EU AI Act).
- **Environment**: Hospital workstations and medical informatics review screens under ambient clinical lighting.
- **Mode**: **Operate** — scanning speed, decision defensibility, signal density, and unambiguous risk telemetry take precedence over decorative consumer aesthetics.

## Key Capabilities
1. **Automated Stress Ladders**: Defocus blur, illumination attenuation, corneal glare, and sensor resolution scaling.
2. **Uncertainty & Calibration (ECE)**: Expected Calibration Error computation to detect dangerous overconfidence.
3. **Shortcut Learning & Discrepancy Detection**: Dual-engine cross-check flagging when classification outputs Grade 0 (Normal) despite active segmented physical lesions.
4. **Model Comparison Arena**: Direct empirical trade-off analysis between edge-efficient mobile architectures and heavyweight hospital server models.
5. **Regulatory Certification**: Automated 1-page FDA-grade PDF Pre-Deployment Safety Audit Certificate.
