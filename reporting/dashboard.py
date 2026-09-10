"""Step 4A: Auditor Dashboard (Streamlit).

Interactive clinical AI stress-testing telemetry visualization dashboard.
Displays high-contrast global status header, 4-pillar radar breakdown, hardware decay curves,
shortcut learning diagnostics, demographic disparity heatmaps, and calibration reliability diagrams.

Run:
  streamlit run reporting/dashboard.py -- --telemetry-path output/audit_run_01/telemetry.json
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed in current environment. Run with: uv run --with streamlit streamlit run reporting/dashboard.py")
    sys.exit(0)

# Page Configuration
st.set_page_config(
    page_title="TrustCheck Clinical AI Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark Clinical Precision Theme CSS (Unslop UI - No purple gradients, clean clinical dark palette)
st.markdown("""
<style>
    .stApp {
        background-color: #0A0E17;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1F2B3F;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 11px;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 11px;
        color: #64748B;
    }
    .verdict-go {
        background-color: rgba(5, 150, 105, 0.15);
        border: 1px solid #059669;
        color: #34D399;
        padding: 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 18px;
    }
    .verdict-caution {
        background-color: rgba(217, 119, 6, 0.15);
        border: 1px solid #D97706;
        color: #FBBF24;
        padding: 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 18px;
    }
    .verdict-nogo {
        background-color: rgba(220, 38, 38, 0.15);
        border: 1px solid #DC2626;
        color: #F87171;
        padding: 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 18px;
    }
    .contra-box {
        background-color: rgba(220, 38, 38, 0.10);
        border-left: 4px solid #DC2626;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


def parse_args():
    telemetry_path = "output/audit_run_01/telemetry.json"
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        extra_args = sys.argv[idx + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--telemetry-path", type=str, default=telemetry_path)
        args, _ = parser.parse_known_args(extra_args)
        return args.telemetry_path
    return telemetry_path


telemetry_file = parse_args()

st.title("🛡️ TrustCheck: Clinical AI Pre-Deployment Auditor")
st.caption("Adversarial Pre-Deployment Stress-Testing & Safety Harness (HLT-08 Compliance | FDA SaMD / CDSCO Class C/D)")

if not os.path.exists(telemetry_file):
    st.warning(f"Telemetry file not found at `{telemetry_file}`. Please execute `python audit_runner.py` first.")
    st.stop()

with open(telemetry_file, "r") as f:
    data = json.load(f)

# Global Telemetry Header Bar
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(f"**Target Model:** `{data.get('target_model', 'N/A')}`")
    st.markdown(f"**Audit Run ID:** `{data.get('audit_run_id', 'N/A')}` | Timestamp: `{data.get('timestamp', 'N/A')}`")
with c2:
    st.metric("Composite TrustScore", f"{data.get('trust_score', 0)} / 100")
with c3:
    verdict = data.get("verdict", "CAUTION")
    if "GO" in verdict and "NO" not in verdict:
        st.markdown('<div class="verdict-go">✅ VERDICT: GO (APPROVED)</div>', unsafe_allow_html=True)
    elif "CAUTION" in verdict:
        st.markdown('<div class="verdict-caution">⚠️ VERDICT: CAUTION (RESTRICTED)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-nogo">🛑 VERDICT: NO-GO (REJECTED)</div>', unsafe_allow_html=True)

st.divider()

metrics = data.get("metrics", {})
rob = metrics.get("robustness", {})
fair = metrics.get("fairness_and_shift", {})
cal = metrics.get("calibration_and_uncertainty", {})

# 4-Pillar Score Cards Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">1. Hardware & Physics (35%)</div>
        <div class="metric-value">{rob.get('robustness_score', 'N/A')} <span style="font-size: 14px; color: #64748B;">/ 100</span></div>
        <div class="metric-sub">Clean AUROC: {rob.get('clean_auroc', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">2. Demographic Equity (30%)</div>
        <div class="metric-value">{fair.get('fairness_score', 'N/A')} <span style="font-size: 14px; color: #64748B;">/ 100</span></div>
        <div class="metric-sub">Max Disparity: {fair.get('max_disparity', 'N/A')}x</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">3. Uncertainty & Calibration (20%)</div>
        <div class="metric-value">{cal.get('calibration_score', 'N/A')} <span style="font-size: 14px; color: #64748B;">/ 100</span></div>
        <div class="metric-sub">ECE: {cal.get('expected_calibration_error', 0)*100:.1f}% | ACE: {cal.get('adaptive_calibration_error', 0)*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">4. Distribution Shift (15%)</div>
        <div class="metric-value">{fair.get('shift_score', 'N/A')} <span style="font-size: 14px; color: #64748B;">/ 100</span></div>
        <div class="metric-sub">Domain AUC: {fair.get('domain_shift_classifier_auroc', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs
tab_exec, tab_rob, tab_fair, tab_cal, tab_pccp = st.tabs([
    "📋 Executive Overview & Contraindications",
    "⚙️ Hardware Degradation & Shortcuts",
    "👥 Demographic Equity & Covariate Drift",
    "🎯 Calibration & Silent Failure Inspector",
    "📜 PCCP Regulatory Plan & Export",
])

# ---------------------------------------------------------------------------
# Tab 1: Executive Overview
# ---------------------------------------------------------------------------
with tab_exec:
    st.subheader("Clinical Indications & Contraindications")
    contraindications = data.get("clinical_contraindications", [])
    if contraindications:
        for c in contraindications:
            st.markdown(f'<div class="contra-box">⚠️ <b>{c}</b></div>', unsafe_allow_html=True)
    else:
        st.success("No active contraindications identified. Candidate model satisfies baseline safety criteria.")

    st.subheader("Pre-Deployment Gating Rules")
    gating_df = pd.DataFrame([
        {"Pillar": "Hardware Stability", "Metric": "Contrast Attenuation T3 AUROC", "Threshold": ">= 0.70", "Observed": f"{rob.get('decay_slopes', {}).get('contrast_attenuation', [0, 0, 0])[2]:.3f}", "Status": "PASS" if not rob.get("contrast_decay_failure", False) else "FAIL"},
        {"Pillar": "Demographic Equity", "Metric": "Pediatric Under-Diagnosis Ratio", "Threshold": "<= 1.80x", "Observed": f"{fair.get('underdiagnosis_ratio_pediatric_vs_adult', 'N/A')}x", "Status": "PASS" if fair.get("underdiagnosis_ratio_pediatric_vs_adult", 1.0) <= 1.80 else "FAIL"},
        {"Pillar": "Demographic Equity", "Metric": "Female Under-Diagnosis Ratio", "Threshold": "<= 1.80x", "Observed": f"{fair.get('underdiagnosis_ratio_female_vs_male', 'N/A')}x", "Status": "PASS" if fair.get("underdiagnosis_ratio_female_vs_male", 1.0) <= 1.80 else "FAIL"},
        {"Pillar": "Calibration", "Metric": "Expected Calibration Error (ECE)", "Threshold": "<= 15.0%", "Observed": f"{cal.get('expected_calibration_error', 0)*100:.1f}%", "Status": "PASS" if cal.get("expected_calibration_error", 1.0) <= 0.15 else "WARNING"},
        {"Pillar": "Safety", "Metric": "Critical High-Confidence FN Misses", "Threshold": "0 cases", "Observed": f"{cal.get('critical_silent_failures_count', 0)} cases", "Status": "PASS" if cal.get("critical_silent_failures_count", 0) == 0 else "FAIL"},
    ])
    st.table(gating_df)

# ---------------------------------------------------------------------------
# Tab 2: Hardware Degradation & Shortcuts
# ---------------------------------------------------------------------------
with tab_rob:
    st.subheader("Hardware Degradation AUROC Decay Slopes (Tiers 1–5)")
    st.caption("Lower red dashed threshold represents the clinical failure floor (AUROC = 0.70).")
    
    decay = rob.get("decay_slopes", {})
    if decay:
        df_decay = pd.DataFrame(decay, index=["Tier 1", "Tier 2", "Tier 3", "Tier 4", "Tier 5"])
        df_decay["Clinical Floor (0.70)"] = 0.70
        st.line_chart(df_decay)
        st.dataframe(df_decay.drop(columns=["Clinical Floor (0.70)"]), use_container_width=True)

    st.subheader("Non-Clinical Visual Shortcut Diagnostics (DeGrave et al. Nature MI 2021)")
    s_diag = rob.get("shortcut_diagnostics", {})
    fp_rates = s_diag.get("distractor_fp_rates", {})
    conf_deltas = s_diag.get("distractor_conf_deltas", {})
    if fp_rates:
        shortcut_rows = []
        for s_name, fp in fp_rates.items():
            delta = conf_deltas.get(s_name, 0.0)
            shortcut_rows.append({
                "Shortcut Archetype": s_name.replace("_", " ").title(),
                "False Positive Trigger Rate": f"{fp*100:.1f}%",
                "Mean |Delta Confidence|": f"{delta:.4f}",
                "Vulnerability": "SUSCEPTIBLE" if (fp > 0.05 or delta > 0.10) else "INVARIANT",
            })
        st.table(pd.DataFrame(shortcut_rows))
        st.caption(f"Overall Shortcut Vulnerability Index (SVI): **{s_diag.get('shortcut_vulnerability_index', 0.0):.4f}**")

# ---------------------------------------------------------------------------
# Tab 3: Demographic Equity & Drift
# ---------------------------------------------------------------------------
with tab_fair:
    st.subheader("Demographic Slices Under-Diagnosis Disparity")
    slices = fair.get("slices", {})

    c_age, c_sex = st.columns(2)
    with c_age:
        st.markdown("**Age Slices**")
        age_data = slices.get("age", {})
        if age_data:
            st.table(pd.DataFrame(age_data).T[["sample_count", "fnr", "tpr", "positive_rate"]])
        p_ci = fair.get("pediatric_disparity_ci", {}).get("ci_95", ["-", "-"])
        st.info(f"Pediatric (<18) Underdiagnosis Disparity: **{fair.get('underdiagnosis_ratio_pediatric_vs_adult', 'N/A')}x** (95% Bootstrap CI: [{p_ci[0]}, {p_ci[1]}])")

    with c_sex:
        st.markdown("**Sex Slices**")
        sex_data = slices.get("sex", {})
        if sex_data:
            st.table(pd.DataFrame(sex_data).T[["sample_count", "fnr", "tpr", "positive_rate"]])
        f_ci = fair.get("female_disparity_ci", {}).get("ci_95", ["-", "-"])
        st.info(f"Female Underdiagnosis Disparity: **{fair.get('underdiagnosis_ratio_female_vs_male', 'N/A')}x** (95% Bootstrap CI: [{f_ci[0]}, {f_ci[1]}])")

    st.subheader("Intersectional Slicing (Sex x Age)")
    inter_data = slices.get("intersectional_sex_age", {})
    if inter_data:
        st.dataframe(pd.DataFrame(inter_data).T[["sample_count", "fnr", "tpr", "positive_rate"]], use_container_width=True)

    st.subheader("Site Latent Covariate Shift Detection")
    st.write({
        "Adversarial Domain Classifier AUROC": fair.get("domain_shift_classifier_auroc", 0.5),
        "Maximum Mean Discrepancy (MMD) Distance": fair.get("latent_mmd_distance", 0.0),
        "Two-Sample KS-Test Min p-value": fair.get("ks_drift_p_value", 1.0),
        "Site Divergence Alarm Triggered": fair.get("site_divergence_detected", False),
    })

# ---------------------------------------------------------------------------
# Tab 4: Calibration & Silent Failures
# ---------------------------------------------------------------------------
with tab_cal:
    st.subheader("Model Calibration & Reliability Diagram (Guo et al. 2017 | Nixon et al. 2019)")
    rel_bins = cal.get("reliability_bins", [])
    if rel_bins:
        df_rel = pd.DataFrame(rel_bins)
        st.bar_chart(df_rel.set_index("range")[["confidence", "accuracy"]])
        st.dataframe(df_rel, use_container_width=True)

    c_cal1, c_cal2 = st.columns(2)
    with c_cal1:
        st.metric("Expected Calibration Error (ECE)", f"{cal.get('expected_calibration_error', 0)*100:.2f}%")
        st.metric("Adaptive Calibration Error (ACE)", f"{cal.get('adaptive_calibration_error', 0)*100:.2f}%")
        cc_ece = cal.get("class_conditional_ece", {})
        st.write(f"Class-Conditional ECE: Pos={cc_ece.get('positive_pathology_ece', 0)*100:.1f}% | Neg={cc_ece.get('negative_pathology_ece', 0)*100:.1f}%")

    with c_cal2:
        t_diag = cal.get("temperature_scaling_diagnostic", {})
        st.metric("Optimal Temperature (T*)", f"{t_diag.get('optimal_temperature_T', 1.0)}")
        st.write(f"Overconfidence Flag: **{t_diag.get('is_overconfident', False)}**")
        st.write(f"Post-Recalibration ECE: **{t_diag.get('post_recalibration_ece', 0)*100:.2f}%**")

    st.subheader("Dangerous Overconfidence Quadrant (Conf >= 0.85 Misses)")
    silent = cal.get("critical_silent_failures", [])
    if silent:
        st.error(f"🚨 {len(silent)} Critical Silent False-Negative Miss(es) Detected!")
        st.table(pd.DataFrame(silent))
    else:
        st.success("Zero high-confidence false negatives detected in validation cohort.")

# ---------------------------------------------------------------------------
# Tab 5: PCCP Regulatory Plan & Export
# ---------------------------------------------------------------------------
with tab_pccp:
    st.subheader("FDA SaMD Predetermined Change Control Plan (PCCP)")
    st.markdown("""
    In accordance with FDA SaMD guidance, TrustCheck establishes explicit automated runtime guardrails for PACS deployment:
    """)
    pccp_df = pd.DataFrame([
        {"Vector": "Hardware Decay", "PACS Runtime Trigger": "Image contrast attenuation drop > 15%", "Mandated Safeguard": "Automated Dual-Read Fallback: AI triage suspended; dual radiologist sign-off."},
        {"Vector": "Demographic Drift", "PACS Runtime Trigger": "Sliding window (N=500) KS drift p < 0.01 / MMD > 0.40", "Mandated Safeguard": "Institutional Ethics Review: Site-specific bias audit triggered."},
        {"Vector": "Calibration Drift", "PACS Runtime Trigger": "Single high-confidence FN miss (Conf > 0.90, Label = 1)", "Mandated Safeguard": "Scanner Node Quarantine: AI inference suspended on originating imaging unit."},
    ])
    st.table(pccp_df)

    pdf_path = os.path.join(os.path.dirname(telemetry_file), "audit_certificate.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button(
            label="📥 Download Official FDA/CDSCO SaMD Safety Dossier (PDF)",
            data=pdf_bytes,
            file_name="trustcheck_samd_audit_dossier.pdf",
            mime="application/pdf",
        )
    else:
        st.info("PDF dossier can be generated via `python audit_runner.py`.")
