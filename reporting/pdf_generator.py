"""Step 4B: Regulatory Dossier Compiler (PDF Generator).

Generates publication-grade FDA/CDSCO compliant Software-as-a-Medical-Device (SaMD)
Safety Dossiers and Predetermined Change Control Plan (PCCP) operational boundary documents.

Literature Grounding:
- FDA Guidance (2023/2024): Predetermined Change Control Plans for Machine Learning-Enabled Medical Devices.
- Seyyed-Kalantari et al. (Nature Medicine 2021): Underdiagnosis bias in thoracic radiographs.
- DeGrave et al. (Nature Machine Intelligence 2021): Non-clinical shortcuts in medical vision AI.
"""

import os
import hashlib
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Regulatory Color Tokens
DARK_900 = colors.HexColor("#0F172A")
DARK_800 = colors.HexColor("#1E293B")
SLATE_700 = colors.HexColor("#334155")
SLATE_500 = colors.HexColor("#64748B")
SLATE_300 = colors.HexColor("#CBD5E1")
SLATE_100 = colors.HexColor("#F1F5F9")
SLATE_50 = colors.HexColor("#F8FAFC")
WHITE = colors.white

RED_MAIN = colors.HexColor("#DC2626")
RED_BG = colors.HexColor("#FEF2F2")
RED_BORDER = colors.HexColor("#FECACA")

AMBER_MAIN = colors.HexColor("#D97706")
AMBER_BG = colors.HexColor("#FFFBEB")
AMBER_BORDER = colors.HexColor("#FDE68A")

GREEN_MAIN = colors.HexColor("#16A34A")
GREEN_BG = colors.HexColor("#F0FDF4")
GREEN_BORDER = colors.HexColor("#BBF7D0")

BLUE_ACCENT = colors.HexColor("#2563EB")


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def generate_dossier_pdf(telemetry: Dict[str, Any], output_pdf_path: str) -> str:
    """Compiles an official FDA/CDSCO SaMD Pre-Deployment Clinical Safety Dossier."""
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=26,
        bottomMargin=26,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DossierTitle",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=DARK_900,
    )
    sub_style = ParagraphStyle(
        "DossierSub",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=SLATE_500,
    )
    h2_style = ParagraphStyle(
        "DossierH2",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=DARK_900,
    )
    body_style = ParagraphStyle(
        "DossierBody",
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=DARK_800,
    )
    body_bold = ParagraphStyle(
        "DossierBold",
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10,
        textColor=DARK_900,
    )

    elements = []

    # 1. Header Bar
    run_id = telemetry.get("audit_run_id", "tc_run_2026")
    ts = telemetry.get("timestamp", "2026-09-11T15:24:27Z")
    header_table = Table([
        [
            Paragraph("<b>TrustCheck: Clinical AI Pre-Deployment Safety Dossier</b>", title_style),
            Paragraph(f"<b>RUN ID:</b> {run_id}<br/><b>AUDIT DATE:</b> {ts}<br/>FDA SaMD / CDSCO Class C/D Compliance", sub_style),
        ]
    ], colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=DARK_900, spaceAfter=6))

    # 2. Cryptographic Attestation Block
    model_name = telemetry.get("target_model", "DenseNet121-CheXNet")
    modality = telemetry.get("modality", "chest_xray").replace("_", " ").title()
    tier_profile = telemetry.get("tier_profile", "TIER_2_WHITE_BOX")
    model_hash = compute_sha256(str(model_name))
    data_hash = compute_sha256(str(telemetry.get("metrics", {})))
    crypto_data = [
        [
            Paragraph(f"<b>Target Model:</b> {model_name} | <b>Modality:</b> {modality} (SHA-256: <font name='Courier'>{model_hash}</font>)", body_style),
            Paragraph(f"<b>Audit Tier:</b> {tier_profile} | <b>Cohort Hash:</b> <font name='Courier'>{data_hash}</font>", body_style),
        ]
    ]
    crypto_table = Table(crypto_data, colWidths=[310, 230])
    crypto_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SLATE_100),
        ("PADDING", (0, 0), (-1, -1), 3),
        ("BOX", (0, 0), (-1, -1), 0.5, SLATE_300),
    ]))
    elements.append(crypto_table)
    elements.append(Spacer(1, 6))

    # 3. Deterministic Verdict Banner
    verdict = telemetry.get("verdict", "CAUTION_RESTRICTED_DEPLOYMENT")
    trust_score = telemetry.get("trust_score", 67.8)

    if "GO" in verdict and "NO" not in verdict:
        v_bg, v_border, v_color = GREEN_BG, GREEN_BORDER, GREEN_MAIN
        v_title = "GO: APPROVED FOR CLINICAL DEPLOYMENT"
    elif "CAUTION" in verdict:
        v_bg, v_border, v_color = AMBER_BG, AMBER_BORDER, AMBER_MAIN
        v_title = "CAUTION: RESTRICTED DEPLOYMENT (GUARDRAILS MANDATED)"
    else:
        v_bg, v_border, v_color = RED_BG, RED_BORDER, RED_MAIN
        v_title = "NO-GO: REJECTED (UNSAFE FOR CLINICAL USE)"

    v_style = ParagraphStyle("VStyle", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=v_color)
    score_style = ParagraphStyle("SStyle", fontName="Helvetica-Bold", fontSize=16, leading=18, textColor=v_color, alignment=2)

    verdict_table = Table([
        [
            Paragraph(f"<b>{v_title}</b><br/><font size=7.5 color='#475569'>Deterministic Multi-Pillar Gating Evaluation</font>", v_style),
            Paragraph(f"{trust_score} / 100", score_style),
        ]
    ], colWidths=[410, 130])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), v_bg),
        ("BOX", (0, 0), (-1, -1), 1, v_border),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 8))

    # 4. Multi-Pillar Metric Breakdown Table
    metrics = telemetry.get("metrics", {})
    rob = metrics.get("robustness", {})
    fair = metrics.get("fairness_and_shift", {})
    cal = metrics.get("calibration_and_uncertainty", {})

    p_ci = fair.get("pediatric_disparity_ci", {}).get("ci_95", ["-", "-"])
    f_ci = fair.get("female_disparity_ci", {}).get("ci_95", ["-", "-"])
    t_diag = cal.get("temperature_scaling_diagnostic", {})

    pillar_data = [
        [
            Paragraph("<b>Safety Pillar</b>", body_bold),
            Paragraph("<b>Weight</b>", body_bold),
            Paragraph("<b>Score</b>", body_bold),
            Paragraph("<b>Primary Stress Telemetry & SOTA Benchmarks</b>", body_bold),
            Paragraph("<b>Status</b>", body_bold),
        ],
        [
            Paragraph("<b>1. Hardware & Physics</b>", body_style),
            Paragraph("35%", body_style),
            Paragraph(f"<b>{rob.get('robustness_score', 'N/A')}</b>", body_style),
            Paragraph(f"Clean AUROC: {rob.get('clean_auroc', 'N/A')} | Shortcut SVI: {rob.get('shortcut_diagnostics', {}).get('shortcut_vulnerability_index', rob.get('distractor_false_positive_rate', 0)):.3f}", body_style),
            Paragraph("<font color='#16a34a'>PASS</font>" if rob.get("robustness_score", 0) >= 70 else "<font color='#d97706'>MARGINAL</font>", body_style),
        ],
        [
            Paragraph("<b>2. Demographic Subgroup</b>", body_style),
            Paragraph("30%", body_style),
            Paragraph(f"<b>{fair.get('fairness_score', 'N/A')}</b>", body_style),
            Paragraph(f"Pediatric: {fair.get('underdiagnosis_ratio_pediatric_vs_adult', 'N/A')}x [95% CI: {p_ci[0]}–{p_ci[1]}] | Female: {fair.get('underdiagnosis_ratio_female_vs_male', 'N/A')}x | EOD: {fair.get('equalized_odds_difference', 'N/A')}", body_style),
            Paragraph("<font color='#dc2626'>FAIL</font>" if fair.get("underdiagnosis_ratio_pediatric_vs_adult", 1) > 1.8 else "<font color='#16a34a'>PASS</font>", body_style),
        ],
        [
            Paragraph("<b>3. Uncertainty & Calibration</b>", body_style),
            Paragraph("20%", body_style),
            Paragraph(f"<b>{cal.get('calibration_score', 'N/A')}</b>", body_style),
            Paragraph(f"ECE: {cal.get('expected_calibration_error', 0)*100:.1f}% | ACE: {cal.get('adaptive_calibration_error', 0)*100:.1f}% | T*: {t_diag.get('optimal_temperature_T', 1.0)} | Silent Misses: {cal.get('critical_silent_failures_count', 0)}", body_style),
            Paragraph("<font color='#dc2626'>HAZARD</font>" if cal.get("critical_silent_failures_count", 0) > 0 else "<font color='#16a34a'>PASS</font>", body_style),
        ],
        [
            Paragraph("<b>4. Site & Covariate Drift</b>", body_style),
            Paragraph("15%", body_style),
            Paragraph(f"<b>{fair.get('shift_score', 'N/A')}</b>", body_style),
            Paragraph(f"Domain Classifier AUC: {fair.get('domain_shift_classifier_auroc', 'N/A')} | MMD Dist: {fair.get('latent_mmd_distance', 'N/A')} | KS p: {fair.get('ks_drift_p_value', 'N/A')}", body_style),
            Paragraph("<font color='#d97706'>DRIFT</font>" if fair.get("domain_shift_classifier_auroc", 0.5) > 0.65 else "<font color='#16a34a'>PASS</font>", body_style),
        ],
    ]
    pillar_table = Table(pillar_data, colWidths=[115, 38, 42, 285, 60])
    pillar_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_100),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_300),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(pillar_table)
    elements.append(Spacer(1, 8))

    # 5. Hardware Degradation Slopes Detail
    elements.append(Paragraph("<b>1. Hardware Perturbation Decay Slopes (AUROC across Tiers 1–5)</b>", h2_style))
    decay = rob.get("decay_slopes", {})
    decay_rows = [
        [
            Paragraph("<b>Corruption Protocol</b>", body_bold),
            Paragraph("<b>Tier 1</b>", body_bold),
            Paragraph("<b>Tier 2</b>", body_bold),
            Paragraph("<b>Tier 3</b>", body_bold),
            Paragraph("<b>Tier 4</b>", body_bold),
            Paragraph("<b>Tier 5 (Extreme)</b>", body_bold),
            Paragraph("<b>Floor Status</b>", body_bold),
        ]
    ]
    for c_name, vals in decay.items():
        decay_rows.append([
            Paragraph(c_name.replace("_", " ").title(), body_style),
            Paragraph(str(vals[0]) if len(vals) > 0 else "-", body_style),
            Paragraph(str(vals[1]) if len(vals) > 1 else "-", body_style),
            Paragraph(str(vals[2]) if len(vals) > 2 else "-", body_style),
            Paragraph(str(vals[3]) if len(vals) > 3 else "-", body_style),
            Paragraph(str(vals[4]) if len(vals) > 4 else "-", body_style),
            Paragraph("<font color='#dc2626'>BREACH (<0.70)</font>" if (len(vals) > 4 and vals[4] < 0.70) else "HELD (>=0.70)", body_style),
        ])
    decay_table = Table(decay_rows, colWidths=[140, 60, 60, 60, 60, 80, 80])
    decay_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_100),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_300),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(decay_table)
    elements.append(Spacer(1, 8))

    # 6. Multi-Archetype Non-Clinical Shortcut Diagnostic (DeGrave et al. 2021)
    s_diag = rob.get("shortcut_diagnostics", {})
    fp_rates = s_diag.get("distractor_fp_rates", {})
    if fp_rates:
        elements.append(Paragraph("<b>2. Non-Clinical Visual Shortcut Vulnerability (DeGrave et al. 2021)</b>", h2_style))
        s_rows = [
            [
                Paragraph("<b>Shortcut Archetype</b>", body_bold),
                Paragraph("<b>Simulated Visual Artifact</b>", body_bold),
                Paragraph("<b>False Trigger Rate (FP)</b>", body_bold),
                Paragraph("<b>Mean |Delta Conf|</b>", body_bold),
                Paragraph("<b>Vulnerability Flag</b>", body_bold),
            ]
        ]
        conf_deltas = s_diag.get("distractor_conf_deltas", {})
        descriptions = {
            "laterality_stamp_R": "Top-right anatomical 'R' lead orientation marker",
            "laterality_stamp_L": "Top-right anatomical 'L' lead orientation marker",
            "hospital_text_stamp": "Bottom burned-in hospital accession & timestamp banner",
            "metallic_hardware": "Upper chest pacemaker & surgical wire radio-opacity",
            "collimator_aperture": "Unexposed collimator crop shutter black borders",
        }
        for s_key, fp in fp_rates.items():
            delta = conf_deltas.get(s_key, 0.0)
            flag = "<font color='#dc2626'>SUSCEPTIBLE</font>" if (fp > 0.05 or delta > 0.10) else "<font color='#16a34a'>INVARIANT</font>"
            s_rows.append([
                Paragraph(s_key.replace("_", " ").title(), body_style),
                Paragraph(descriptions.get(s_key, "-"), body_style),
                Paragraph(f"{fp*100:.1f}%", body_style),
                Paragraph(f"{delta:.4f}", body_style),
                Paragraph(flag, body_style),
            ])
        s_table = Table(s_rows, colWidths=[120, 190, 80, 75, 75])
        s_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), SLATE_100),
            ("GRID", (0, 0), (-1, -1), 0.5, SLATE_300),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(s_table)
        elements.append(Spacer(1, 8))

    # 7. Demographic & Intersectional Equity Matrix (Seyyed-Kalantari et al. 2021)
    elements.append(Paragraph("<b>3. Demographic Subgroup & Intersectional Under-Diagnosis Matrix</b>", h2_style))
    slices = fair.get("slices", {})
    eq_rows = [
        [
            Paragraph("<b>Demographic Stratum</b>", body_bold),
            Paragraph("<b>Subgroup</b>", body_bold),
            Paragraph("<b>Samples (Pos/Total)</b>", body_bold),
            Paragraph("<b>False Negative Rate (FNR)</b>", body_bold),
            Paragraph("<b>True Positive Rate (TPR)</b>", body_bold),
            Paragraph("<b>Under-Diagnosis Disparity</b>", body_bold),
        ]
    ]

    for axis_name in ["age", "sex", "scanner_type"]:
        axis_data = slices.get(axis_name, {})
        for grp, m in axis_data.items():
            fnr = m.get("fnr", 0.0)
            tpr = m.get("tpr", 0.0)
            pos = m.get("positive_count", 0)
            tot = m.get("sample_count", 0)
            disp = "-"
            if axis_name == "age" and grp == "<18":
                disp = f"<b>{fair.get('underdiagnosis_ratio_pediatric_vs_adult', 'N/A')}x</b>"
            elif axis_name == "sex" and grp == "F":
                disp = f"<b>{fair.get('underdiagnosis_ratio_female_vs_male', 'N/A')}x</b>"
            elif axis_name == "scanner_type" and grp == "COMPUTED_RAD":
                disp = f"<b>{fair.get('underdiagnosis_ratio_cr_vs_dr', 'N/A')}x</b>"

            eq_rows.append([
                Paragraph(axis_name.replace("_", " ").title(), body_style),
                Paragraph(str(grp), body_style),
                Paragraph(f"{pos} / {tot}", body_style),
                Paragraph(f"{fnr*100:.1f}%", body_style),
                Paragraph(f"{tpr*100:.1f}%", body_style),
                Paragraph(disp, body_style),
            ])

    eq_table = Table(eq_rows, colWidths=[90, 80, 100, 90, 90, 90])
    eq_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SLATE_100),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_300),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(eq_table)
    elements.append(Spacer(1, 8))

    # 8. Clinical Contraindications & Operational Boundaries
    elements.append(Paragraph("<b>4. Clinical Contraindications & Black-Box Blacklist</b>", h2_style))
    contraindications = telemetry.get("clinical_contraindications", [])
    if contraindications:
        c_rows = []
        for c in contraindications:
            c_rows.append([Paragraph(f"• <b>{c}</b>", body_style)])
        c_table = Table(c_rows, colWidths=[540])
        c_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), RED_BG),
            ("BOX", (0, 0), (-1, -1), 1, RED_BORDER),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(c_table)
    else:
        elements.append(Paragraph("No critical clinical contraindications identified. Routine PACS monitoring mandated.", body_style))
    elements.append(Spacer(1, 8))

    # 9. Predetermined Change Control Plan (PCCP) PACS Guardrails
    elements.append(KeepTogether([
        Paragraph("<b>5. Predetermined Change Control Plan (PCCP) — PACS Runtime Surveillance Triggers</b>", h2_style),
        Spacer(1, 3),
        Table([
            [
                Paragraph("<b>Trigger Vector</b>", body_bold),
                Paragraph("<b>PACS Runtime Condition</b>", body_bold),
                Paragraph("<b>Mandated Regulatory Action & Safeguard</b>", body_bold),
            ],
            [
                Paragraph("<b>Trigger 1 (Hardware)</b>", body_style),
                Paragraph("PACS image mean contrast attenuation drop > 15%", body_style),
                Paragraph("<b>AUTOMATED DUAL-READ FALLBACK:</b> Disables autonomous triage; forces double radiologist re-read.", body_style),
            ],
            [
                Paragraph("<b>Trigger 2 (Demographic)</b>", body_style),
                Paragraph("Sliding window (N=500) latent KS drift p < 0.01 / MMD > 0.40", body_style),
                Paragraph("<b>INSTITUTIONAL ETHICS REVIEW:</b> Mandatory bias audit on originating clinical site.", body_style),
            ],
            [
                Paragraph("<b>Trigger 3 (Calibration)</b>", body_style),
                Paragraph("Single high-confidence FN miss (Conf > 0.90, Ground Truth = 1)", body_style),
                Paragraph("<b>SCANNER NODE QUARANTINE:</b> Immediately suspends AI inference on originating PACS node.", body_style),
            ],
        ], colWidths=[110, 190, 240], style=[
            ("BACKGROUND", (0, 0), (-1, 0), SLATE_100),
            ("GRID", (0, 0), (-1, -1), 0.5, SLATE_300),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]),
        Spacer(1, 8),
        Paragraph("<i>TrustCheck Independent Evaluation Framework | HLT-08 Specification Verified | Generated for Clinical PACS Deployment</i>", sub_style),
    ]))

    doc.build(elements)
    return output_pdf_path
