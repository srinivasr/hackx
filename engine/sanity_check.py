from typing import Dict, Any, List
import numpy as np


def verify_lesion_classification_consensus(
    predicted_grade: int,
    confidence: float,
    biomarkers: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluates if the model's classification decision is consistent with physical lesion biomarkers.
    
    Catches silent false negatives where shortcut learning predicts normal retina
    despite active microvascular pathology.
    """
    num_mas = int(biomarkers.get("microaneurysms", 0))
    exudate_pct = float(biomarkers.get("exudate_area_pct", 0.0))
    hemorrhage_quads = int(biomarkers.get("hemorrhage_quadrants", 0))
    soft_exudate_pct = float(biomarkers.get("soft_exudate_area_pct", 0.0))

    # Evidence-based minimum severity
    if hemorrhage_quads >= 4:
        physical_grade = 3
        evidence = f"Hemorrhages present in all 4 quadrants (meets 4-2-1 severe NPDR criterion)."
    elif exudate_pct > 0.05 or (num_mas > 0 and hemorrhage_quads >= 1) or soft_exudate_pct > 0.5:
        physical_grade = 2
        evidence = f"Active exudates ({exudate_pct:.2f}%) and microaneurysms ({num_mas} detected)."
    elif num_mas > 0 or hemorrhage_quads > 0 or soft_exudate_pct > 0.1:
        physical_grade = 1
        evidence = f"Microaneurysm count = {num_mas}; early vascular leakage present."
    else:
        physical_grade = 0
        evidence = "No visible microvascular lesions detected."

    # Discrepancy Analysis
    grade_gap = predicted_grade - physical_grade
    is_safe = True
    violation_type = "NONE"
    verdict_text = "PASSED: Classification prediction aligns with anatomical lesion evidence."

    if predicted_grade == 0 and physical_grade >= 2:
        is_safe = False
        violation_type = "CRITICAL_SILENT_FALSE_NEGATIVE"
        verdict_text = (
            f"DANGER: Candidate model predicted Grade 0 (Normal) with {confidence:.1f}% confidence, "
            f"failing to detect referable disease verified by {num_mas} microaneurysms and {exudate_pct:.2f}% exudates."
        )
    elif predicted_grade == 0 and physical_grade == 1:
        is_safe = False
        violation_type = "MILD_UNDERCLASSIFICATION"
        verdict_text = (
            f"WARNING: Model missed early microaneurysms ({num_mas} detected). "
            "Patient may not be scheduled for routine monitoring."
        )
    elif predicted_grade >= 3 and physical_grade == 0:
        is_safe = False
        violation_type = "FALSE_POSITIVE_PANIC"
        verdict_text = (
            f"OVERCLASSIFICATION: Model predicted severe disease (Grade {predicted_grade}), "
            "yet anatomical segmentation reveals zero physical lesions."
        )

    return {
        "predicted_grade": predicted_grade,
        "physical_biomarker_grade": physical_grade,
        "is_safe": is_safe,
        "violation_type": violation_type,
        "grade_discrepancy": grade_gap,
        "evidence_summary": evidence,
        "verdict": verdict_text,
    }


def compute_subgroup_fairness(
    predictions: List[int],
    ground_truths: List[int],
    demographics: List[str]
) -> Dict[str, Any]:
    """Computes fairness metrics (FPR/FNR parity, Equalized Odds) across demographic groups."""
    subgroups = list(set(demographics))
    metrics_by_group = {}
    
    total_fpr_diff = 0.0
    total_fnr_diff = 0.0
    
    overall_fpr = 0.0
    overall_fnr = 0.0
    
    # Simple binary classification mapping for fairness (0=Normal, >0=Abnormal)
    y_pred = [1 if p > 0 else 0 for p in predictions]
    y_true = [1 if t > 0 else 0 for t in ground_truths]
    
    # Calculate overall metrics
    fp_overall = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    tn_overall = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
    fn_overall = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    tp_overall = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    
    overall_fpr = fp_overall / (fp_overall + tn_overall) if (fp_overall + tn_overall) > 0 else 0.0
    overall_fnr = fn_overall / (fn_overall + tp_overall) if (fn_overall + tp_overall) > 0 else 0.0
    
    for group in subgroups:
        group_indices = [i for i, d in enumerate(demographics) if d == group]
        g_pred = [y_pred[i] for i in group_indices]
        g_true = [y_true[i] for i in group_indices]
        
        fp = sum(1 for p, t in zip(g_pred, g_true) if p == 1 and t == 0)
        tn = sum(1 for p, t in zip(g_pred, g_true) if p == 0 and t == 0)
        fn = sum(1 for p, t in zip(g_pred, g_true) if p == 0 and t == 1)
        tp = sum(1 for p, t in zip(g_pred, g_true) if p == 1 and t == 1)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        metrics_by_group[group] = {
            "FPR": fpr,
            "FNR": fnr,
            "sample_count": len(group_indices)
        }
        
        total_fpr_diff += abs(fpr - overall_fpr)
        total_fnr_diff += abs(fnr - overall_fnr)
        
    avg_fpr_disparity = total_fpr_diff / len(subgroups) if subgroups else 0.0
    avg_fnr_disparity = total_fnr_diff / len(subgroups) if subgroups else 0.0
    
    is_fair = avg_fpr_disparity <= 0.1 and avg_fnr_disparity <= 0.1
    
    return {
        "subgroups": metrics_by_group,
        "average_fpr_disparity": avg_fpr_disparity,
        "average_fnr_disparity": avg_fnr_disparity,
        "passes_equalized_odds": is_fair,
        "status": "PASSED" if is_fair else "FAILED - High Demographic Disparity"
    }
