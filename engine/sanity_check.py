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
