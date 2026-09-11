"""Clinical Utility, Decision Curve Analysis (DCA), and Prevalence Shift Engine.

Literature Grounding:
- Vickers & Elkin (BMJ / Ann Intern Med 2006): Decision Curve Analysis (DCA) and Net Benefit.
- TRIPOD Statement: Recommends DCA over raw accuracy/AUROC for clinical prediction models.
- Wong et al. (JAMA Internal Medicine 2021): Epic Sepsis Model failure under prevalence shift and alert fatigue.
- Youden (Cancer 1950): Index for optimal clinical diagnostic cutoffs.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


def compute_decision_curve_analysis(
    labels: np.ndarray,
    confidences: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Calculates Decision Curve Analysis (DCA) Net Benefit across clinical decision thresholds.
    
    Net Benefit (p_t) = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    Treat All Net Benefit = (P / N) - (1 - P / N) * (p_t / (1 - p_t))
    Treat None Net Benefit = 0.0
    
    Vickers & Elkin (2006).
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    prevalence = n_pos / max(1, n_total)

    dca_points = []
    has_positive_net_benefit = True

    for p_t in thresholds:
        if p_t >= 1.0 or p_t <= 0.0:
            continue
        
        # Binarize at clinical decision threshold
        preds_at_pt = (confidences >= p_t).astype(int)
        tp = int(np.sum((preds_at_pt == 1) & (labels == 1)))
        fp = int(np.sum((preds_at_pt == 1) & (labels == 0)))

        # Net Benefit of Model
        weight = p_t / (1.0 - p_t)
        nb_model = (tp / n_total) - (fp / n_total) * weight

        # Net Benefit of Default Strategies
        nb_treat_all = prevalence - (1.0 - prevalence) * weight
        nb_treat_none = 0.0

        is_superior = bool(nb_model > max(nb_treat_all, nb_treat_none))
        if p_t in [0.10, 0.20] and not is_superior:
            has_positive_net_benefit = False

        dca_points.append({
            "threshold_probability": round(p_t, 3),
            "net_benefit_model": round(float(nb_model), 4),
            "net_benefit_treat_all": round(float(nb_treat_all), 4),
            "net_benefit_treat_none": 0.0,
            "superior_to_defaults": is_superior,
        })

    return {
        "dca_points": dca_points,
        "clinical_utility_verified": has_positive_net_benefit,
        "prevalence": round(prevalence, 4),
    }


def compute_prevalence_shift_ladder(
    sensitivity: float,
    specificity: float,
    prevalence_ladder: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Simulates Positive Predictive Value (PPV) collapse under prevalence shifts (Wong et al. 2021).
    
    PPV(pi) = (Sens * pi) / (Sens * pi + (1 - Spec) * (1 - pi))
    NPV(pi) = (Spec * (1 - pi)) / (Spec * (1 - pi) + (1 - Sens) * pi)
    """
    if prevalence_ladder is None:
        # From rare outpatient screening (2%) to high-acuity tertiary ICU (40%)
        prevalence_ladder = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]

    ladder_results = []
    collapse_point = None

    for prev in prevalence_ladder:
        denom_ppv = (sensitivity * prev) + ((1.0 - specificity) * (1.0 - prev))
        ppv = (sensitivity * prev) / max(1e-6, denom_ppv)
        denom_npv = (specificity * (1.0 - prev)) + ((1.0 - sensitivity) * prev)
        npv = (specificity * (1.0 - prev)) / max(1e-6, denom_npv)

        # False alarm burden: FP / (TP + FP) = 1 - PPV
        alert_fatigue_rate = 1.0 - ppv

        if ppv < 0.50 and collapse_point is None:
            collapse_point = prev

        ladder_results.append({
            "disease_prevalence": round(prev, 3),
            "ppv": round(float(ppv), 4),
            "npv": round(float(npv), 4),
            "alert_fatigue_false_alarm_pct": round(float(alert_fatigue_rate) * 100.0, 1),
            "clinical_alert_burden": "HIGH_ALERT_FATIGUE" if alert_fatigue_rate > 0.60 else "ACCEPTABLE",
        })

    return {
        "prevalence_ladder": ladder_results,
        "prevalence_collapse_point": collapse_point,
        "collapse_detected": bool(collapse_point is not None and collapse_point >= 0.05),
    }


def find_clinical_operating_points(
    labels: np.ndarray,
    confidences: np.ndarray,
) -> Dict[str, Any]:
    """Calculates clinically actionable operating thresholds beyond arbitrary 0.50 cutoff.
    
    1. Optimal Youden's J threshold: max(Sensitivity + Specificity - 1)
    2. High-Sensitivity Triage threshold: Operating cutoff ensuring >= 95% recall (rule-out mode)
    """
    if len(set(labels)) < 2:
        return {
            "youden_optimal_threshold": 0.50,
            "youden_j_statistic": 0.0,
            "high_sensitivity_95_threshold": 0.50,
        }

    fpr, tpr, thresholds = roc_curve(labels, confidences)
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_j_idx])

    # Find cutoff where sensitivity (tpr) >= 0.95
    sens_95_indices = np.where(tpr >= 0.95)[0]
    if len(sens_95_indices) > 0:
        thresh_95 = float(thresholds[sens_95_indices[0]])
    else:
        thresh_95 = float(thresholds[np.argmax(tpr)])

    return {
        "youden_optimal_threshold": round(best_threshold, 3),
        "youden_j_statistic": round(float(j_scores[best_j_idx]), 3),
        "youden_sensitivity": round(float(tpr[best_j_idx]), 3),
        "youden_specificity": round(float(1.0 - fpr[best_j_idx]), 3),
        "high_sensitivity_95_threshold": round(thresh_95, 3),
    }
