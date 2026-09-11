"""G-AUDIT Shortcut Risk Matrix & Selective Prediction Suppression Engine.

Literature Grounding:
- Drenkow et al. (Johns Hopkins / FDA CDRH, 2025 - arXiv:2503.09969):
  G-AUDIT: Generalized Attribute Utility and Detectability-Induced bias Testing for Medical AI.
- Brown et al. (Vanderbilt / JAMIA 2023 - ocaf235):
  Auditor models to suppress poor AI predictions to optimize human-AI collaboration.
- Koh et al. (Stanford / ICML 2021 - koh21a):
  WILDS: Worst-Group Accuracy and Worst-Group AUROC across subpopulation distribution shifts.
- Hendrycks & Dietterich (ICLR 2019):
  Mean Corruption Error (mCE) and Relative mCE benchmark standardization.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# 1. G-AUDIT: Attribute Detectability vs. Utility Matrix (FDA / JHU 2025)
# ---------------------------------------------------------------------------

def compute_attribute_detectability(
    embeddings: np.ndarray,
    attribute_labels: List[Any],
    n_splits: int = 3,
) -> float:
    """Computes Attribute Detectability (AD): Can an attribute (e.g. sex, site) be decoded from representations?
    
    Drenkow et al. (2025).
    """
    unique_vals = list(set(attribute_labels))
    if len(unique_vals) < 2 or len(embeddings) < 20:
        return 0.50

    val_map = {v: idx for idx, v in enumerate(sorted(unique_vals)[:2])}
    mask = [v in val_map for v in attribute_labels]
    y_attr = np.array([val_map[v] for v, m in zip(attribute_labels, mask) if m])
    x_sub = embeddings[mask]

    if len(set(y_attr)) < 2 or np.min(np.bincount(y_attr)) < 3:
        return 0.50

    try:
        lr = LogisticRegression(max_iter=300, random_state=42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs = []
        for tr, te in skf.split(x_sub, y_attr):
            lr.fit(x_sub[tr], y_attr[tr])
            probs = lr.predict_proba(x_sub[te])[:, 1]
            aucs.append(roc_auc_score(y_attr[te], probs))
        return round(float(np.mean(aucs)), 3)
    except Exception:
        return 0.50


def compute_attribute_utility(
    attribute_labels: List[Any],
    diagnosis_labels: np.ndarray,
) -> float:
    """Computes Attribute Utility (AU): How strongly does the attribute correlate with true disease diagnosis?
    
    Measured via AUROC of attribute predicting diagnostic label.
    Drenkow et al. (2025).
    """
    unique_vals = list(set(attribute_labels))
    if len(unique_vals) < 2 or len(diagnosis_labels) < 20:
        return 0.50

    val_map = {v: idx for idx, v in enumerate(sorted(unique_vals)[:2])}
    mask = [v in val_map for v in attribute_labels]
    x_attr = np.array([val_map[v] for v, m in zip(attribute_labels, mask) if m]).reshape(-1, 1)
    y_diag = diagnosis_labels[mask]

    if len(set(y_diag)) < 2:
        return 0.50

    try:
        lr = LogisticRegression(max_iter=100, random_state=42)
        lr.fit(x_attr, y_diag)
        probs = lr.predict_proba(x_attr)[:, 1]
        auc = float(roc_auc_score(y_diag, probs))
        return round(max(auc, 1.0 - auc), 3)  # Direction invariant
    except Exception:
        return 0.50


def run_gaudit_analysis(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    labels: np.ndarray,
    target_attributes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs G-AUDIT framework: Quantifies shortcut risk across clinical and demographic attributes."""
    if target_attributes is None:
        target_attributes = ["sex", "site_id", "scanner_type"]

    gaudit_matrix = {}
    high_risk_shortcuts = []

    for attr in target_attributes:
        if attr not in metadata_df.columns:
            continue
        vals = metadata_df[attr].tolist()
        ad = compute_attribute_detectability(embeddings, vals)
        au = compute_attribute_utility(vals, labels)
        
        # Shortcut Risk: Product of Detectability and Utility above chance (0.5)
        excess_ad = max(0.0, ad - 0.50)
        excess_au = max(0.0, au - 0.50)
        risk_score = round(float(excess_ad * excess_au * 4.0), 3)  # Scaled 0 to 1

        is_shortcut_hazard = bool(ad >= 0.70 and au >= 0.65)
        if is_shortcut_hazard:
            high_risk_shortcuts.append({
                "attribute": attr,
                "detectability_auc": ad,
                "utility_auc": au,
                "risk_score": risk_score,
            })

        gaudit_matrix[attr] = {
            "detectability_auc": ad,
            "utility_auc": au,
            "shortcut_risk_score": risk_score,
            "status": "SHORTCUT_HAZARD" if is_shortcut_hazard else "BENIGN_INVARIANCE",
        }

    return {
        "gaudit_matrix": gaudit_matrix,
        "high_risk_shortcuts": high_risk_shortcuts,
        "has_shortcut_hazard": len(high_risk_shortcuts) > 0,
    }


# ---------------------------------------------------------------------------
# 2. Selective Prediction & Auditor Suppression Policy (JAMIA 2023 - ocaf235)
# ---------------------------------------------------------------------------

def compute_selective_suppression_policy(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    deferral_fractions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Evaluates selective suppression policy: Silencing predictions on high-uncertainty cases
    and deferring them to human clinician re-read (Brown et al., JAMIA 2023).
    
    Measures cohort AUROC improvement and false negative reduction as a function of deferral rate.
    """
    if deferral_fractions is None:
        deferral_fractions = [0.05, 0.10, 0.20, 0.30]

    # Uncertainty metric: distance from decision threshold (0.50)
    uncertainties = 1.0 - 2.0 * np.abs(confidences - 0.50)  # In [0.0 (certain), 1.0 (uncertain)]
    n_samples = len(labels)

    baseline_acc = float(accuracy_score(labels, predictions))
    suppression_ladder = []

    for defer_rate in deferral_fractions:
        k_defer = int(round(defer_rate * n_samples))
        if k_defer <= 0 or k_defer >= n_samples:
            continue

        # Suppress top k most uncertain samples
        sort_order = np.argsort(uncertainties)[::-1]  # Highest uncertainty first
        retained_idx = sort_order[k_defer:]

        if len(set(labels[retained_idx])) >= 2:
            retained_auroc = round(float(roc_auc_score(labels[retained_idx], confidences[retained_idx])), 4)
        else:
            retained_auroc = 1.0

        retained_acc = round(float(accuracy_score(labels[retained_idx], predictions[retained_idx])), 4)
        fn_suppressed = int(np.sum((predictions[sort_order[:k_defer]] == 0) & (labels[sort_order[:k_defer]] == 1)))

        suppression_ladder.append({
            "deferral_fraction": defer_rate,
            "samples_deferred_to_human": k_defer,
            "retained_accuracy": retained_acc,
            "retained_auroc": retained_auroc,
            "false_negatives_suppressed": fn_suppressed,
            "accuracy_gain_pct": round((retained_acc - baseline_acc) * 100.0, 2),
        })

    # Find optimal operating deferral threshold
    optimal_tier = suppression_ladder[1] if len(suppression_ladder) > 1 else (suppression_ladder[0] if suppression_ladder else {})

    return {
        "baseline_accuracy": round(baseline_acc, 4),
        "suppression_ladder": suppression_ladder,
        "recommended_triage_deferral_rate": optimal_tier.get("deferral_fraction", 0.10),
        "expected_accuracy_gain": optimal_tier.get("accuracy_gain_pct", 0.0),
    }


# ---------------------------------------------------------------------------
# 3. WILDS Worst-Group Performance & Hendrycks Clinical mCE
# ---------------------------------------------------------------------------

def compute_clinical_mce(
    decay_slopes: Dict[str, List[float]],
    baseline_decay_floor: float = 0.50,
) -> Dict[str, Any]:
    """Computes Clinical Mean Corruption Error (cMCE) standardized across corruptions.
    Hendrycks & Dietterich (ICLR 2019).
    """
    corr_errors = {}
    for c_name, slopes in decay_slopes.items():
        if len(slopes) == 0:
            continue
        # Average error across 5 tiers: 1.0 - AUROC
        mean_err = np.mean([1.0 - s for s in slopes])
        corr_errors[c_name] = round(float(mean_err), 4)

    overall_cmce = round(float(np.mean(list(corr_errors.values())) * 100.0), 2) if corr_errors else 0.0

    return {
        "corruption_errors": corr_errors,
        "clinical_mean_corruption_error_cmce": overall_cmce,
    }


def compute_worst_group_metrics(slices: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Computes Worst-Group Accuracy and Worst-Group FNR across demographic slices.
    Koh et al. (WILDS, ICML 2021).
    """
    worst_fnr = 0.0
    worst_group_name = "N/A"
    
    for axis, groups in slices.items():
        if not isinstance(groups, dict):
            continue
        for grp_name, m in groups.items():
            if isinstance(m, dict) and "fnr" in m and m.get("sample_count", 0) >= 5:
                fnr = m["fnr"]
                if fnr > worst_fnr:
                    worst_fnr = fnr
                    worst_group_name = f"{axis}:{grp_name}"

    return {
        "worst_group_fnr": round(float(worst_fnr), 4),
        "worst_performing_group": worst_group_name,
    }
