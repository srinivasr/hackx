from typing import Dict, List, Tuple, Any
import numpy as np


def compute_ece(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Computes Expected Calibration Error (ECE) and bin-wise reliability metrics.
    
    Args:
        probs: (N, C) softmax confidence probabilities.
        labels: (N,) ground-truth class indices.
        n_bins: Number of confidence bins (standard is 10).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    reliability_bins = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            gap = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += gap * prop_in_bin
            
            reliability_bins.append({
                "bin_lower": round(float(bin_lower), 2),
                "bin_upper": round(float(bin_upper), 2),
                "sample_count": int(np.sum(in_bin)),
                "avg_confidence": round(avg_confidence_in_bin, 4),
                "accuracy": round(accuracy_in_bin, 4),
                "gap": round(gap, 4),
            })
        else:
            reliability_bins.append({
                "bin_lower": round(float(bin_lower), 2),
                "bin_upper": round(float(bin_upper), 2),
                "sample_count": 0,
                "avg_confidence": round(float((bin_lower + bin_upper) / 2.0), 4),
                "accuracy": 0.0,
                "gap": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "ece_percent": round(float(ece) * 100, 2),
        "reliability_bins": reliability_bins,
        "calibration_risk": assess_calibration_risk(float(ece)),
    }


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes multi-class Brier score (mean squared error of probability vector)."""
    n_samples, n_classes = probs.shape
    one_hot = np.zeros((n_samples, n_classes))
    for i, l in enumerate(labels):
        one_hot[i, int(l)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def assess_calibration_risk(ece: float) -> str:
    """Classifies calibration into clinical risk categories."""
    if ece <= 0.05:
        return "SAFE: Model confidences accurately reflect true clinical likelihood (ECE <= 5%)."
    elif ece <= 0.12:
        return "MODERATE RISK: Minor overconfidence in borderline diagnostic grades."
    else:
        return "HIGH RISK: Severe overconfidence; model expresses high confidence on erroneous decisions."
