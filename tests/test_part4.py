import pandas as pd
import numpy as np
from engines.fairness_engine import FairnessEngine
from sklearn.linear_model import LogisticRegression

def test_cartesian_intersection():
    df = pd.DataFrame({
        "age": [20, 70, 30, 80],
        "sex": ["M", "F", "F", "M"],
        "scanner_type": ["DR", "DR", "CR", "CR"],
        "site_id": ["A", "B", "A", "B"]
    })
    predictions = np.array([1, 0, 1, 0])
    confidences = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 1, 1, 0])
    
    engine = FairnessEngine()
    report = engine.slice_cohort(df, predictions, confidences, labels)
    
    assert "cartesian_intersection" in report["slices"]
    assert len(report["slices"]["cartesian_intersection"]) == 4

def test_optimize_thresholds():
    X = pd.DataFrame({"feat1": [0.1, 0.8, 0.2, 0.9]})
    y = pd.Series([0, 1, 0, 1])
    sensitive = pd.Series(["A", "A", "B", "B"])
    
    estimator = LogisticRegression()
    estimator.fit(X, y)
    
    engine = FairnessEngine()
    optimizer = engine.optimize_thresholds_equalized_odds(estimator, X, y, sensitive)
    
    # Fairlearn's optimizer returns an object that can predict
    preds = optimizer.predict(X, sensitive_features=sensitive)
    assert len(preds) == 4
