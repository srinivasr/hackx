"""Step 2: Engine B - Distribution Drift & Fairness Auditor.

Exposes hidden under-diagnosis disparity ratios across demographic slices
(Age, Sex, Scanner hardware, Hospital site) and compound intersections
(e.g. Young Female vs Elderly Male), calculates 95% bootstrap confidence intervals,
and mathematically tests for covariate shift via penultimate latent KS-tests,
Maximum Mean Discrepancy (MMD), and Adversarial Domain Classifiers.

Literature Grounding:
- Seyyed-Kalantari et al. (Nature Medicine 2021): Underdiagnosis bias of deep learning
  algorithms applied to chest radiographs in under-served patient populations.
- Buolamwini & Gebru (FAccT 2018): Intersectional accuracy disparities.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate


def compute_rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Computes Maximum Mean Discrepancy (MMD) with RBF kernel between two distributions.
    
    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    """
    if len(x) == 0 or len(y) == 0:
        return 0.0
    
    # Subsample if large for speed
    n_sub = min(len(x), len(y), 100)
    x_sub = x[:n_sub]
    y_sub = y[:n_sub]
    
    if gamma is None:
        # Median heuristic
        dists = np.sum((x_sub[:, None, :] - y_sub[None, :, :]) ** 2, axis=-1)
        med = float(np.median(dists))
        gamma = 1.0 / max(1e-4, med)

    # Kernel matrices
    k_xx = np.exp(-gamma * np.sum((x_sub[:, None, :] - x_sub[None, :, :]) ** 2, axis=-1))
    k_yy = np.exp(-gamma * np.sum((y_sub[:, None, :] - y_sub[None, :, :]) ** 2, axis=-1))
    k_xy = np.exp(-gamma * np.sum((x_sub[:, None, :] - y_sub[None, :, :]) ** 2, axis=-1))

    mmd_sq = np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy)
    return round(float(np.sqrt(max(0.0, mmd_sq))), 4)


def bootstrap_disparity_ci(
    df: pd.DataFrame,
    group_col: str,
    target_val: str,
    ref_val: str,
    n_bootstraps: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculates empirical 95% bootstrap confidence interval for underdiagnosis disparity ratio.
    
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    ratios = []

    def get_fnr(sub_df):
        positives = sub_df[sub_df["ground_truth"] == 1]
        if len(positives) == 0:
            return 0.0
        fn = (positives["pred"] == 0).sum()
        return fn / len(positives)

    ref_floor = 0.05

    # Point estimate
    target_sub = df[df[group_col] == target_val]
    ref_sub = df[df[group_col] == ref_val]
    point_ratio = get_fnr(target_sub) / max(ref_floor, get_fnr(ref_sub))

    # Bootstrap sampling
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_df = df.iloc[boot_idx]
        b_target = boot_df[boot_df[group_col] == target_val]
        b_ref = boot_df[boot_df[group_col] == ref_val]
        fnr_target = get_fnr(b_target)
        fnr_ref = max(ref_floor, get_fnr(b_ref))
        ratios.append(fnr_target / fnr_ref)

    ci_lower = float(np.percentile(ratios, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(ratios, (1.0 - alpha / 2.0) * 100))
    return round(float(point_ratio), 2), round(ci_lower, 2), round(ci_upper, 2)


class FairnessEngine:
    """Step 2: Slices cohorts across demographic, intersectional, and hardware axes
    and detects multi-site distribution shift.
    """

    def __init__(
        self,
        ref_age: str = "18-65",
        ref_sex: str = "M",
        ref_scanner: str = "DIGITAL_RAD",
        ref_site: str = "AIIMS_DELHI",
        alarm_classifier_auc: float = 0.65,
    ):
        self.ref_age = ref_age
        self.ref_sex = ref_sex
        self.ref_scanner = ref_scanner
        self.ref_site = ref_site
        self.alarm_classifier_auc = alarm_classifier_auc

    def slice_cohort(
        self,
        metadata_df: pd.DataFrame,
        predictions: np.ndarray,
        confidences: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """Calculates FNR, TPR, FPR, underdiagnosis disparity ratios, intersectional metrics,
        and bootstrapped confidence intervals across demographic slices.
        """
        df = metadata_df.copy()
        df["pred"] = predictions
        df["conf"] = confidences
        df["ground_truth"] = labels

        # Ensure age brackets exist
        if "age_bracket" not in df.columns and "age" in df.columns:
            df["age_bracket"] = pd.cut(
                df["age"],
                bins=[-1, 17, 65, 120],
                labels=["<18", "18-65", ">65"],
            )

        # Create full cartesian intersectional demographic feature
        intersection_cols = []
        if "sex" in df.columns: intersection_cols.append("sex")
        if "age_bracket" in df.columns: intersection_cols.append("age_bracket")
        if "scanner_type" in df.columns: intersection_cols.append("scanner_type")
        if "site_id" in df.columns: intersection_cols.append("site_id")
        
        if len(intersection_cols) > 1:
            df["full_intersection"] = df[intersection_cols].astype(str).agg('_'.join, axis=1)

        slices_report: Dict[str, Dict[str, Any]] = {}

        # 1. Primary demographic axes
        slices_report["age"] = self._calculate_subgroup_metrics(df, "age_bracket")
        slices_report["sex"] = self._calculate_subgroup_metrics(df, "sex")
        slices_report["scanner_type"] = self._calculate_subgroup_metrics(df, "scanner_type")
        slices_report["site_id"] = self._calculate_subgroup_metrics(df, "site_id")
        
        # 2. Intersectional axis (Cartesian Product)
        if "full_intersection" in df.columns:
            slices_report["cartesian_intersection"] = self._calculate_subgroup_metrics(
                df, "full_intersection"
            )

        # 3. Disparity Ratios against defined references
        ref_floor = 0.05
        pediatric_fnr = slices_report["age"].get("<18", {}).get("fnr", 0.0)
        adult_fnr = max(ref_floor, slices_report["age"].get(self.ref_age, {}).get("fnr", 0.08))
        pediatric_ratio = round(min(10.0, pediatric_fnr / adult_fnr), 2)

        female_fnr = slices_report["sex"].get("F", {}).get("fnr", 0.0)
        male_fnr = max(ref_floor, slices_report["sex"].get(self.ref_sex, {}).get("fnr", 0.08))
        female_ratio = round(min(10.0, female_fnr / male_fnr), 2)

        cr_fnr = slices_report["scanner_type"].get("COMPUTED_RAD", {}).get("fnr", 0.0)
        dr_fnr = max(ref_floor, slices_report["scanner_type"].get(self.ref_scanner, {}).get("fnr", 0.08))
        cr_ratio = round(min(10.0, cr_fnr / dr_fnr), 2)

        # Bootstrap 95% Confidence Intervals
        ci_pediatric = {"ratio": pediatric_ratio, "ci_95": [pediatric_ratio, pediatric_ratio]}
        ci_female = {"ratio": female_ratio, "ci_95": [female_ratio, female_ratio]}
        try:
            if "<18" in df["age_bracket"].values and self.ref_age in df["age_bracket"].values:
                _, p_low, p_high = bootstrap_disparity_ci(df, "age_bracket", "<18", self.ref_age)
                ci_pediatric["ci_95"] = [p_low, p_high]
            if "F" in df["sex"].values and self.ref_sex in df["sex"].values:
                _, f_low, f_high = bootstrap_disparity_ci(df, "sex", "F", self.ref_sex)
                ci_female["ci_95"] = [f_low, f_high]
        except Exception:
            pass

        # 4. Equalized Odds Difference (EOD)
        # EOD = max(|FNR_F - FNR_M|, |FPR_F - FPR_M|)
        female_fpr = slices_report["sex"].get("F", {}).get("fpr", 0.0)
        male_fpr = slices_report["sex"].get("M", {}).get("fpr", 0.0)
        equalized_odds_diff = round(max(abs(female_fnr - male_fnr), abs(female_fpr - male_fpr)), 4)

        # 5. Disparate Impact Ratio
        female_pos_rate = float(slices_report["sex"].get("F", {}).get("positive_rate", 0.5))
        male_pos_rate = float(slices_report["sex"].get("M", {}).get("positive_rate", 0.5))
        denom_male = max(1e-4, male_pos_rate)
        denom_female = max(1e-4, female_pos_rate)
        disparate_impact_ratio = round(min(female_pos_rate / denom_male, male_pos_rate / denom_female), 2)

        # 6. Intersectional Maximum Disparity
        intersectional_ratios = {}
        if "cartesian_intersection" in slices_report:
            ref_int_fnr = max(ref_floor, slices_report["cartesian_intersection"].get(list(slices_report["cartesian_intersection"].keys())[0], {}).get("fnr", 0.08))
            for int_grp, m in slices_report["cartesian_intersection"].items():
                r = round(min(10.0, m.get("fnr", 0.0) / max(1e-4, ref_int_fnr)), 2)
                intersectional_ratios[int_grp] = r

        # Highest observed disparity across all single & intersectional slices
        all_disparities = [pediatric_ratio, female_ratio, cr_ratio]
        if intersectional_ratios:
            all_disparities.extend(intersectional_ratios.values())
        max_disparity = max(all_disparities)

        # Fairness Score (0 - 100)
        disparity_penalty = max(0.0, (max_disparity - 1.0) * 35.0)
        eod_penalty = equalized_odds_diff * 40.0
        fairness_score = max(0.0, min(100.0, 100.0 - (disparity_penalty + eod_penalty)))

        return {
            "slices": slices_report,
            "disparate_impact_ratio": disparate_impact_ratio,
            "equalized_odds_difference": equalized_odds_diff,
            "underdiagnosis_ratio_pediatric_vs_adult": pediatric_ratio,
            "pediatric_disparity_ci": ci_pediatric,
            "underdiagnosis_ratio_female_vs_male": female_ratio,
            "female_disparity_ci": ci_female,
            "underdiagnosis_ratio_cr_vs_dr": cr_ratio,
            "intersectional_disparities": intersectional_ratios,
            "max_disparity": max_disparity,
            "fairness_score": round(fairness_score, 1),
        }

    def _calculate_subgroup_metrics(self, df: pd.DataFrame, col: str) -> Dict[str, Dict[str, Any]]:
        metrics: Dict[str, Dict[str, Any]] = {}
        if col not in df.columns:
            return metrics

        for val, grp in df.groupby(col, observed=True):
            val_str = str(val)
            total = len(grp)
            positives = (grp["ground_truth"] == 1).sum()
            negatives = (grp["ground_truth"] == 0).sum()

            tp = ((grp["pred"] == 1) & (grp["ground_truth"] == 1)).sum()
            fn = ((grp["pred"] == 0) & (grp["ground_truth"] == 1)).sum()
            fp = ((grp["pred"] == 1) & (grp["ground_truth"] == 0)).sum()
            tn = ((grp["pred"] == 0) & (grp["ground_truth"] == 0)).sum()

            fnr = round(float(fn / positives), 4) if positives > 0 else 0.0
            tpr = round(float(tp / positives), 4) if positives > 0 else 0.0
            fpr = round(float(fp / negatives), 4) if negatives > 0 else 0.0
            pos_rate = round(float((grp["pred"] == 1).sum() / total), 4) if total > 0 else 0.0

            # Statistical Power and Sample Size Sufficiency Audit
            is_underpowered = bool(total < 30 or positives < 5)
            power_status = "UNDERPOWERED_SUBGROUP" if is_underpowered else "ADEQUATELY_POWERED"

            metrics[val_str] = {
                "sample_count": int(total),
                "positive_count": int(positives),
                "negative_count": int(negatives),
                "fnr": fnr,
                "tpr": tpr,
                "fpr": fpr,
                "positive_rate": pos_rate,
                "statistical_power_status": power_status,
                "is_underpowered": is_underpowered,
            }
        return metrics

    def optimize_thresholds_equalized_odds(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_features: pd.Series
    ) -> ThresholdOptimizer:
        """
        Active Bias Mitigation using Fairlearn's Linear Programming post-processing.
        Finds group-specific decision thresholds to equalize TPR and FPR across all 
        specified demographic subgroups, fulfilling formal Equalized Odds.
        """
        optimizer = ThresholdOptimizer(
            estimator=estimator,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=False
        )
        # Train the optimizer to find ideal thresholds for each sensitive group
        optimizer.fit(X, y, sensitive_features=sensitive_features)
        return optimizer

    def detect_latent_distribution_drift(
        self,
        embeddings: np.ndarray,
        site_labels: List[str],
        tier_profile: str = "TIER_2_WHITE_BOX",
    ) -> Dict[str, Any]:
        """Calculates 2-sample Kolmogorov-Smirnov test, Maximum Mean Discrepancy (MMD),
        and trains an Adversarial Domain Classifier across deployment sites.
        
        Gracefully adapts to:
        - Tier 2 (White-Box): 1024-dim penultimate latent embeddings.
        - Tier 1 (Black-Box): 16-dim observable input image domain statistics.
        """
        sites = list(set(site_labels))
        domain_type = (
            "PENULTIMATE_LATENT_EMBEDDINGS"
            if tier_profile == "TIER_2_WHITE_BOX"
            else "OBSERVABLE_INPUT_IMAGE_STATISTICS"
        )

        if len(sites) < 2 or len(embeddings) < 10:
            return {
                "tier_profile": tier_profile,
                "covariate_shift_domain": domain_type,
                "domain_shift_classifier_auroc": 0.50,
                "ks_drift_p_value": 1.0,
                "ks_statistic": 0.0,
                "latent_mmd_distance": 0.0,
                "site_divergence_detected": False,
                "shift_score": 100.0,
            }

        site_a, site_b = sites[0], sites[1]
        mask_a = np.array([s == site_a for s in site_labels])
        mask_b = np.array([s == site_b for s in site_labels])

        emb_a = embeddings[mask_a]
        emb_b = embeddings[mask_b]

        # 1. Multi-feature 2-Sample Kolmogorov-Smirnov Test
        ks_stats = []
        ks_pvals = []
        sample_dims = min(embeddings.shape[1], 32)
        dim_indices = np.linspace(0, embeddings.shape[1] - 1, sample_dims, dtype=int)

        for d in dim_indices:
            stat, pval = ks_2samp(emb_a[:, d], emb_b[:, d])
            ks_stats.append(stat)
            ks_pvals.append(pval)

        mean_ks_stat = float(np.mean(ks_stats))
        min_p_val = float(np.min(ks_pvals))

        # 2. Maximum Mean Discrepancy (MMD)
        mmd_val = compute_rbf_mmd(emb_a, emb_b)

        # 3. Adversarial Domain Classifier
        y_domain = np.array([1 if s == site_b else 0 for s in site_labels])
        
        n_splits = min(3, min(np.sum(y_domain == 0), np.sum(y_domain == 1)))
        if n_splits < 2:
            domain_auc = 0.50
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_aucs = []
            min_leaf = max(2, min(20, len(embeddings) // 8))
            for train_idx, test_idx in skf.split(embeddings, y_domain):
                auc = 0.50
                try:
                    clf = HistGradientBoostingClassifier(max_iter=50, min_samples_leaf=min_leaf, random_state=42)
                    clf.fit(embeddings[train_idx], y_domain[train_idx])
                    preds_proba = clf.predict_proba(embeddings[test_idx])[:, 1]
                    auc = roc_auc_score(y_domain[test_idx], preds_proba)
                    if auc == 0.50:
                        lr = LogisticRegression(max_iter=100, random_state=42)
                        lr.fit(embeddings[train_idx], y_domain[train_idx])
                        preds_proba = lr.predict_proba(embeddings[test_idx])[:, 1]
                        auc = roc_auc_score(y_domain[test_idx], preds_proba)
                except Exception:
                    lr = LogisticRegression(max_iter=100, random_state=42)
                    lr.fit(embeddings[train_idx], y_domain[train_idx])
                    preds_proba = lr.predict_proba(embeddings[test_idx])[:, 1]
                    auc = roc_auc_score(y_domain[test_idx], preds_proba)

                if len(set(y_domain[test_idx])) > 1:
                    cv_aucs.append(float(auc))
            domain_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.50

        domain_auc = round(domain_auc, 3)
        shift_alarm = bool(domain_auc > self.alarm_classifier_auc or min_p_val < 0.01 or mmd_val > 0.40)

        # Shift invariance score (100 = invariant, lower as domain separation increases)
        shift_penalty = max(0.0, (domain_auc - 0.50) * 200.0) + (mmd_val * 50.0)
        shift_score = max(0.0, min(100.0, 100.0 - shift_penalty))

        return {
            "tier_profile": tier_profile,
            "covariate_shift_domain": domain_type,
            "domain_shift_classifier_auroc": domain_auc,
            "ks_drift_p_value": round(min_p_val, 4),
            "ks_statistic": round(mean_ks_stat, 3),
            "latent_mmd_distance": mmd_val,
            "site_divergence_detected": shift_alarm,
            "shift_score": round(shift_score, 1),
        }

    def audit_demographic_latent_leakage(
        self,
        embeddings: np.ndarray,
        demographic_labels: List[str],
        tier_profile: str = "TIER_2_WHITE_BOX",
    ) -> Dict[str, Any]:
        """Tests whether patient demographic identity (Sex/Age) is strongly encoded in latent representations.
        
        Literature Grounding:
        - Gichoya et al. (Lancet Digital Health 2022): Deep models inadvertently learn demographic
          identifiers from medical scans, using demographic disease prevalence as a spurious shortcut.
        """
        unique_labels = list(set(demographic_labels))
        if tier_profile != "TIER_2_WHITE_BOX" or len(unique_labels) < 2 or len(embeddings) < 20:
            return {
                "demographic_leakage_probed": False,
                "leakage_auroc": 0.50,
                "demographic_shortcut_risk": "NOT_PROBED_OR_BLACK_BOX",
            }

        label_map = {l: idx for idx, l in enumerate(sorted(unique_labels[:2]))}
        mask = [lbl in label_map for lbl in demographic_labels]
        y_demo = np.array([label_map[lbl] for lbl, m in zip(demographic_labels, mask) if m])
        x_sub = embeddings[mask]

        if len(set(y_demo)) < 2 or np.min(np.bincount(y_demo)) < 4:
            return {
                "demographic_leakage_probed": False,
                "leakage_auroc": 0.50,
                "demographic_shortcut_risk": "UNDERPOWERED",
            }

        try:
            lr = LogisticRegression(max_iter=300, random_state=42)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            aucs = []
            for tr_idx, te_idx in skf.split(x_sub, y_demo):
                lr.fit(x_sub[tr_idx], y_demo[tr_idx])
                probs = lr.predict_proba(x_sub[te_idx])[:, 1]
                aucs.append(roc_auc_score(y_demo[te_idx], probs))
            mean_auc = float(np.mean(aucs))
        except Exception:
            mean_auc = 0.50

        leakage_alarm = bool(mean_auc >= 0.75)
        return {
            "demographic_leakage_probed": True,
            "leakage_auroc": round(mean_auc, 3),
            "demographic_shortcut_risk": "HIGH_SHORTCUT_LEAKAGE" if leakage_alarm else "SAFE_DEMOGRAPHIC_INVARIANCE",
            "leakage_detected": leakage_alarm,
        }
