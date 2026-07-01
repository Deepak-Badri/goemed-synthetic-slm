"""
shap_analysis.py
----------------
SHAP (SHapley Additive exPlanations) feature importance analysis
using XGBoost models trained on synthetic tabular data.

XGBoost SHAP is used as the primary interpretability method because:
1. TreeSHAP is exact (not approximated) and computationally efficient
2. Feature-level importance is clinically more interpretable than token-level
3. Results validate that the synthetic data generator encoded correct
   clinical relationships

Clinical validation: top SHAP features should match known risk factors
from medical literature for each condition.

Output:
    - outputs/figures/shap_summary.png
    - outputs/figures/shap_beeswarm.png
    - outputs/figures/shap_per_label/
    - outputs/reports/shap_results.csv
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List

import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data/processed"
FIGURES   = REPO_ROOT / "outputs/figures"
REPORTS   = REPO_ROOT / "outputs/reports"
SHAP_FIG  = FIGURES / "shap_per_label"
SHAP_FIG.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Feature columns ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "sbp", "dbp",
    "total_cholesterol", "hdl_cholesterol",
    "hba1c", "fasting_glucose", "waist_cm", "weight_kg",
    "creatinine", "pack_years",
    "smoker", "physical_activity_low", "bp_treated",
    "family_history_dm", "family_history_cvd",
    "snoring", "tired", "alcohol_use", "sex_male",
]

FEATURE_DISPLAY = {
    "age":                  "Age",
    "bmi":                  "BMI",
    "sbp":                  "Systolic BP",
    "dbp":                  "Diastolic BP",
    "total_cholesterol":    "Total Cholesterol",
    "hdl_cholesterol":      "HDL Cholesterol",
    "hba1c":                "HbA1c",
    "fasting_glucose":      "Fasting Glucose",
    "waist_cm":             "Waist Circumference",
    "weight_kg":            "Weight",
    "creatinine":           "Creatinine",
    "pack_years":           "Pack Years (Smoking)",
    "smoker":               "Current Smoker",
    "physical_activity_low":"Low Physical Activity",
    "bp_treated":           "On BP Medication",
    "family_history_dm":    "Family Hx Diabetes",
    "family_history_cvd":   "Family Hx CVD",
    "snoring":              "Snoring",
    "tired":                "Daytime Fatigue",
    "alcohol_use":          "Alcohol Use",
    "sex_male":             "Male Sex",
}

LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

LABEL_DISPLAY = {
    "hypertension":       "Hypertension",
    "diabetes":           "Diabetes",
    "cvd_risk":           "CVD Risk",
    "ckd":                "CKD",
    "osa":                "Sleep Apnea",
    "depression":         "Depression",
    "copd":               "COPD",
    "metabolic_syndrome": "Metabolic Syndrome",
    "hypothyroidism":     "Hypothyroidism",
    "prediabetes":        "Prediabetes",
    "colorectal_cancer":  "Colorectal Cancer",
}

# Expected top features per condition — for clinical validation
CLINICAL_EXPECTATIONS = {
    "hypertension":       ["sbp", "dbp", "age", "bmi", "bp_treated"],
    "diabetes":           ["hba1c", "fasting_glucose", "bmi",
                           "family_history_dm", "age"],
    "cvd_risk":           ["sbp", "total_cholesterol", "hdl_cholesterol",
                           "age", "smoker", "sex_male", "bp_treated"],
    "ckd":                ["creatinine", "age", "sbp", "hba1c"],
    "osa":                ["bmi", "snoring", "sex_male", "age", "waist_cm"],
    "depression":         ["sex_male", "age", "physical_activity_low"],
    "copd":               ["pack_years", "smoker", "age"],
    "metabolic_syndrome": ["waist_cm", "fasting_glucose", "hdl_cholesterol",
                           "sbp", "bmi"],
    "hypothyroidism":     ["sex_male", "age"],
    "prediabetes":        ["hba1c", "fasting_glucose", "bmi", "age"],
    "colorectal_cancer":  ["age", "family_history_cvd", "bmi",
                           "alcohol_use"],
}


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_parquet(PROCESSED / "synthetic_patients.parquet")
    log.info(f"Loaded {len(df):,} synthetic patient records")

    df["sex_male"] = (df["sex"] == "M").astype(int)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in feat_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    X = df[feat_cols].fillna(df[feat_cols].median())
    Y = df[LABEL_COLS].astype(int)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=SEED)

    log.info(f"Features: {len(feat_cols)}  "
             f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    return X_train, X_test, Y_train, Y_test, feat_cols


# ══════════════════════════════════════════════════════════════════════════
# TRAIN XGBOOST PER LABEL
# ══════════════════════════════════════════════════════════════════════════

def train_xgboost_models(
    X_train, X_test, Y_train, Y_test
) -> Dict:
    """Train one XGBoost model per label for SHAP analysis."""
    log.info("\nTraining XGBoost models for SHAP...")
    models = {}

    for label in LABEL_COLS:
        pos_weight = (Y_train[label] == 0).sum() / \
                     max((Y_train[label] == 1).sum(), 1)

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=SEED,
            verbosity=0,
        )
        clf.fit(X_train, Y_train[label])
        models[label] = clf
        log.info(f"  Trained XGBoost for {label}")

    return models


# ══════════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def run_shap_analysis(
    models: Dict,
    X_test: pd.DataFrame,
    feat_cols: List[str],
) -> Dict:
    """Compute TreeSHAP values for each label."""
    log.info("\nComputing TreeSHAP values...")

    # Use a sample for speed
    X_sample = X_test.sample(
        min(2000, len(X_test)), random_state=SEED)

    all_shap = {}

    for label, model in models.items():
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary XGBoost, shap_values is array (n, features)
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class
        else:
            sv = shap_values

        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(sv).mean(axis=0)

        feature_importance = dict(zip(feat_cols, mean_abs_shap))
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )

        all_shap[label] = {
            "shap_values":       sv,
            "X_sample":          X_sample,
            "feature_importance":feature_importance,
            "top_features":      top_features,
        }

        log.info(f"\n  {LABEL_DISPLAY[label]} — Top 8 features:")
        for feat, val in top_features[:8]:
            disp = FEATURE_DISPLAY.get(feat, feat)
            log.info(f"    {disp:<28} mean|SHAP|={val:.4f}")

    return all_shap


# ══════════════════════════════════════════════════════════════════════════
# CLINICAL VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def validate_clinical_alignment(shap_results: Dict) -> pd.DataFrame:
    """Check whether top SHAP features match expected clinical features."""
    log.info("\n── Clinical Alignment Validation ──")
    rows = []

    for label, results in shap_results.items():
        top_feats  = [f for f, _ in results["top_features"][:8]]
        expected   = CLINICAL_EXPECTATIONS.get(label, [])
        matches    = [e for e in expected if e in top_feats]
        match_rate = len(matches) / max(len(expected), 1)
        status     = "✅ ALIGNED" if match_rate >= 0.50 else "⚠️  REVIEW"

        log.info(f"  {status} {label:<22} "
                 f"matches={len(matches)}/{len(expected)} "
                 f"({match_rate:.0%})")
        if matches:
            log.info(f"    Matched: "
                     f"{', '.join(FEATURE_DISPLAY.get(m,m) for m in matches)}")

        rows.append({
            "label":         label,
            "match_rate":    round(match_rate, 3),
            "n_matches":     len(matches),
            "n_expected":    len(expected),
            "matched_terms": "; ".join(matches),
            "top_8_features":"; ".join(top_feats),
            "status":        status,
        })

    df = pd.DataFrame(rows)
    log.info(f"\n  Overall alignment rate: "
             f"{df['match_rate'].mean():.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_shap_summary(shap_results: Dict, feat_cols: List[str]):
    """Global SHAP summary — mean |SHAP| per feature across all labels."""
    all_importance = np.zeros(len(feat_cols))

    for label, results in shap_results.items():
        for i, feat in enumerate(feat_cols):
            all_importance[i] += results["feature_importance"].get(feat, 0)

    all_importance /= len(shap_results)
    sorted_idx = np.argsort(all_importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(feat_cols)))

    feat_names = [FEATURE_DISPLAY.get(feat_cols[i], feat_cols[i])
                  for i in sorted_idx]
    vals       = all_importance[sorted_idx]

    bars = ax.barh(feat_names[::-1], vals[::-1],
                   color=colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (averaged across all conditions)",
                  fontsize=10)
    ax.set_title(
        "Global Feature Importance — XGBoost SHAP\n"
        "Averaged across all 11 health condition labels",
        fontweight="bold", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/shap_summary.png")


def plot_shap_per_label(shap_results: Dict, feat_cols: List[str]):
    """Bar chart of top SHAP features per label."""
    n      = len(shap_results)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, nrows * 4))
    axes = axes.flatten()

    colors = ["#1D9E75","#378ADD","#D85A30","#7F77DD","#BA7517",
              "#639922","#D85A30","#1D9E75","#378ADD","#7F77DD","#BA7517"]

    for ax, (label, results), color in zip(
            axes, shap_results.items(), colors):
        top = results["top_features"][:10]
        feats = [FEATURE_DISPLAY.get(f, f) for f, _ in top]
        vals  = [v for _, v in top]

        ax.barh(feats[::-1], vals[::-1], color=color, alpha=0.85)
        ax.set_title(LABEL_DISPLAY[label],
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(
        "XGBoost SHAP Feature Importance by Health Condition",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_per_label.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/shap_per_label.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("SHAP Feature Importance Analysis — XGBoost TreeSHAP")
    log.info("=" * 55)

    X_train, X_test, Y_train, Y_test, feat_cols = load_data()
    models      = train_xgboost_models(
        X_train, X_test, Y_train, Y_test)
    shap_results = run_shap_analysis(models, X_test, feat_cols)
    alignment_df = validate_clinical_alignment(shap_results)

    plot_shap_summary(shap_results, feat_cols)
    plot_shap_per_label(shap_results, feat_cols)

    alignment_df.to_csv(REPORTS / "shap_results.csv", index=False)
    log.info(f"\nSaved -> outputs/reports/shap_results.csv")
    log.info("\nDone.")


if __name__ == "__main__":
    main()