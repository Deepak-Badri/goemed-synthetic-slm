"""
baseline_models.py
------------------
Trains Logistic Regression and XGBoost baseline models on the
synthetic tabular data for multi-label health condition prediction.

These baselines establish benchmark AUROC performance that
Bio_ClinicalBERT must meet or exceed to justify its added complexity.

Output:
    - outputs/reports/baseline_results.csv
    - outputs/figures/baseline_auroc.png
    - outputs/figures/baseline_pr_curves.png
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, brier_score_loss,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb
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
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Feature columns for tabular models ────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "sbp", "dbp",
    "total_cholesterol", "hdl_cholesterol",
    "hba1c", "fasting_glucose", "waist_cm", "weight_kg",
    "creatinine", "pack_years",
    "smoker", "physical_activity_low", "bp_treated",
    "family_history_dm", "family_history_cvd",
    "snoring", "tired", "alcohol_use",
]

# ── Label columns ──────────────────────────────────────────────────────────
LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

# Labels whose definition is a deterministic function of the input features.
# These are excluded from headline macro averages because a model that
# reconstructs the definition will trivially score AUROC ~= 1.0.
DETERMINISTIC_LABELS = {"hypertension", "metabolic_syndrome", "ckd"}

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


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load synthetic dataset and prepare feature/label matrices."""
    df = pd.read_parquet(PROCESSED / "synthetic_patients.parquet")
    log.info(f"Loaded {len(df):,} synthetic patient records")

    # Encode sex as binary
    df["sex_male"] = (df["sex"] == "M").astype(int)

    # Encode race as dummies
    race_dummies = pd.get_dummies(df["race"], prefix="race")
    df = pd.concat([df, race_dummies], axis=1)
    race_cols = race_dummies.columns.tolist()

    # Final feature set
    feat_cols = FEATURE_COLS + ["sex_male"] + race_cols
    feat_cols = [c for c in feat_cols if c in df.columns]

    # Convert boolean features to int
    for col in feat_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Stratified 70/15/15 split matching serialization
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:n_train + n_val]
    test_df  = df.iloc[n_train + n_val:]

    # Preserve NaN so XGBoost can use its native sparsity-aware splits.
    # LogisticRegression handles missingness separately in its own pipeline.
    X_train = train_df[feat_cols]
    X_val   = val_df[feat_cols]
    X_test  = test_df[feat_cols]

    Y_train = train_df[LABEL_COLS].astype(int)
    Y_val   = val_df[LABEL_COLS].astype(int)
    Y_test  = test_df[LABEL_COLS].astype(int)

    log.info(f"Features: {len(feat_cols)}  Labels: {len(LABEL_COLS)}")
    log.info(f"Train: {len(X_train):,}  Val: {len(X_val):,}  "
             f"Test: {len(X_test):,}")

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, feat_cols


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model_name: str,
    Y_true: pd.DataFrame,
    Y_pred_proba: np.ndarray,
) -> pd.DataFrame:
    """Compute per-label AUROC, F1, Brier score, Average Precision."""
    rows = []
    for i, label in enumerate(LABEL_COLS):
        y_true = Y_true[label].values
        y_prob = Y_pred_proba[:, i]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auroc = roc_auc_score(y_true, y_prob)
        except Exception:
            auroc = float("nan")

        try:
            ap = average_precision_score(y_true, y_prob)
        except Exception:
            ap = float("nan")

        f1    = f1_score(y_true, y_pred, zero_division=0)
        brier = brier_score_loss(y_true, y_prob)

        rows.append({
            "model":     model_name,
            "label":     label,
            "auroc":     round(auroc, 4),
            "avg_prec":  round(ap, 4),
            "f1":        round(f1, 4),
            "brier":     round(brier, 4),
            "prevalence": round(y_true.mean(), 4),
        })

    df = pd.DataFrame(rows)
    df["deterministic"] = df["label"].isin(DETERMINISTIC_LABELS)

    # Headline macro average excludes deterministic labels
    learned = df[~df["deterministic"]]
    deterministic = df[df["deterministic"]]
    macro_auroc = learned["auroc"].mean()
    macro_f1    = learned["f1"].mean()

    log.info(f"\n  {model_name} — Test Set Performance:")
    log.info(f"  {'Label':<22} {'AUROC':>7} {'F1':>7} "
             f"{'AvgPrec':>8} {'Brier':>7}")
    log.info("  " + "-" * 55)
    for _, row in df.iterrows():
        det_marker = " *" if row["deterministic"] else ""
        log.info(f"  {row['label']+det_marker:<22} {row['auroc']:>7.4f} "
                 f"{row['f1']:>7.4f} {row['avg_prec']:>8.4f} "
                 f"{row['brier']:>7.4f}")
    log.info("  " + "-" * 55)
    log.info(f"  {'MACRO (learned only)':<22} {macro_auroc:>7.4f} "
             f"{macro_f1:>7.4f}")
    if len(deterministic) > 0:
        det_auroc = deterministic["auroc"].mean()
        log.info(f"  {'MACRO (deterministic *)':<22} {det_auroc:>7.4f}")
        log.info("  * = label is a deterministic function of features")
    return df


# ══════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train, X_test, Y_train, Y_test
) -> pd.DataFrame:
    """One-vs-rest Logistic Regression per label."""
    log.info("\n── Training Logistic Regression (OvR) ──")

    # LR requires imputation — use train medians only
    train_medians = X_train.median()
    X_train_imp = X_train.fillna(train_medians)
    X_test_imp  = X_test.fillna(train_medians)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train_imp)
    X_te   = scaler.transform(X_test_imp)

    Y_pred_proba = np.zeros((len(X_test), len(LABEL_COLS)))

    for i, label in enumerate(LABEL_COLS):
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=SEED,
            class_weight="balanced",
        )
        clf.fit(X_tr, Y_train[label].values)
        Y_pred_proba[:, i] = clf.predict_proba(X_te)[:, 1]
        log.info(f"  Trained LR for {label}")

    return evaluate_model("Logistic Regression", Y_test, Y_pred_proba)


def train_xgboost(
    X_train, X_val, X_test, Y_train, Y_val, Y_test
) -> pd.DataFrame:
    """XGBoost per label with early stopping on validation set."""
    log.info("\n── Training XGBoost (per label) ──")

    Y_pred_proba = np.zeros((len(X_test), len(LABEL_COLS)))

    for i, label in enumerate(LABEL_COLS):
        pos_weight = (Y_train[label] == 0).sum() / \
                     max((Y_train[label] == 1).sum(), 1)

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="auc",
            early_stopping_rounds=20,
            random_state=SEED,
            verbosity=0,
        )
        clf.fit(
            X_train, Y_train[label].values,
            eval_set=[(X_val, Y_val[label].values)],
            verbose=False,
        )
        Y_pred_proba[:, i] = clf.predict_proba(X_test)[:, 1]
        log.info(f"  Trained XGBoost for {label} "
                 f"(best_iter={clf.best_iteration})")

    return evaluate_model("XGBoost", Y_test, Y_pred_proba)


# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════

def plot_auroc_comparison(results: pd.DataFrame):
    """Bar chart comparing AUROC per label across models."""
    models   = results["model"].unique()
    labels   = LABEL_COLS
    x        = np.arange(len(labels))
    width    = 0.35
    colors   = ["#378ADD", "#D85A30"]

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = results[results["model"] == model].set_index("label")
        aurocs = [vals.loc[l, "auroc"] if l in vals.index
                  else 0 for l in labels]
        bars = ax.bar(x + i * width, aurocs, width,
                      label=model, color=color, alpha=0.85)

    ax.axhline(0.7, color="gray", linestyle="--",
               linewidth=1, label="0.70 threshold")
    ax.axhline(0.5, color="red", linestyle=":",
               linewidth=1, label="Random (0.50)")
    ax.set_xlabel("Condition Label")
    ax.set_ylabel("AUROC")
    ax.set_title("Baseline Model AUROC by Condition Label",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        [LABEL_DISPLAY.get(l, l) for l in labels],
        rotation=30, ha="right"
    )
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "baseline_auroc.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/baseline_auroc.png")


def plot_macro_comparison(results: pd.DataFrame):
    """Summary bar comparing macro-average metrics."""
    summary = results.groupby("model")[
        ["auroc", "f1", "avg_prec", "brier"]].mean()

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    metrics = ["auroc", "f1", "avg_prec", "brier"]
    titles  = ["Macro AUROC", "Macro F1",
               "Macro Avg Precision", "Macro Brier Score"]
    colors  = ["#378ADD", "#D85A30"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = summary[metric]
        bars = ax.bar(vals.index, vals.values,
                      color=colors[:len(vals)])
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center",
                    fontsize=10, fontweight="bold")

    plt.suptitle("Baseline Model Summary — Macro-Average Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "baseline_macro_comparison.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/baseline_macro_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Baseline ML Models — Multi-Label Health Prediction")
    log.info("=" * 55)

    X_train, X_val, X_test, Y_train, Y_val, Y_test, feat_cols = \
        load_data()

    # Train both models
    lr_results  = train_logistic_regression(
        X_train, X_test, Y_train, Y_test)
    xgb_results = train_xgboost(
        X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # Combine results
    all_results = pd.concat([lr_results, xgb_results],
                            ignore_index=True)

    # Summary comparison
    log.info("\n" + "=" * 55)
    log.info("MODEL COMPARISON — MACRO AVERAGES")
    log.info("=" * 55)
    summary = all_results.groupby("model")[
        ["auroc","f1","avg_prec","brier"]].mean()
    log.info(f"\n{summary.round(4).to_string()}")

    # Plots
    plot_auroc_comparison(all_results)
    plot_macro_comparison(all_results)

    # Save results
    out_path = REPORTS / "baseline_results.csv"
    all_results.to_csv(out_path, index=False)
    log.info(f"\nSaved -> {out_path}")
    log.info("\nDone. Bio_ClinicalBERT must beat these baselines.")


if __name__ == "__main__":
    main()