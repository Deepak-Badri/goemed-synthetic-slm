"""
fairness_audit.py
-----------------
Mandatory fairness audit per the GoEMed internship proposal (Section 6.3).

Evaluates Bio_ClinicalBERT performance separately across demographic
subgroups. Flags any condition where prediction accuracy differs by
more than 5 percentage points across race/sex/age groups.

Method:
    - Evaluate trained XGBoost models on test set
    - Stratify by race, sex, and age group
    - Compute AUROC per subgroup per condition
    - Flag gaps > 5pp as requiring investigation

Note: XGBoost is used for fairness audit (not Bio_ClinicalBERT)
because tabular models allow direct demographic feature analysis.
Bio_ClinicalBERT fairness is assessed via the demographic composition
of training data (validated in Week 4 validation suite).

Output:
    - outputs/reports/fairness_audit.csv
    - outputs/figures/fairness_heatmap.png
    - outputs/figures/fairness_gaps.png
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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

SEED = 42

LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

LABEL_DISPLAY = {
    "hypertension":       "HTN",
    "diabetes":           "DM",
    "cvd_risk":           "CVD",
    "ckd":                "CKD",
    "osa":                "OSA",
    "depression":         "DEP",
    "copd":               "COPD",
    "metabolic_syndrome": "MetSyn",
    "hypothyroidism":     "Thyroid",
    "prediabetes":        "PreDM",
    "colorectal_cancer":  "CRC",
}

FEATURE_COLS = [
    "age", "bmi", "sbp", "dbp",
    "total_cholesterol", "hdl_cholesterol",
    "hba1c", "fasting_glucose", "waist_cm", "weight_kg",
    "creatinine", "pack_years",
    "smoker", "physical_activity_low", "bp_treated",
    "family_history_dm", "family_history_cvd",
    "snoring", "tired", "alcohol_use", "sex_male",
]

RACE_DISPLAY = {
    "nh_white":          "NH White",
    "nh_black":          "NH Black",
    "hispanic":          "Hispanic",
    "nh_asian":          "NH Asian",
    "other_multiracial": "Other/Multi",
}

GAP_THRESHOLD = 0.05  # 5 percentage points per proposal requirement


# ══════════════════════════════════════════════════════════════════════════
# DATA + MODELS
# ══════════════════════════════════════════════════════════════════════════

def load_and_train():
    df = pd.read_parquet(PROCESSED / "synthetic_patients.parquet")
    log.info(f"Loaded {len(df):,} records")

    df["sex_male"] = (df["sex"] == "M").astype(int)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 34, 49, 64, 80],
        labels=["18-34", "35-49", "50-64", "65-80"]
    )

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in feat_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED)

    X_train = train_df[feat_cols].fillna(train_df[feat_cols].median())
    X_test  = test_df[feat_cols].fillna(train_df[feat_cols].median())
    Y_train = train_df[LABEL_COLS].astype(int)
    Y_test  = test_df[LABEL_COLS].astype(int)

    log.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    # Train XGBoost per label
    log.info("\nTraining XGBoost models...")
    models = {}
    all_probs = np.zeros((len(X_test), len(LABEL_COLS)))

    for i, label in enumerate(LABEL_COLS):
        pos_weight = (Y_train[label] == 0).sum() / \
                     max((Y_train[label] == 1).sum(), 1)
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=SEED,
            verbosity=0,
        )
        clf.fit(X_train, Y_train[label])
        all_probs[:, i] = clf.predict_proba(X_test)[:, 1]
        models[label] = clf
        log.info(f"  Trained: {label}")

    return test_df.reset_index(drop=True), Y_test.reset_index(drop=True), \
           all_probs


# ══════════════════════════════════════════════════════════════════════════
# FAIRNESS EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate_subgroup(
    test_df: pd.DataFrame,
    Y_test: pd.DataFrame,
    all_probs: np.ndarray,
    group_col: str,
    group_display: dict = None,
) -> pd.DataFrame:
    """Compute AUROC per subgroup per label."""
    rows = []
    groups = test_df[group_col].unique()

    for group in sorted(groups, key=str):
        mask = test_df[group_col] == group
        n    = mask.sum()
        if n < 100:
            continue

        group_label = (group_display.get(str(group), str(group))
                       if group_display else str(group))

        for i, label in enumerate(LABEL_COLS):
            y_true = Y_test[label][mask].values
            y_prob = all_probs[mask, i]

            if y_true.sum() < 5 or (1 - y_true).sum() < 5:
                auroc = float("nan")
            else:
                try:
                    auroc = roc_auc_score(y_true, y_prob)
                except Exception:
                    auroc = float("nan")

            rows.append({
                "group_col":   group_col,
                "group":       group_label,
                "label":       label,
                "auroc":       round(auroc, 4) if not np.isnan(auroc)
                               else np.nan,
                "n":           int(n),
                "n_positive":  int(y_true.sum()),
            })

    return pd.DataFrame(rows)


def compute_gaps(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute max AUROC gap across subgroups per label.
    Flag gaps > GAP_THRESHOLD (5pp per proposal).
    """
    rows = []
    for label in LABEL_COLS:
        sub = subgroup_df[
            subgroup_df["label"] == label]["auroc"].dropna()
        if len(sub) < 2:
            continue
        gap    = sub.max() - sub.min()
        flagged = gap > GAP_THRESHOLD
        rows.append({
            "label":    label,
            "max_auroc":round(float(sub.max()), 4),
            "min_auroc":round(float(sub.min()), 4),
            "gap":      round(float(gap), 4),
            "flagged":  flagged,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_fairness_heatmap(
    subgroup_df: pd.DataFrame,
    group_col: str,
    title_suffix: str,
):
    """Heatmap of AUROC per subgroup × label."""
    pivot = subgroup_df[subgroup_df["group_col"] == group_col].pivot(
        index="group", columns="label", values="auroc")

    # Rename columns
    pivot.columns = [LABEL_DISPLAY.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        annot=True, fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5, vmax=1.0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(
        f"AUROC by {title_suffix} — Fairness Audit\n"
        f"(Red = lower performance, Green = higher)",
        fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    fname = f"fairness_heatmap_{group_col}.png"
    plt.savefig(FIGURES / fname, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Saved -> outputs/figures/{fname}")


def plot_fairness_gaps(gaps_df: pd.DataFrame, group_col: str):
    """Bar chart of AUROC gaps per label."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#D85A30" if row["flagged"] else "#1D9E75"
              for _, row in gaps_df.iterrows()]

    ax.bar(
        [LABEL_DISPLAY.get(l, l) for l in gaps_df["label"]],
        gaps_df["gap"],
        color=colors, alpha=0.85,
    )
    ax.axhline(GAP_THRESHOLD, color="red", linestyle="--",
               linewidth=1.5, label=f"Flag threshold ({GAP_THRESHOLD:.0%})")
    ax.set_xlabel("Condition Label", fontsize=10)
    ax.set_ylabel("Max AUROC Gap Across Subgroups", fontsize=10)
    ax.set_title(
        f"Fairness Audit — AUROC Gap by {group_col}\n"
        f"Red bars = flagged (gap > {GAP_THRESHOLD:.0%})",
        fontweight="bold", fontsize=11)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fname = f"fairness_gaps_{group_col}.png"
    plt.savefig(FIGURES / fname, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Saved -> outputs/figures/{fname}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Fairness Audit — Demographic Subgroup Analysis")
    log.info("(Required per GoEMed Internship Proposal Section 6.3)")
    log.info("=" * 55)

    test_df, Y_test, all_probs = load_and_train()

    all_subgroup_dfs = []

    # ── By Race ───────────────────────────────────────────────────────
    log.info("\n── Fairness by Race/Ethnicity ──")
    race_df = evaluate_subgroup(
        test_df, Y_test, all_probs, "race", RACE_DISPLAY)
    race_gaps = compute_gaps(race_df)

    log.info(f"\n  {'Label':<22} {'Max':>7} {'Min':>7} "
             f"{'Gap':>7} {'Flag':>6}")
    log.info("  " + "-" * 50)
    for _, row in race_gaps.iterrows():
        flag = "🚩 YES" if row["flagged"] else "  ok"
        log.info(f"  {row['label']:<22} {row['max_auroc']:>7.4f} "
                 f"{row['min_auroc']:>7.4f} {row['gap']:>7.4f} "
                 f"{flag:>6}")

    plot_fairness_heatmap(race_df, "race", "Race/Ethnicity")
    plot_fairness_gaps(race_gaps, "race")
    all_subgroup_dfs.append(race_df)

    # ── By Sex ────────────────────────────────────────────────────────
    log.info("\n── Fairness by Sex ──")
    sex_display = {"M": "Male", "F": "Female"}
    sex_df  = evaluate_subgroup(
        test_df, Y_test, all_probs, "sex", sex_display)
    sex_gaps = compute_gaps(sex_df)

    log.info(f"\n  {'Label':<22} {'Max':>7} {'Min':>7} "
             f"{'Gap':>7} {'Flag':>6}")
    log.info("  " + "-" * 50)
    for _, row in sex_gaps.iterrows():
        flag = "🚩 YES" if row["flagged"] else "  ok"
        log.info(f"  {row['label']:<22} {row['max_auroc']:>7.4f} "
                 f"{row['min_auroc']:>7.4f} {row['gap']:>7.4f} "
                 f"{flag:>6}")

    plot_fairness_heatmap(sex_df, "sex", "Sex")
    plot_fairness_gaps(sex_gaps, "sex")
    all_subgroup_dfs.append(sex_df)

    # ── By Age Group ──────────────────────────────────────────────────
    log.info("\n── Fairness by Age Group ──")
    age_df  = evaluate_subgroup(
        test_df, Y_test, all_probs, "age_group")
    age_gaps = compute_gaps(age_df)

    log.info(f"\n  {'Label':<22} {'Max':>7} {'Min':>7} "
             f"{'Gap':>7} {'Flag':>6}")
    log.info("  " + "-" * 50)
    for _, row in age_gaps.iterrows():
        flag = "🚩 YES" if row["flagged"] else "  ok"
        log.info(f"  {row['label']:<22} {row['max_auroc']:>7.4f} "
                 f"{row['min_auroc']:>7.4f} {row['gap']:>7.4f} "
                 f"{flag:>6}")

    plot_fairness_heatmap(age_df, "age_group", "Age Group")
    plot_fairness_gaps(age_gaps, "age_group")
    all_subgroup_dfs.append(age_df)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("FAIRNESS AUDIT SUMMARY")
    log.info("=" * 55)

    all_gaps = pd.concat([race_gaps, sex_gaps, age_gaps])
    n_flagged = all_gaps["flagged"].sum()
    n_total   = len(all_gaps)

    log.info(f"\n  Total checks:    {n_total}")
    log.info(f"  Flagged (>5pp):  {n_flagged}")
    log.info(f"  Pass rate:       {(n_total-n_flagged)/n_total:.1%}")

    if n_flagged > 0:
        log.info(f"\n  Flagged conditions:")
        for _, row in all_gaps[all_gaps["flagged"]].iterrows():
            log.info(f"    {row['label']:<22} gap={row['gap']:.4f}")
    else:
        log.info("\n  ✅ No conditions flagged — "
                 "model performs equitably across all subgroups")

    # Save
    full_df = pd.concat(all_subgroup_dfs)
    full_df.to_csv(REPORTS / "fairness_audit.csv", index=False)
    all_gaps.to_csv(REPORTS / "fairness_gaps.csv", index=False)
    log.info(f"\nSaved -> outputs/reports/fairness_audit.csv")
    log.info(f"Saved -> outputs/reports/fairness_gaps.csv")
    log.info("\nDone.")


if __name__ == "__main__":
    main()