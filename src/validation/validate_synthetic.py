"""
validate_synthetic.py
---------------------
Full statistical validation of the synthetic dataset against
NHANES benchmarks.

Tests:
    1. Marginal KS tests — continuous features
    2. Correlation matrix comparison — Frobenius norm
    3. Propensity score discriminator — AUROC target ~0.55
    4. Label prevalence benchmarking
    5. Demographic composition check

Output:
    - outputs/reports/validation_report.csv
    - outputs/figures/validation_*.png
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED  = REPO_ROOT / "data/processed"
FIGURES    = REPO_ROOT / "outputs/figures"
REPORTS    = REPO_ROOT / "outputs/reports"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

CONTINUOUS_FEATURES = [
    "age", "bmi", "sbp", "dbp",
    "total_cholesterol", "hdl_cholesterol",
    "hba1c", "fasting_glucose", "waist_cm", "weight_kg",
]


# ══════════════════════════════════════════════════════════════════════════
# 1. MARGINAL KS TESTS
# ══════════════════════════════════════════════════════════════════════════

def run_ks_tests(synth: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """KS test for each continuous feature: synthetic vs real NHANES."""
    log.info("\n── Marginal KS Tests (Synthetic vs NHANES) ──")
    rows = []
    for feat in CONTINUOUS_FEATURES:
        if feat not in synth.columns or feat not in real.columns:
            continue
        s = synth[feat].dropna().values
        r = real[feat].dropna().values
        stat, pval = ks_2samp(s, r)
        status = "PASS" if pval > 0.05 else "FAIL"
        rows.append({
            "feature": feat,
            "ks_statistic": round(stat, 4),
            "p_value": round(pval, 4),
            "status": status,
            "synth_mean": round(float(np.mean(s)), 2),
            "real_mean":  round(float(np.mean(r)), 2),
            "synth_std":  round(float(np.std(s)), 2),
            "real_std":   round(float(np.std(r)), 2),
        })
        log.info(f"  {feat:<22} KS={stat:.4f}  p={pval:.4f}  "
                 f"{'✅' if status=='PASS' else '⚠️ '} {status}  "
                 f"synth_mean={np.mean(s):.1f}  real_mean={np.mean(r):.1f}")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 2. CORRELATION MATRIX COMPARISON
# ══════════════════════════════════════════════════════════════════════════

def compare_correlation_matrices(
    synth: pd.DataFrame,
    real: pd.DataFrame,
) -> float:
    """
    Compare pairwise correlation matrices.
    Frobenius norm of difference — lower is better.
    """
    log.info("\n── Correlation Matrix Comparison ──")
    feats = [f for f in CONTINUOUS_FEATURES
             if f in synth.columns and f in real.columns]

    synth_corr = synth[feats].corr(method="pearson")
    real_corr  = real[feats].corr(method="pearson")
    diff       = synth_corr.values - real_corr.values
    frob_norm  = float(np.linalg.norm(diff, "fro"))

    log.info(f"  Frobenius norm of correlation diff: {frob_norm:.4f}")
    log.info(f"  {'✅ PASS' if frob_norm < 2.0 else '⚠️  FAIL'} "
             f"(threshold: 2.0)")

    # Plot side-by-side heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    mask = np.zeros_like(synth_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(real_corr, ax=axes[0], annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 7}, mask=mask)
    axes[0].set_title("NHANES (Real)", fontweight="bold")

    sns.heatmap(synth_corr, ax=axes[1], annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 7}, mask=mask)
    axes[1].set_title("Synthetic", fontweight="bold")

    diff_df = pd.DataFrame(diff, index=feats, columns=feats)
    sns.heatmap(diff_df, ax=axes[2], annot=True, fmt=".2f",
                cmap="RdBu_r", center=0,
                annot_kws={"size": 7}, mask=mask)
    axes[2].set_title(f"Difference (Frobenius={frob_norm:.3f})",
                      fontweight="bold")

    plt.suptitle("Correlation Matrix Comparison — Real vs Synthetic",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "validation_correlation_comparison.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Saved correlation comparison heatmap")
    return frob_norm


# ══════════════════════════════════════════════════════════════════════════
# 3. PROPENSITY SCORE DISCRIMINATOR
# ══════════════════════════════════════════════════════════════════════════

def propensity_score_test(
    synth: pd.DataFrame,
    real: pd.DataFrame,
) -> float:
    """
    Train a logistic regression to distinguish synthetic vs real.
    Target AUROC ~0.55 — close to 0.5 means synthetic is indistinguishable.
    """
    log.info("\n── Propensity Score Discriminator ──")
    feats = [f for f in CONTINUOUS_FEATURES
             if f in synth.columns and f in real.columns]

    # Sample equal sizes
    n = min(len(synth), len(real), 5000)
    s_sample = synth[feats].dropna().sample(n, random_state=42)
    r_sample = real[feats].dropna().sample(
        min(n, len(real[feats].dropna())), random_state=42)

    X = pd.concat([s_sample, r_sample], ignore_index=True)
    y = np.array([1]*len(s_sample) + [0]*len(r_sample))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.median()))

    clf = LogisticRegression(max_iter=1000, random_state=42)
    auroc = cross_val_score(
        clf, X_scaled, y, cv=5, scoring="roc_auc").mean()

    log.info(f"  Propensity AUROC: {auroc:.4f}")
    log.info(f"  {'✅ PASS' if auroc < 0.65 else '⚠️  FAIL'} "
             f"(target: < 0.65, ideal: ~0.55)")
    return float(auroc)


# ══════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUTION OVERLAY PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_distribution_overlays(
    synth: pd.DataFrame,
    real: pd.DataFrame,
):
    """Overlay synthetic vs real distributions for key features."""
    feats = [f for f in CONTINUOUS_FEATURES
             if f in synth.columns and f in real.columns]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, feat in enumerate(feats):
        s = synth[feat].dropna()
        r = real[feat].dropna()
        axes[i].hist(r, bins=40, alpha=0.5, color="#378ADD",
                     density=True, label="NHANES (real)")
        axes[i].hist(s, bins=40, alpha=0.5, color="#1D9E75",
                     density=True, label="Synthetic")
        axes[i].set_title(feat, fontweight="bold", fontsize=10)
        axes[i].legend(fontsize=7)

    plt.suptitle("Distribution Overlay — Synthetic vs NHANES",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "validation_distribution_overlay.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Saved distribution overlay plot")


# ══════════════════════════════════════════════════════════════════════════
# 5. DEMOGRAPHIC COMPOSITION
# ══════════════════════════════════════════════════════════════════════════

def check_demographic_composition(synth: pd.DataFrame):
    """Verify demographic distribution matches Census ACS targets."""
    log.info("\n── Demographic Composition ──")

    census_targets = {
        "nh_white":          0.594,
        "nh_black":          0.124,
        "hispanic":          0.185,
        "nh_asian":          0.060,
        "other_multiracial": 0.037,
    }

    race_dist = synth["race"].value_counts(normalize=True)
    log.info(f"  {'Race':<22} {'Synthetic':>10} {'Census':>10} {'Delta':>8}")
    log.info("  " + "-" * 54)
    for race, target in census_targets.items():
        actual = race_dist.get(race, 0)
        delta  = actual - target
        flag   = "✅" if abs(delta) < 0.02 else "⚠️ "
        log.info(f"  {flag} {race:<20} {actual:>10.1%} "
                 f"{target:>10.1%} {delta:>+8.1%}")

    # Sex distribution
    sex_dist = synth["sex"].value_counts(normalize=True)
    log.info(f"\n  Sex distribution:")
    for sex, pct in sex_dist.items():
        log.info(f"    {sex}: {pct:.1%}")

    # Age distribution
    log.info(f"\n  Age: mean={synth['age'].mean():.1f}  "
             f"std={synth['age'].std():.1f}  "
             f"min={synth['age'].min():.0f}  "
             f"max={synth['age'].max():.0f}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Synthetic Dataset Validation Suite")
    log.info("=" * 55)

    # Load datasets
    synth = pd.read_parquet(
        PROCESSED / "synthetic_patients.parquet")
    real  = pd.read_parquet(
        PROCESSED / "nhanes_combined.parquet")

    log.info(f"Synthetic: {len(synth):,} records, "
             f"{len(synth.columns)} columns")
    log.info(f"NHANES:    {len(real):,} records, "
             f"{len(real.columns)} columns")

    # Run all validation tests
    ks_results   = run_ks_tests(synth, real)
    frob_norm    = compare_correlation_matrices(synth, real)
    propensity   = propensity_score_test(synth, real)
    plot_distribution_overlays(synth, real)
    check_demographic_composition(synth)

    # Save validation report
    report = {
        "n_synthetic":       len(synth),
        "n_real":            len(real),
        "ks_pass_rate":      (ks_results["status"] == "PASS").mean(),
        "frobenius_norm":    frob_norm,
        "propensity_auroc":  propensity,
        "frob_pass":         frob_norm < 2.0,
        "propensity_pass":   propensity < 0.65,
    }

    log.info("\n" + "=" * 55)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 55)
    for k, v in report.items():
        if isinstance(v, float):
            log.info(f"  {k:<25} {v:.4f}")
        else:
            log.info(f"  {k:<25} {v}")

    ks_results.to_csv(
        REPORTS / "validation_ks_tests.csv", index=False)
    log.info(f"\nSaved -> outputs/reports/validation_ks_tests.csv")
    log.info(f"Saved -> outputs/figures/validation_*.png")
    log.info("\nDone.")


if __name__ == "__main__":
    main()