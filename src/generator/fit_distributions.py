"""
fit_distributions.py
--------------------
Fits marginal statistical distributions to each continuous feature
in the NHANES combined dataset, stratified by sex and race/ethnicity.

Outputs:
    - configs/feature_schema.yaml  — distribution parameters per stratum
    - outputs/figures/qq_plots/    — QQ plots per feature
    - outputs/reports/ks_tests.csv — KS test results per feature/stratum
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import kstest, norm, lognorm, gamma, beta
import warnings
warnings.filterwarnings("ignore")
import logging

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED  = REPO_ROOT / "data/processed/nhanes_combined.parquet"
CONFIG_DIR = REPO_ROOT / "configs"
FIGURES    = REPO_ROOT / "outputs/figures/qq_plots"
REPORTS    = REPO_ROOT / "outputs/reports"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Feature distribution spec ──────────────────────────────────────────────
# Each entry: (column_name, distribution, bounds, description)
# Distribution choices based on clinical literature:
#   normal   — symmetric, unbounded (BP, cholesterol)
#   lognorm  — right-skewed, positive (BMI, triglycerides, glucose)
#   gamma    — right-skewed, positive (glucose, waist)
FEATURES = {
    "age": {
        "dist": "normal",
        "bounds": (18, 80),
        "units": "years",
        "description": "Age in years (80 = 80+ per NHANES top-coding)",
    },
    "bmi": {
        "dist": "lognorm",
        "bounds": (15, 70),
        "units": "kg/m²",
        "description": "Body Mass Index",
    },
    "sbp": {
        "dist": "normal",
        "bounds": (80, 220),
        "units": "mmHg",
        "description": "Systolic blood pressure (average of readings 2&3)",
    },
    "dbp": {
        "dist": "normal",
        "bounds": (40, 130),
        "units": "mmHg",
        "description": "Diastolic blood pressure (average of readings 2&3)",
    },
    "total_cholesterol": {
        "dist": "normal",
        "bounds": (100, 400),
        "units": "mg/dL",
        "description": "Total cholesterol",
    },
    "hdl_cholesterol": {
        "dist": "lognorm",
        "bounds": (20, 120),
        "units": "mg/dL",
        "description": "HDL cholesterol",
    },
    "hba1c": {
        "dist": "lognorm",
        "bounds": (4.0, 15.0),
        "units": "%",
        "description": "Glycated hemoglobin — key diabetes marker",
    },
    "fasting_glucose": {
        "dist": "gamma",
        "bounds": (60, 500),
        "units": "mg/dL",
        "description": "Fasting plasma glucose",
    },
    "waist_cm": {
        "dist": "normal",
        "bounds": (50, 180),
        "units": "cm",
        "description": "Waist circumference",
    },
    "weight_kg": {
        "dist": "lognorm",
        "bounds": (30, 250),
        "units": "kg",
        "description": "Body weight",
    },
}

# ── Strata ─────────────────────────────────────────────────────────────────
RACE_MAP = {
    1.0: "mexican_american",
    2.0: "other_hispanic",
    3.0: "nh_white",
    4.0: "nh_black",
    6.0: "nh_asian",
    7.0: "other_multiracial",
}
SEXES = ["M", "F"]


def fit_normal(data: np.ndarray) -> dict:
    """Fit normal distribution using MLE."""
    mu, sigma = norm.fit(data)
    return {"mu": round(float(mu), 4), "sigma": round(float(sigma), 4)}


def fit_lognorm(data: np.ndarray) -> dict:
    """Fit log-normal distribution using MLE."""
    shape, loc, scale = lognorm.fit(data, floc=0)
    mu_log    = round(float(np.log(scale)), 4)
    sigma_log = round(float(shape), 4)
    return {"mu_log": mu_log, "sigma_log": sigma_log,
            "loc": round(float(loc), 4), "scale": round(float(scale), 4)}


def fit_gamma(data: np.ndarray) -> dict:
    """Fit gamma distribution using MLE."""
    a, loc, scale = gamma.fit(data, floc=0)
    return {"alpha": round(float(a), 4),
            "loc":   round(float(loc), 4),
            "scale": round(float(scale), 4)}


def run_ks_test(data: np.ndarray, dist_name: str, params: dict) -> dict:
    """Run KS test to evaluate goodness of fit."""
    if dist_name == "normal":
        cdf = lambda x: norm.cdf(x, loc=params["mu"], scale=params["sigma"])
    elif dist_name == "lognorm":
        cdf = lambda x: lognorm.cdf(
            x, s=params["sigma_log"],
            loc=params["loc"], scale=params["scale"])
    elif dist_name == "gamma":
        cdf = lambda x: gamma.cdf(
            x, a=params["alpha"],
            loc=params["loc"], scale=params["scale"])
    else:
        return {}

    ks_stat, p_value = kstest(data, cdf)
    return {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value":      round(float(p_value), 4),
        "pass":         bool(p_value > 0.05),
    }


def fit_feature(data: np.ndarray, dist_name: str) -> dict:
    """Dispatch to the right fitting function."""
    data = data[~np.isnan(data)]
    if len(data) < 50:
        return None
    if dist_name == "normal":
        return fit_normal(data)
    elif dist_name == "lognorm":
        return fit_lognorm(data)
    elif dist_name == "gamma":
        return fit_gamma(data)
    return None


def make_qq_plot(data: np.ndarray, dist_name: str,
                 params: dict, feature: str,
                 stratum: str, ax: plt.Axes):
    """Draw a QQ plot for a fitted distribution."""
    data = np.sort(data[~np.isnan(data)])
    n = len(data)
    probs = (np.arange(1, n+1) - 0.5) / n

    if dist_name == "normal":
        theoretical = norm.ppf(probs, loc=params["mu"], scale=params["sigma"])
    elif dist_name == "lognorm":
        theoretical = lognorm.ppf(
            probs, s=params["sigma_log"],
            loc=params["loc"], scale=params["scale"])
    elif dist_name == "gamma":
        theoretical = gamma.ppf(
            probs, a=params["alpha"],
            loc=params["loc"], scale=params["scale"])
    else:
        return

    ax.scatter(theoretical, data, alpha=0.3, s=4, color="#1D9E75")
    mn = min(theoretical.min(), data.min())
    mx = max(theoretical.max(), data.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax.set_title(f"{feature}\n{stratum}", fontsize=8)
    ax.set_xlabel("Theoretical", fontsize=7)
    ax.set_ylabel("Observed", fontsize=7)


def main():
    log.info("=" * 60)
    log.info("Distribution Fitting — NHANES Combined Dataset")
    log.info("=" * 60)

    df = pd.read_parquet(PROCESSED)
    df["race_label"] = df["race_ethnicity"].map(RACE_MAP)
    log.info(f"Loaded {len(df):,} adults")

    schema   = {}   # will become feature_schema.yaml
    ks_rows  = []   # will become ks_tests.csv

    for feature, spec in FEATURES.items():
        if feature not in df.columns:
            log.warning(f"  MISSING column: {feature} — skipping")
            continue

        log.info(f"\nFitting: {feature} ({spec['dist']})")
        schema[feature] = {
            "description": spec["description"],
            "units":       spec["units"],
            "distribution": spec["dist"],
            "bounds":      {"min": spec["bounds"][0],
                            "max": spec["bounds"][1]},
            "strata":      {},
            "overall":     {},
        }

        # ── Overall fit (population-level) ─────────────────────────────
        overall_data = df[feature].dropna().values
        overall_data = overall_data[
            (overall_data >= spec["bounds"][0]) &
            (overall_data <= spec["bounds"][1])
        ]
        params = fit_feature(overall_data, spec["dist"])
        if params:
            ks    = run_ks_test(overall_data, spec["dist"], params)
            schema[feature]["overall"] = {**params, "ks_test": ks,
                                           "n": int(len(overall_data))}
            log.info(f"  Overall  n={len(overall_data):>6,}  "
                     f"KS={ks['ks_statistic']:.4f}  "
                     f"p={ks['p_value']:.4f}  "
                     f"{'PASS' if ks['pass'] else 'FAIL'}")

        # ── QQ plots — one figure per feature ──────────────────────────
        races = [r for r in RACE_MAP.values()
                 if r in df["race_label"].unique()]
        n_strata = len(SEXES) * len(races)
        ncols = len(SEXES)
        nrows = len(races)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.5, nrows * 3),
            squeeze=False
        )

        # ── Stratum-level fits ─────────────────────────────────────────
        for ri, race in enumerate(races):
            schema[feature]["strata"][race] = {}
            for ci, sex in enumerate(SEXES):
                stratum_key = f"{race}_{sex}"
                mask = (
                    (df["race_label"] == race) &
                    (df["sex"] == sex)
                )
                raw = df.loc[mask, feature].dropna().values
                raw = raw[
                    (raw >= spec["bounds"][0]) &
                    (raw <= spec["bounds"][1])
                ]

                if len(raw) < 50:
                    log.warning(f"  {stratum_key}: too few observations "
                                f"({len(raw)}) — skipping")
                    axes[ri][ci].set_visible(False)
                    continue

                params = fit_feature(raw, spec["dist"])
                if params is None:
                    continue

                ks = run_ks_test(raw, spec["dist"], params)
                schema[feature]["strata"][race][sex] = {
                    **params,
                    "n":       int(len(raw)),
                    "ks_test": ks,
                }

                ks_rows.append({
                    "feature":      feature,
                    "race":         race,
                    "sex":          sex,
                    "distribution": spec["dist"],
                    "n":            len(raw),
                    "ks_statistic": ks["ks_statistic"],
                    "p_value":      ks["p_value"],
                    "pass":         ks["pass"],
                    **{f"param_{k}": v for k, v in params.items()},
                })

                log.info(f"  {stratum_key:<35} n={len(raw):>5,}  "
                         f"KS={ks['ks_statistic']:.4f}  "
                         f"p={ks['p_value']:.4f}  "
                         f"{'PASS' if ks['pass'] else 'FAIL'}")

                make_qq_plot(raw, spec["dist"], params,
                             feature, stratum_key, axes[ri][ci])

        plt.suptitle(f"QQ Plots — {feature} ({spec['dist']})",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        qq_path = FIGURES / f"qq_{feature}.png"
        plt.savefig(qq_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"  QQ plot saved -> {qq_path.name}")

    # ── Save feature_schema.yaml ───────────────────────────────────────
    yaml_path = CONFIG_DIR / "feature_schema.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
    log.info(f"\nFeature schema saved -> {yaml_path}")

    # ── Save KS test results ───────────────────────────────────────────
    ks_df = pd.DataFrame(ks_rows)
    ks_path = REPORTS / "ks_tests.csv"
    ks_df.to_csv(ks_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────
    pass_rate = ks_df["pass"].mean() * 100
    log.info(f"\nKS test results saved -> {ks_path}")
    log.info(f"Overall KS pass rate: {pass_rate:.1f}% "
             f"({ks_df['pass'].sum()}/{len(ks_df)} strata)")
    log.info("\nDone.")


if __name__ == "__main__":
    main()