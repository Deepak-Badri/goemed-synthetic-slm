"""
generator.py
------------
Synthetic patient record generator using Gaussian Copula + embedded
clinical risk models for label assignment.

Pipeline:
    1. Sample demographic stratum (age/sex/race) — Census-weighted
    2. Sample correlated continuous features via Gaussian Copula
    3. Sample binary/categorical features conditionally
    4. Evaluate risk models → assign condition labels via Bernoulli draws
    5. Apply comorbidity cascades (T2DM→CKD, HTN→CVD risk bump)

Output:
    - data/processed/synthetic_patients.parquet
    - data/processed/synthetic_patients.csv
"""

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, lognorm, gamma
from joblib import Parallel, delayed
from typing import Dict, Any, List

# ── Local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.generator.risk_models import (
    framingham_cvd_risk, ascvd_risk, findrisc_t2dm_risk,
    ckd_epi_egfr, ckd_label, stop_bang_score, osa_risk,
    hypertension_label, obesity_class, diabetes_label,
    charlson_index,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
CONFIG_DIR  = REPO_ROOT / "configs"
PROCESSED   = REPO_ROOT / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Random seed ────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════════
# LOAD CONFIGS
# ══════════════════════════════════════════════════════════════════════════

def load_configs():
    schema_path = CONFIG_DIR / "feature_schema.yaml"
    corr_path   = CONFIG_DIR / "correlation_matrix.yaml"

    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    with open(corr_path) as f:
        corr   = yaml.safe_load(f)

    log.info(f"Loaded feature schema: {len(schema)} features")
    log.info(f"Loaded correlation matrices: {len(corr)} strata")
    return schema, corr


# ══════════════════════════════════════════════════════════════════════════
# DEMOGRAPHIC SAMPLER
# Census ACS 2022 adult population shares
# ══════════════════════════════════════════════════════════════════════════

# Race/ethnicity population weights (US adults, Census ACS 2022)
RACE_WEIGHTS = {
    "nh_white":          0.594,
    "nh_black":          0.124,
    "hispanic":          0.185,
    "nh_asian":          0.060,
    "other_multiracial": 0.037,
}

# Sex weights
SEX_WEIGHTS = {"M": 0.492, "F": 0.508}

# Map generator race labels to correlation matrix stratum keys
RACE_TO_STRATUM = {
    "nh_white":          "nh_white",
    "nh_black":          "nh_black",
    "hispanic":          "mexican_american",   # best available proxy
    "nh_asian":          "nh_asian",
    "other_multiracial": "other_multiracial",
}

# Age distribution — US adults 18-80 (Census ACS)
AGE_BINS   = [18, 25, 35, 45, 55, 65, 75, 80]
AGE_WEIGHTS = [0.114, 0.133, 0.133, 0.133, 0.133, 0.120, 0.094, 0.040]


def sample_demographics(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample age, sex, race from Census-weighted distributions."""
    race = rng.choice(
        list(RACE_WEIGHTS.keys()),
        p=list(RACE_WEIGHTS.values())
    )
    sex = rng.choice(
        list(SEX_WEIGHTS.keys()),
        p=list(SEX_WEIGHTS.values())
    )
    # Sample age bin then uniform within bin
    bin_idx = rng.choice(len(AGE_BINS) - 1, p=AGE_WEIGHTS[:-1] /
                         np.array(AGE_WEIGHTS[:-1]).sum())
    age = rng.uniform(AGE_BINS[bin_idx], AGE_BINS[bin_idx + 1])
    age = float(np.clip(age, 18, 80))

    return {"age": age, "sex": sex, "race": race}


# ══════════════════════════════════════════════════════════════════════════
# GAUSSIAN COPULA SAMPLER
# ══════════════════════════════════════════════════════════════════════════

COPULA_FEATURES = [
    "bmi", "sbp", "dbp", "total_cholesterol",
    "hdl_cholesterol", "hba1c", "fasting_glucose",
    "waist_cm", "weight_kg",
]

def _get_corr_matrix(corr_config: Dict, stratum_key: str) -> np.ndarray:
    """Extract correlation matrix for a stratum as numpy array."""
    features = COPULA_FEATURES
    entry    = corr_config.get(stratum_key, corr_config.get("nh_white_M"))
    matrix   = entry["matrix"]
    n        = len(features)
    R        = np.eye(n)
    for i, fi in enumerate(features):
        for j, fj in enumerate(features):
            R[i, j] = matrix.get(fi, {}).get(fj, 0.0 if i != j else 1.0)
    return R


def sample_copula_features(
    demographics: Dict,
    schema: Dict,
    corr_config: Dict,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Sample correlated continuous features using Gaussian Copula.

    Steps:
        1. Look up stratum-specific correlation matrix R
        2. Draw z ~ N(0, R) — correlated normal vector
        3. Convert z to uniform via Phi (standard normal CDF)
        4. Invert each marginal distribution (quantile function)
        5. Clamp to physiological bounds
    """
    race    = demographics["race"]
    sex     = demographics["sex"]
    stratum = RACE_TO_STRATUM.get(race, "nh_white")
    key     = f"{stratum}_{sex}"

    R = _get_corr_matrix(corr_config, key)

    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Fallback: add small ridge if not PSD
        R += 1e-6 * np.eye(len(R))
        L  = np.linalg.cholesky(R)

    # Draw independent standard normals then correlate
    z_indep = rng.standard_normal(len(COPULA_FEATURES))
    z_corr  = L @ z_indep

    # Convert to uniform marginals via Phi
    u = norm.cdf(z_corr)
    u = np.clip(u, 1e-6, 1 - 1e-6)

    features = {}
    age = demographics["age"]

    for i, feat in enumerate(COPULA_FEATURES):
        spec = schema.get(feat, {})
        dist = spec.get("distribution", "normal")
        bounds = spec.get("bounds", {})
        lo = bounds.get("min", -np.inf)
        hi = bounds.get("max",  np.inf)

        # Get stratum-specific params, fall back to overall
        strata_params = spec.get("strata", {}).get(stratum, {}).get(sex, {})
        if not strata_params:
            strata_params = spec.get("overall", {})

        # Quantile inversion
        try:
            if dist == "normal":
                mu    = strata_params.get("mu", 0)
                sigma = strata_params.get("sigma", 1)
                val   = norm.ppf(u[i], loc=mu, scale=sigma)

            elif dist == "lognorm":
                mu_log    = strata_params.get("mu_log", 0)
                sigma_log = strata_params.get("sigma_log", 0.3)
                scale     = strata_params.get("scale", np.exp(mu_log))
                val       = lognorm.ppf(u[i], s=sigma_log,
                                        loc=0, scale=scale)

            elif dist == "gamma":
                alpha = strata_params.get("alpha", 2)
                scale = strata_params.get("scale", 10)
                val   = gamma.ppf(u[i], a=alpha,
                                  loc=strata_params.get("loc", 0),
                                  scale=scale)
            else:
                val = float(u[i])

        except Exception:
            val = float(u[i])

        # Apply age modulation for BP and cholesterol
        if feat == "sbp":
            val += (age - 50) * 0.45   # ~4.5 mmHg per decade above 50
        elif feat == "dbp":
            val += (age - 50) * 0.10
        elif feat == "total_cholesterol":
            val += (age - 50) * 0.30
        elif feat == "fasting_glucose":
            val -= 7.0

        features[feat] = float(np.clip(val, lo, hi))

    return features


# ══════════════════════════════════════════════════════════════════════════
# BINARY / CATEGORICAL FEATURE SAMPLER
# ══════════════════════════════════════════════════════════════════════════

def sample_binary_features(
    demographics: Dict,
    continuous: Dict,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Sample binary and categorical features conditional on demographics
    and continuous features. Probabilities from NHANES/BRFSS.
    """
    age  = demographics["age"]
    sex  = demographics["sex"]
    race = demographics["race"]
    bmi  = continuous.get("bmi", 27)
    sbp  = continuous.get("sbp", 120)

    features = {}

    # ── Smoking status ─────────────────────────────────────────────────
    # BRFSS 2022: ~11% current, ~24% former, ~65% never
    # Adjust slightly by age and race
    base_current = 0.11
    if age > 45:  base_current += 0.03
    if race == "nh_black":    base_current += 0.02
    if race == "nh_asian":    base_current -= 0.04

    base_former = 0.24
    if age > 55: base_former += 0.05

    p_current = np.clip(base_current, 0, 1)
    p_former  = np.clip(base_former, 0, 1)
    p_never   = np.clip(1 - p_current - p_former, 0, 1)

    smoke_draw = rng.choice(
        ["current", "former", "never"],
        p=np.array([p_current, p_former, p_never]) /
          (p_current + p_former + p_never)
    )
    features["smoking_status"] = smoke_draw
    features["smoker"]         = smoke_draw == "current"

    # Pack-years for current/former smokers
    if smoke_draw == "current":
        features["pack_years"] = float(rng.gamma(3, 7))
    elif smoke_draw == "former":
        features["pack_years"] = float(rng.gamma(2, 8))
    else:
        features["pack_years"] = 0.0

    # ── Physical activity ──────────────────────────────────────────────
    # BRFSS: ~25% sedentary, correlated with BMI
    p_sedentary = 0.25 + (bmi - 25) * 0.01
    p_sedentary = np.clip(p_sedentary, 0.05, 0.60)
    features["physical_activity_low"] = bool(rng.random() < p_sedentary)

    # ── BP treatment ──────────────────────────────────────────────────
    # ~75% of diagnosed hypertensives are on meds (AHA 2023)
    p_bp_trt = 0.75 if sbp >= 140 else (0.40 if sbp >= 130 else 0.05)
    features["bp_treated"] = bool(rng.random() < p_bp_trt)

    # ── Family history flags ──────────────────────────────────────────
    features["family_history_dm"]     = bool(rng.random() < 0.30)
    features["family_history_cvd"]    = bool(rng.random() < 0.25)
    features["family_history_cancer"] = bool(rng.random() < 0.20)

    # ── Diet quality ──────────────────────────────────────────────────
    # Low vegetable/fruit intake (BRFSS: ~24% eat <1 serving/day)
    features["diet_low_veg"] = bool(rng.random() < 0.24)

    # ── Alcohol use ──────────────────────────────────────────────────
    # NSDUH: ~6% AUD, ~29% current drinkers
    features["alcohol_use"] = bool(rng.random() < 0.55)
    features["heavy_drinker"] = bool(
        features["alcohol_use"] and rng.random() < 0.20)

    # ── Snoring / OSA proxy ──────────────────────────────────────────
    p_snore = 0.40 if sex == "M" else 0.25
    if bmi > 30: p_snore += 0.10
    features["snoring"]        = bool(rng.random() < p_snore)
    features["tired"]          = bool(rng.random() < 0.25)
    features["observed_apnea"] = bool(rng.random() < 0.10)

    # Neck circumference proxy from BMI
    if sex == "M":
        features["neck_cm"] = float(np.clip(32 + bmi * 0.35, 28, 55))
    else:
        features["neck_cm"] = float(np.clip(28 + bmi * 0.28, 24, 50))

    # ── Creatinine proxy if not from NHANES ──────────────────────────
    # Mean 0.90 mg/dL, modulated by age and sex
    base_cr = 0.72 if sex == "F" else 0.95
    age_adj = (age - 40) * 0.004
    cr_noise = rng.normal(0, 0.12)
    features["creatinine"] = float(
        np.clip(base_cr + age_adj + cr_noise, 0.4, 8.0))

    return features


# ══════════════════════════════════════════════════════════════════════════
# LABEL ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════

def assign_labels(
    patient: Dict,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Evaluate all risk models and assign condition labels via Bernoulli draws.
    Then apply comorbidity cascades.
    """
    labels = {}

# ── Hard threshold labels ─────────────────────────────────────────
    labels["hypertension"]    = hypertension_label(patient)
    labels["obesity_class"]   = obesity_class(patient)
    dm_status                 = diabetes_label(patient)
    labels["diabetes"]        = dm_status == "diabetes"
    labels["prediabetes"]     = dm_status == "prediabetes"

    # ── Probabilistic labels ──────────────────────────────────────────
    # CVD — use ASCVD as primary, Framingham as secondary weight
    cvd_prob = ascvd_risk(patient) * 0.60 + \
               framingham_cvd_risk(patient) * 0.40
    cvd_prob *= 0.28
    labels["cvd_risk"] = bool(rng.random() < np.clip(cvd_prob, 0, 1))

    # T2DM — use FINDRISC + hard ADA override
    hba1c = patient.get("hba1c", 5.0)
    fpg   = patient.get("fasting_glucose", 90)
    if hba1c >= 7.5 or fpg >= 150:
        labels["diabetes"] = True
    elif hba1c >= 6.5 or fpg >= 126:
        # Borderline — probabilistic
        labels["diabetes"] = bool(rng.random() < 0.35)
    else:
        t2dm_prob = findrisc_t2dm_risk(patient) * 0.10
        labels["diabetes"] = bool(rng.random() < t2dm_prob)
    labels["prediabetes"] = (not labels["diabetes"] and
                             (hba1c >= 5.7 or fpg >= 100))

    t2dm_prob = findrisc_t2dm_risk(patient)
    labels["t2dm_findrisc"] = bool(rng.random() < t2dm_prob * 0.35)

    # CKD
    egfr_val = ckd_epi_egfr(patient)
    labels["ckd"]  = egfr_val < 60.0
    labels["egfr"] = egfr_val
    if 60 <= egfr_val < 75:
        age = patient.get("age", 40)
        p_ckd_borderline = 0.20 + (age - 50) * 0.01
        if rng.random() < np.clip(p_ckd_borderline, 0, 0.40):
            labels["ckd"] = True

    # OSA
    labels["osa"] = bool(rng.random() < osa_risk(patient))

    # COPD — driven by smoking pack-years + age
    pack_years = patient.get("pack_years", 0)
    age        = patient.get("age", 40)
    p_copd = 0.02
    if pack_years > 20: p_copd += 0.15
    if pack_years > 10: p_copd += 0.08
    if age > 60:        p_copd += 0.05
    labels["copd"] = bool(rng.random() < np.clip(p_copd, 0, 1))

    # Depression/anxiety
    p_depression = 0.08
    if patient.get("sex") == "F": p_depression += 0.04
    if age < 35:                  p_depression += 0.03
    labels["depression"] = bool(rng.random() < p_depression)

    # Metabolic syndrome — ATP III: 3 of 5 criteria
    criteria = 0
    waist = patient.get("waist_cm", 85)
    sex   = patient.get("sex", "M")
    if (sex == "M" and waist >= 98) or \
       (sex == "F" and waist >= 84):    criteria += 1
    if patient.get("sbp", 110) >= 130 or \
       patient.get("dbp", 70)  >= 85:   criteria += 1
    if patient.get("fasting_glucose", 90) >= 100: criteria += 1
    if patient.get("hdl_cholesterol", 50) < \
       (40 if sex == "M" else 50):       criteria += 1
    if patient.get("total_cholesterol", 180) >= 220: criteria += 1
    labels["metabolic_syndrome"] = criteria >= 3

    # Hypothyroidism
    p_hypo = 0.046
    if sex == "F": p_hypo += 0.04
    if age > 60:   p_hypo += 0.02
    labels["hypothyroidism"] = bool(rng.random() < p_hypo)

    # ── Comorbidity cascades ──────────────────────────────────────────

    # ── Comorbidity cascades ──────────────────────────────────────────
    # T2DM increases CKD risk
    if labels["diabetes"] and not labels["ckd"]:
        if rng.random() < 0.25:
            labels["ckd"] = True

    # Hypertension increases CVD risk
    if labels["hypertension"] and not labels["cvd_risk"]:
        if rng.random() < 0.10:
            labels["cvd_risk"] = True

    # Obesity increases OSA risk
    if patient.get("bmi", 25) > 35 and not labels["osa"]:
        if rng.random() < 0.15:
            labels["osa"] = True

    # Charlson comorbidity index
    charlson_input = {**patient, **labels}
    labels["charlson_index"]  = charlson_index(charlson_input)

    return labels


# ══════════════════════════════════════════════════════════════════════════
# SINGLE PATIENT GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_patient(
    schema: Dict,
    corr_config: Dict,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Generate one complete synthetic patient record."""

    # Step 1 — Demographics
    demo = sample_demographics(rng)

    # Step 2 — Correlated continuous features
    continuous = sample_copula_features(demo, schema, corr_config, rng)

    # Step 3 — Binary/categorical features
    binary = sample_binary_features(demo, continuous, rng)

    # Assemble full patient dict
    patient = {**demo, **continuous, **binary}

    # Step 4 — Label assignment
    labels = assign_labels(patient, rng)

    # Final record
    record = {**patient, **labels}
    return record


# ══════════════════════════════════════════════════════════════════════════
# BATCH GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_batch(
    n: int,
    schema: Dict,
    corr_config: Dict,
    seed: int = SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Generate n synthetic patient records in parallel.

    Args:
        n:          Number of records to generate
        schema:     Feature schema from configs/feature_schema.yaml
        corr_config: Correlation matrices from configs/correlation_matrix.yaml
        seed:       Random seed for reproducibility
        n_jobs:     CPU cores to use (-1 = all available)
    """
    log.info(f"Generating {n:,} synthetic patient records "
             f"(seed={seed}, n_jobs={n_jobs})")

    # Generate per-worker seeds for reproducibility
    seed_seq = np.random.SeedSequence(seed)
    n_jobs_actual = n_jobs if n_jobs > 0 else 8
    child_seeds   = seed_seq.spawn(n_jobs_actual)

    # Split work across workers
    chunk_size = max(1, n // n_jobs_actual)
    chunks     = [chunk_size] * n_jobs_actual
    chunks[-1] += n - sum(chunks)   # remainder to last chunk

    def worker(chunk_n: int, child_seed) -> List[Dict]:
        worker_rng = np.random.default_rng(child_seed)
        return [generate_patient(schema, corr_config, worker_rng)
                for _ in range(chunk_n)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(worker)(chunk_n, cs)
        for chunk_n, cs in zip(chunks, child_seeds)
    )

    records = [r for batch in results for r in batch]
    df      = pd.DataFrame(records)

    log.info(f"Generated {len(df):,} records, "
             f"{len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════
# VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_prevalence_report(df: pd.DataFrame):
    """Compare synthetic label prevalences to published benchmarks."""
    log.info("\n" + "=" * 55)
    log.info("PREVALENCE VALIDATION vs Published Benchmarks")
    log.info("=" * 55)

    checks = [
        ("hypertension",      df["hypertension"].mean(),      0.47, "AHA 2023"),
        ("diabetes",          df["diabetes"].mean(),           0.11, "CDC 2023"),
        ("obesity_class",
         df["obesity_class"].isin(
             ["obese_1","obese_2","obese_3"]).mean(),          0.42, "CDC NHANES"),
        ("cvd_risk",          df["cvd_risk"].mean(),           0.12, "AHA 2023"),
        ("ckd",               df["ckd"].mean(),                0.15, "KDIGO/USRDS"),
        ("osa",               df["osa"].mean(),                0.26, "NHANES"),
        ("depression",        df["depression"].mean(),         0.08, "NIMH"),
        ("copd",              df["copd"].mean(),               0.06, "ALA"),
        ("metabolic_syndrome",df["metabolic_syndrome"].mean(), 0.33, "ATP III"),
        ("hypothyroidism",    df["hypothyroidism"].mean(),     0.05, "NHANES"),
    ]

    log.info(f"  {'Label':<22} {'Synthetic':>9} {'Benchmark':>10} "
             f"{'Source':<12} {'Delta':>8}")
    log.info("  " + "-" * 65)

    for label, synth, bench, source in checks:
        delta = synth - bench
        flag  = "⚠️ " if abs(delta) > 0.05 else "✅"
        log.info(f"  {flag} {label:<20} {synth:>9.1%} {bench:>10.1%} "
                 f"{source:<12} {delta:>+8.1%}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main(n: int = 10_000):
    log.info("=" * 55)
    log.info("GoEMed Synthetic Patient Generator")
    log.info("=" * 55)

    schema, corr_config = load_configs()

    df = generate_batch(n=n, schema=schema, corr_config=corr_config)

    # Prevalence validation
    print_prevalence_report(df)

    # Save outputs
    parquet_path = PROCESSED / "synthetic_patients.parquet"
    csv_path     = PROCESSED / "synthetic_patients.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    log.info(f"\nSaved -> {parquet_path}")
    log.info(f"Saved -> {csv_path}")
    log.info(f"File sizes: "
             f"Parquet={parquet_path.stat().st_size/1e6:.1f}MB  "
             f"CSV={csv_path.stat().st_size/1e6:.1f}MB")
    log.info("\nDone.")


if __name__ == "__main__":
    main(n=100_000)