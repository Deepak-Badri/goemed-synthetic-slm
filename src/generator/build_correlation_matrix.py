"""
build_correlation_matrix.py
---------------------------
Computes stratified multivariate correlation matrices (Pearson & Spearman)
across demographics to anchor the Gaussian Copula engine.
"""
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

REPO_ROOT      = Path(__file__).resolve().parents[2]
PROCESSED_DATA = REPO_ROOT / "data/processed/nhanes_combined.parquet"
OUTPUT_DIR     = REPO_ROOT / "configs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RACE_MAP = {
    1.0: "mexican_american",
    2.0: "other_hispanic",
    3.0: "nh_white",
    4.0: "nh_black",
    6.0: "nh_asian",
    7.0: "other_multiracial",
}

CONTINUOUS_FEATURES = [
    "age", "bmi", "weight_kg", "height_cm", "sbp",
    "dbp", "hdl_cholesterol", "total_cholesterol",
    "hba1c", "fasting_glucose",
]


def verify_positive_semi_definite(matrix, stratum_id):
    """Ensures linear algebra stability for Copula sampling."""
    eigenvalues = np.linalg.eigvals(matrix)
    min_eig = np.min(eigenvalues.real)
    if min_eig < 0:
        log.warning(f"  Stratum {stratum_id} not PSD "
                    f"(min eigenvalue={min_eig:.5f}) — applying ridge fix")
        adjusted = matrix + (abs(min_eig) + 1e-5) * np.eye(matrix.shape[0])
        return adjusted, False
    return matrix, True


def build_stratified_matrices():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"Missing: {PROCESSED_DATA}")

    df = pd.read_parquet(PROCESSED_DATA)
    df["race_label"] = df["race_ethnicity"].map(RACE_MAP)

    log.info(f"Loaded {len(df):,} adults")
    log.info(f"Building correlation matrices for "
             f"{df['race_label'].nunique()} races x 2 sexes\n")

    correlation_schema = {}
    strata_groups = df.groupby(["race_label", "sex"])

    for (race, sex), group in strata_groups:
        stratum_key = f"{race}_{sex}"

        if len(group) < 30:
            log.warning(f"  {stratum_key}: too few rows — skipping")
            continue

        sub_df = group[CONTINUOUS_FEATURES].dropna()

        if len(sub_df) < 30:
            log.warning(f"  {stratum_key}: too few complete rows "
                        f"after dropna ({len(sub_df)}) — skipping")
            continue

        corr_matrix = sub_df.corr(method="pearson")
        matrix_np   = corr_matrix.to_numpy()

        fixed_np, was_psd = verify_positive_semi_definite(
            matrix_np, stratum_key)

        if not was_psd:
            fixed_df     = pd.DataFrame(
                fixed_np,
                index=CONTINUOUS_FEATURES,
                columns=CONTINUOUS_FEATURES,
            )
            matrix_dict = fixed_df.to_dict()
        else:
            matrix_dict = corr_matrix.to_dict()

        correlation_schema[stratum_key] = {
            "sample_size": int(len(sub_df)),
            "matrix":      matrix_dict,
        }

        # Log key clinical correlations as a sanity check
        bmi_sbp = corr_matrix.loc["bmi", "sbp"] \
            if "sbp" in corr_matrix.columns else float("nan")
        age_sbp = corr_matrix.loc["age", "sbp"] \
            if "sbp" in corr_matrix.columns else float("nan")
        bmi_hdl = corr_matrix.loc["bmi", "hdl_cholesterol"] \
            if "hdl_cholesterol" in corr_matrix.columns else float("nan")

        log.info(f"  {stratum_key:<35} n={len(sub_df):>5,}  "
                 f"BMI↔SBP={bmi_sbp:+.3f}  "
                 f"age↔SBP={age_sbp:+.3f}  "
                 f"BMI↔HDL={bmi_hdl:+.3f}  "
                 f"{'PSD✓' if was_psd else 'PSD-fixed'}")

    output_path = OUTPUT_DIR / "correlation_matrix.yaml"
    with open(output_path, "w") as f:
        yaml.dump(correlation_schema, f, default_flow_style=False)

    log.info(f"\nSaved {len(correlation_schema)} strata -> {output_path}")
    log.info("Done.")


if __name__ == "__main__":
    build_stratified_matrices()