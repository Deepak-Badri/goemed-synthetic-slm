"""
load_nhanes.py
--------------
Loads and merges all NHANES XPT files for both cycles into a single
clean DataFrame, joined on participant ID (SEQN).

Cycles:
    - 2017-March 2020 Pre-Pandemic (P_ prefix)
    - August 2021-August 2023 (_L suffix)

Output:
    - data/processed/nhanes_combined.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
RAW_2017    = REPO_ROOT / "data/raw/nhanes/2017_2020"
RAW_2021    = REPO_ROOT / "data/raw/nhanes/2021_2023"
PROCESSED   = REPO_ROOT / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── File manifest ──────────────────────────────────────────────────────────
FILES = {
    "2017_2020": {
        "P_DEMO.xpt":  "demographics",
        "P_BMX.xpt":   "body_measures",
        "P_BPXO.xpt":  "blood_pressure",
        "P_TCHOL.xpt": "total_cholesterol",
        "P_HDL.xpt":   "hdl_cholesterol",
        "P_DIQ.xpt":   "diabetes_questionnaire",
        "P_GHB.xpt":   "hba1c",
        "P_GLU.xpt":   "fasting_glucose",
        "P_SMQ.xpt":   "smoking",
        "P_PAQ.xpt":   "physical_activity",
    },
    "2021_2023": {
        "DEMO_L.xpt":  "demographics",
        "BMX_L.xpt":   "body_measures",
        "BPXO_L.xpt":  "blood_pressure",
        "TCHOL_L.xpt": "total_cholesterol",
        "HDL_L.xpt":   "hdl_cholesterol",
        "DIQ_L.xpt":   "diabetes_questionnaire",
        "GHB_L.xpt":   "hba1c",
        "GLU_L.xpt":   "fasting_glucose",
        "SMQ_L.xpt":   "smoking",
        "PAQ_L.xpt":   "physical_activity",
    },
}

# ── Column rename map ──────────────────────────────────────────────────────
RENAME = {
    # Demographics
    "SEQN":     "participant_id",
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",
    "RIDRETH3": "race_ethnicity",
    "DMDEDUC2": "education",
    "INDFMPIR": "poverty_ratio",
    # Body measures
    "BMXBMI":   "bmi",
    "BMXWAIST": "waist_cm",
    "BMXHT":    "height_cm",
    "BMXWT":    "weight_kg",
    # Blood pressure
    "BPXOSY1": "sbp_1", "BPXOSY2": "sbp_2", "BPXOSY3": "sbp_3",
    "BPXODI1": "dbp_1", "BPXODI2": "dbp_2", "BPXODI3": "dbp_3",
    # Cholesterol
    "LBXTC":   "total_cholesterol",
    "LBDHDD":  "hdl_cholesterol",
    # Diabetes
    "DIQ010":  "diagnosed_diabetes",
    "DIQ050":  "taking_insulin",
    # HbA1c
    "LBXGH":   "hba1c",
    # Fasting glucose
    "LBXGLU":  "fasting_glucose",
    # Smoking
    "SMQ020":  "ever_smoked_100",
    "SMQ040":  "current_smoker",
    # Physical activity
    "PAQ605":  "vigorous_work_activity",
    "PAQ620":  "moderate_work_activity",
    "PAD680":  "sedentary_minutes",
}


def load_xpt(path: Path, module: str, cycle: str) -> pd.DataFrame:
    """Read a single XPT file and return a DataFrame with SEQN as index."""
    log.info(f"  Loading {cycle} / {module} <- {path.name}")
    df = pd.read_sas(path, encoding="utf-8")
    if "SEQN" not in df.columns:
        raise ValueError(f"SEQN not found in {path.name}")
    df["SEQN"] = df["SEQN"].astype(int)
    df = df.set_index("SEQN")
    return df


def load_cycle(cycle: str, folder: Path) -> pd.DataFrame:
    """Load and merge all modules for one cycle into a single DataFrame."""
    log.info(f"\nLoading cycle: {cycle}")
    merged = None

    for filename, module in FILES[cycle].items():
        path = folder / filename
        if not path.exists():
            log.warning(f"  FILE NOT FOUND — skipping: {path}")
            continue

        df = load_xpt(path, module, cycle)

        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="left", rsuffix=f"_{module}")

    merged = merged.reset_index()
    merged["cycle"] = cycle
    log.info(f"  Cycle {cycle}: {len(merged):,} participants, "
             f"{len(merged.columns)} columns")
    return merged


def compute_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Average BP readings 2 and 3 per NHANES protocol."""
    sbp_cols = ["sbp_2", "sbp_3"]
    dbp_cols = ["dbp_2", "dbp_3"]

    available_sbp = [c for c in sbp_cols if c in df.columns]
    available_dbp = [c for c in dbp_cols if c in df.columns]

    if available_sbp:
        df["sbp"] = df[available_sbp].mean(axis=1, skipna=True)
    elif "sbp_1" in df.columns:
        df["sbp"] = df["sbp_1"]

    if available_dbp:
        df["dbp"] = df[available_dbp].mean(axis=1, skipna=True)
    elif "dbp_1" in df.columns:
        df["dbp"] = df["dbp_1"]

    drop_cols = ["sbp_1","sbp_2","sbp_3","dbp_1","dbp_2","dbp_3"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def clean_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and derived variables."""
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    df = compute_bp_averages(df)

    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "M", 2: "F"})

    if "age" in df.columns:
        before = len(df)
        df = df[df["age"] >= 18].copy()
        log.info(f"  Dropped {before - len(df):,} participants under 18")

    missing_pct = df.isnull().mean()
    drop_sparse = missing_pct[missing_pct > 0.80].index.tolist()
    if drop_sparse:
        log.info(f"  Dropping {len(drop_sparse)} columns >80% missing")
        df = df.drop(columns=drop_sparse)

    return df


def main():
    log.info("=" * 60)
    log.info("NHANES Data Loader")
    log.info("=" * 60)

    df_2017 = load_cycle("2017_2020", RAW_2017)
    df_2021 = load_cycle("2021_2023", RAW_2021)

    log.info("\nCombining cycles...")
    combined = pd.concat([df_2017, df_2021], axis=0, ignore_index=True)
    log.info(f"  Combined: {len(combined):,} participants, "
             f"{len(combined.columns)} columns")

    log.info("\nCleaning combined dataset...")
    combined = clean_combined(combined)
    log.info(f"  After cleaning: {len(combined):,} adults, "
             f"{len(combined.columns)} columns")

    log.info("\nKey variable summary:")
    key_vars = ["age", "bmi", "sbp", "dbp", "total_cholesterol",
                "hdl_cholesterol", "hba1c", "fasting_glucose"]
    for var in key_vars:
        if var in combined.columns:
            s = combined[var].dropna()
            log.info(f"  {var:<22} n={len(s):>6,}  "
                     f"mean={s.mean():>7.2f}  "
                     f"std={s.std():>6.2f}  "
                     f"missing={combined[var].isnull().mean():.1%}")

    out_path = PROCESSED / "nhanes_combined.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"\nSaved -> {out_path}")
    log.info(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")
    log.info("\nDone.")


if __name__ == "__main__":
    main()