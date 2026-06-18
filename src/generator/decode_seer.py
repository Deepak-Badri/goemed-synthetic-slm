"""
decode_seer.py
--------------
Decodes SEER*Stat numeric codes into human-readable labels
and extracts age/race/sex-specific colorectal cancer incidence
rates for use in the synthetic data generator.

Input:  data/raw/seer/colorectal_rates.csv
Output: configs/seer_colorectal_rates.yaml
"""

import yaml
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEER_PATH = REPO_ROOT / "data/raw/seer/colorectal_rates.csv"
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ── SEER code mappings ─────────────────────────────────────────────────────
SEX_MAP = {0: "male", 1: "female", 2: "both"}

RACE_MAP = {
    0: "white",
    1: "black",
    2: "other",
    3: "hispanic",
    4: "unknown",
}

# SEER age recode — maps numeric code to age group label
AGE_MAP = {
    0:  "<1",
    1:  "1-4",
    2:  "5-9",
    3:  "10-14",
    4:  "15-19",
    5:  "20-24",
    6:  "25-29",
    7:  "30-34",
    8:  "35-39",
    9:  "40-44",
    10: "45-49",
    11: "50-54",
    12: "55-59",
    13: "60-64",
    14: "65-69",
    15: "70-74",
    16: "75-79",
    17: "80-84",
    18: "85-89",
    19: "90+",
    20: "unknown",
}

# Generator age groups — maps SEER age codes to our 4 groups
GENERATOR_AGE_MAP = {
    0:  None, 1:  None, 2:  None, 3:  None, 4:  None,
    5:  None, 6:  None, 7:  None, 8:  None,  # under 18
    9:  "18_49", 10: "18_49", 11: "18_49",   # 40-54
    12: "50_64", 13: "50_64", 14: "50_64",   # 55-69
    15: "65_74", 16: "65_74",                # 70-79
    17: "75_plus", 18: "75_plus",
    19: "75_plus", 20: None,
}


def main():
    log.info("=" * 55)
    log.info("SEER Colorectal Cancer Rate Decoder")
    log.info("=" * 55)

    # Load raw SEER export
    df = pd.read_csv(SEER_PATH)
    log.info(f"Loaded {len(df):,} rows from SEER export")
    log.info(f"Columns: {df.columns.tolist()}")

    # Rename columns to standard names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "sex" in cl:
            col_map[col] = "sex_code"
        elif "age recode" in cl:
            col_map[col] = "age_code"
        elif "age-adjusted" in cl or "rate" in cl:
            col_map[col] = "rate"
        elif "race" in cl:
            col_map[col] = "race_code"
        elif "count" in cl:
            col_map[col] = "count"
        elif "population" in cl:
            col_map[col] = "population"
    df = df.rename(columns=col_map)
    log.info(f"Renamed columns: {df.columns.tolist()}")

    # Decode numeric codes
    df["sex"]       = df["sex_code"].map(SEX_MAP)
    df["race"]      = df["race_code"].map(RACE_MAP)
    df["age_group"] = df["age_code"].map(AGE_MAP)
    df["gen_age"]   = df["age_code"].map(GENERATOR_AGE_MAP)

    # Convert rate to numeric — suppress (~) becomes NaN
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    # Filter to adults only and exclude "both sexes" row
    df = df[
        (df["sex"].isin(["male", "female"])) &
        (df["gen_age"].notna()) &
        (df["rate"].notna())
    ].copy()

    log.info(f"After filtering to adults: {len(df):,} rows")

    # ── Build calibration table ────────────────────────────────────────
    # Rate is per 100,000 — convert to probability per year
    df["annual_prob"] = df["rate"] / 100_000

    # Group by generator age group, sex, race
    calibration = (
        df.groupby(["gen_age", "sex", "race"])["annual_prob"]
        .mean()
        .reset_index()
    )

    log.info("\nColorectal Cancer Annual Incidence Probabilities:")
    log.info(f"  {'Age Group':<12} {'Sex':<8} {'Race':<8} "
             f"{'Annual Prob':>12}")
    log.info("  " + "-" * 44)
    for _, row in calibration.iterrows():
        log.info(f"  {row['gen_age']:<12} {row['sex']:<8} "
                 f"{row['race']:<8} {row['annual_prob']:>12.6f}")

    # ── Save to YAML ───────────────────────────────────────────────────
    schema = {"colorectal_cancer": {"source": "SEER 1975-2023",
                                     "rate_per": 100_000,
                                     "rates": {}}}

    for _, row in calibration.iterrows():
        key = f"{row['gen_age']}_{row['sex']}_{row['race']}"
        schema["colorectal_cancer"]["rates"][key] = {
            "annual_prob": round(float(row["annual_prob"]), 6),
            "age_group":   row["gen_age"],
            "sex":         row["sex"],
            "race":        row["race"],
        }

    out_path = CONFIG_DIR / "seer_colorectal_rates.yaml"
    with open(out_path, "w") as f:
        yaml.dump(schema, f, default_flow_style=False)

    log.info(f"\nSaved {len(schema['colorectal_cancer']['rates'])} "
             f"rate entries -> {out_path}")
    log.info("\nDone.")


if __name__ == "__main__":
    main()