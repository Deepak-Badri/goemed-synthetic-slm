"""
serialize_prompts.py
--------------------
Converts synthetic patient records into natural language prompts
for Bio_ClinicalBERT fine-tuning.

Each patient record is serialized into 1 of 5 prompt templates
(randomly selected) to improve model robustness.

Output:
    - data/splits/train.jsonl
    - data/splits/val.jsonl
    - data/splits/test.jsonl
    - outputs/reports/split_summary.csv
"""

import json
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data/processed"
SPLITS    = REPO_ROOT / "data/splits"
REPORTS   = REPO_ROOT / "outputs/reports"
SPLITS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Condition label columns ────────────────────────────────────────────────
LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

# ── Human-readable label names ─────────────────────────────────────────────
LABEL_NAMES = {
    "hypertension":       "HYPERTENSION",
    "diabetes":           "DIABETES",
    "cvd_risk":           "CVD_RISK",
    "ckd":                "CHRONIC_KIDNEY_DISEASE",
    "osa":                "SLEEP_APNEA",
    "depression":         "DEPRESSION",
    "copd":               "COPD",
    "metabolic_syndrome": "METABOLIC_SYNDROME",
    "hypothyroidism":     "HYPOTHYROIDISM",
    "prediabetes":        "PREDIABETES",
    "colorectal_cancer":  "COLORECTAL_CANCER",
}

# ── Race display names ─────────────────────────────────────────────────────
RACE_DISPLAY = {
    "nh_white":          "White",
    "nh_black":          "Black",
    "hispanic":          "Hispanic",
    "nh_asian":          "Asian",
    "other_multiracial": "Multiracial",
}

# ── Obesity class display ──────────────────────────────────────────────────
OBESITY_DISPLAY = {
    "underweight": "underweight",
    "normal":      "normal weight",
    "overweight":  "overweight",
    "obese_1":     "obese (Class I)",
    "obese_2":     "obese (Class II)",
    "obese_3":     "severely obese (Class III)",
}


# ══════════════════════════════════════════════════════════════════════════
# HELPER FORMATTERS
# ══════════════════════════════════════════════════════════════════════════

def fmt_age(age: float) -> str:
    return f"{int(round(age))}-year-old"

def fmt_sex(sex: str) -> str:
    return "male" if sex == "M" else "female"

def fmt_race(race: str) -> str:
    return RACE_DISPLAY.get(race, race)

def fmt_bmi(bmi: float) -> str:
    cls = OBESITY_DISPLAY.get(
        "normal" if bmi < 25 else
        "overweight" if bmi < 30 else
        "obese_1" if bmi < 35 else
        "obese_2" if bmi < 40 else "obese_3",
        "unknown"
    )
    return f"{bmi:.1f} kg/m² ({cls})"

def fmt_bp(sbp: float, dbp: float) -> str:
    return f"{int(round(sbp))}/{int(round(dbp))} mmHg"

def fmt_smoke(status: str, pack_years: float) -> str:
    if status == "current":
        return f"current smoker ({pack_years:.0f} pack-years)"
    elif status == "former":
        return f"former smoker ({pack_years:.0f} pack-years)"
    else:
        return "non-smoker"

def fmt_activity(low: bool) -> str:
    return "sedentary / low activity" if low else "moderately to highly active"

def fmt_labels(row: pd.Series) -> str:
    active = [LABEL_NAMES[col] for col in LABEL_COLS
              if col in row.index and bool(row[col])]
    return ", ".join(active) if active else "NONE"


# ══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════

def template_1(row: pd.Series) -> str:
    """Clinical note style."""
    return (
        f"Patient: {fmt_age(row['age'])} {fmt_race(row['race'])} "
        f"{fmt_sex(row['sex'])}. "
        f"BMI: {fmt_bmi(row['bmi'])}. "
        f"Blood pressure: {fmt_bp(row['sbp'], row['dbp'])}. "
        f"Total cholesterol: {row['total_cholesterol']:.0f} mg/dL. "
        f"HDL cholesterol: {row['hdl_cholesterol']:.0f} mg/dL. "
        f"HbA1c: {row['hba1c']:.1f}%. "
        f"Fasting glucose: {row['fasting_glucose']:.0f} mg/dL. "
        f"Smoking: {fmt_smoke(row.get('smoking_status','never'), row.get('pack_years',0))}. "
        f"Physical activity: {fmt_activity(row.get('physical_activity_low', False))}. "
        f"Family history — diabetes: {'yes' if row.get('family_history_dm') else 'no'}, "
        f"CVD: {'yes' if row.get('family_history_cvd') else 'no'}. "
        f"BP treatment: {'yes' if row.get('bp_treated') else 'no'}. "
        f"Creatinine: {row.get('creatinine', 0.9):.2f} mg/dL. "
        f"Predicted conditions: [{fmt_labels(row)}]"
    )


def template_2(row: pd.Series) -> str:
    """Structured list style."""
    labels = fmt_labels(row)
    return (
        f"Demographics: {fmt_age(row['age'])} {fmt_sex(row['sex'])}, "
        f"{fmt_race(row['race'])} ethnicity. "
        f"Vitals: BP {fmt_bp(row['sbp'], row['dbp'])}, "
        f"BMI {row['bmi']:.1f}. "
        f"Labs: HbA1c {row['hba1c']:.1f}%, "
        f"glucose {row['fasting_glucose']:.0f} mg/dL, "
        f"total cholesterol {row['total_cholesterol']:.0f} mg/dL, "
        f"HDL {row['hdl_cholesterol']:.0f} mg/dL, "
        f"creatinine {row.get('creatinine', 0.9):.2f} mg/dL. "
        f"Lifestyle: {fmt_smoke(row.get('smoking_status','never'), row.get('pack_years',0))}, "
        f"{fmt_activity(row.get('physical_activity_low', False))}. "
        f"Family history: DM={'yes' if row.get('family_history_dm') else 'no'}, "
        f"CVD={'yes' if row.get('family_history_cvd') else 'no'}. "
        f"Conditions: [{labels}]"
    )


def template_3(row: pd.Series) -> str:
    """Narrative paragraph style."""
    sex_pronoun = "He" if row["sex"] == "M" else "She"
    labels = fmt_labels(row)
    bp_status = ("elevated blood pressure" if row["sbp"] >= 130
                 else "normal blood pressure")
    glucose_status = ("diabetic-range glucose" if row["fasting_glucose"] >= 126
                      else "pre-diabetic glucose" if row["fasting_glucose"] >= 100
                      else "normal glucose")
    return (
        f"A {fmt_age(row['age'])} {fmt_race(row['race'])} "
        f"{fmt_sex(row['sex'])} presents for health screening. "
        f"{sex_pronoun} has a BMI of {row['bmi']:.1f} and {bp_status} "
        f"at {fmt_bp(row['sbp'], row['dbp'])}. "
        f"Laboratory values show HbA1c of {row['hba1c']:.1f}% and "
        f"{glucose_status} at {row['fasting_glucose']:.0f} mg/dL. "
        f"Cholesterol panel: total {row['total_cholesterol']:.0f}, "
        f"HDL {row['hdl_cholesterol']:.0f} mg/dL. "
        f"{sex_pronoun} is a {fmt_smoke(row.get('smoking_status','never'), row.get('pack_years',0))} "
        f"and is {fmt_activity(row.get('physical_activity_low', False))}. "
        f"Family history is notable for "
        f"{'diabetes and ' if row.get('family_history_dm') else ''}"
        f"{'cardiovascular disease' if row.get('family_history_cvd') else 'no significant conditions'}. "
        f"Health risk assessment: [{labels}]"
    )


def template_4(row: pd.Series) -> str:
    """Abbreviated EHR style."""
    labels = fmt_labels(row)
    return (
        f"Pt: {fmt_age(row['age'])} {fmt_sex(row['sex'])}, "
        f"{fmt_race(row['race'])}. "
        f"BMI {row['bmi']:.1f}. "
        f"BP {fmt_bp(row['sbp'], row['dbp'])}. "
        f"HbA1c {row['hba1c']:.1f}%. "
        f"FPG {row['fasting_glucose']:.0f}. "
        f"TC {row['total_cholesterol']:.0f} / HDL {row['hdl_cholesterol']:.0f}. "
        f"Cr {row.get('creatinine', 0.9):.2f}. "
        f"Smk: {row.get('smoking_status','never')}. "
        f"FHx DM: {'Y' if row.get('family_history_dm') else 'N'}, "
        f"CVD: {'Y' if row.get('family_history_cvd') else 'N'}. "
        f"Meds: {'antihypertensive' if row.get('bp_treated') else 'none noted'}. "
        f"Dx: [{labels}]"
    )


def template_5(row: pd.Series) -> str:
    """Risk-focused style."""
    labels = fmt_labels(row)
    risk_factors = []
    if row["bmi"] >= 30:
        risk_factors.append(f"obesity (BMI {row['bmi']:.1f})")
    if row["sbp"] >= 130:
        risk_factors.append(f"elevated BP ({fmt_bp(row['sbp'], row['dbp'])})")
    if row["hba1c"] >= 5.7:
        risk_factors.append(f"elevated HbA1c ({row['hba1c']:.1f}%)")
    if row.get("smoking_status") == "current":
        risk_factors.append("active smoking")
    if row.get("family_history_cvd"):
        risk_factors.append("family history of CVD")
    if row.get("family_history_dm"):
        risk_factors.append("family history of diabetes")
    if row.get("physical_activity_low"):
        risk_factors.append("sedentary lifestyle")

    rf_str = (", ".join(risk_factors) if risk_factors
              else "no major modifiable risk factors identified")

    return (
        f"Health risk profile: {fmt_age(row['age'])} "
        f"{fmt_race(row['race'])} {fmt_sex(row['sex'])}. "
        f"Key risk factors: {rf_str}. "
        f"Cholesterol: total {row['total_cholesterol']:.0f} mg/dL, "
        f"HDL {row['hdl_cholesterol']:.0f} mg/dL. "
        f"Renal function: creatinine {row.get('creatinine', 0.9):.2f} mg/dL. "
        f"Predicted health conditions: [{labels}]"
    )


TEMPLATES = [template_1, template_2, template_3, template_4, template_5]


# ══════════════════════════════════════════════════════════════════════════
# SERIALIZER
# ══════════════════════════════════════════════════════════════════════════

def serialize_record(row: pd.Series, record_id: int) -> Dict[str, Any]:
    """Convert one patient record to a JSONL-ready dict."""
    template_fn = random.choice(TEMPLATES)
    text        = template_fn(row)

    # Build multi-label binary vector
    labels = {col: bool(row[col]) for col in LABEL_COLS
              if col in row.index}

    return {
        "id":     record_id,
        "text":   text,
        "labels": labels,
    }


# ══════════════════════════════════════════════════════════════════════════
# TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════

def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> tuple:
    """
    70/15/15 stratified split by race + sex to ensure
    demographic balance across all three sets.
    """
    train_dfs, val_dfs, test_dfs = [], [], []

    for (race, sex), group in df.groupby(["race", "sex"]):
        group = group.sample(frac=1, random_state=SEED)
        n     = len(group)
        n_tr  = int(n * train_frac)
        n_val = int(n * val_frac)

        train_dfs.append(group.iloc[:n_tr])
        val_dfs.append(group.iloc[n_tr:n_tr + n_val])
        test_dfs.append(group.iloc[n_tr + n_val:])

    train = pd.concat(train_dfs).sample(frac=1, random_state=SEED)
    val   = pd.concat(val_dfs).sample(frac=1, random_state=SEED)
    test  = pd.concat(test_dfs).sample(frac=1, random_state=SEED)

    return train, val, test


# ══════════════════════════════════════════════════════════════════════════
# WRITE JSONL
# ══════════════════════════════════════════════════════════════════════════

def write_jsonl(records: List[Dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log.info(f"  Wrote {len(records):,} records -> {path}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Prompt Serialization — Synthetic Patients → JSONL")
    log.info("=" * 55)

    # Load synthetic dataset
    df = pd.read_parquet(PROCESSED / "synthetic_patients.parquet")
    log.info(f"Loaded {len(df):,} synthetic patient records")

    # Fill missing values
    df["smoking_status"]       = df.get("smoking_status", "never")
    df["pack_years"]           = df.get("pack_years", 0.0)
    df["physical_activity_low"]= df.get("physical_activity_low", False)
    df["family_history_dm"]    = df.get("family_history_dm", False)
    df["family_history_cvd"]   = df.get("family_history_cvd", False)
    df["bp_treated"]           = df.get("bp_treated", False)
    df["creatinine"]           = df.get("creatinine", 0.9)

    # Stratified split
    log.info("Splitting 70/15/15 stratified by race × sex...")
    train_df, val_df, test_df = stratified_split(df)
    log.info(f"  Train: {len(train_df):,}  "
             f"Val: {len(val_df):,}  "
             f"Test: {len(test_df):,}")

    # Serialize each split
    log.info("\nSerializing records to natural language prompts...")
    splits = {
        "train": (train_df, SPLITS / "train.jsonl"),
        "val":   (val_df,   SPLITS / "val.jsonl"),
        "test":  (test_df,  SPLITS / "test.jsonl"),
    }

    summary_rows = []
    for split_name, (split_df, out_path) in splits.items():
        records = []
        for idx, (_, row) in enumerate(split_df.iterrows()):
            records.append(serialize_record(row, idx))
        write_jsonl(records, out_path)

        # Label distribution per split
        label_means = {
            col: split_df[col].mean()
            for col in LABEL_COLS if col in split_df.columns
        }
        summary_rows.append({"split": split_name,
                              "n": len(records), **label_means})

    # Preview first 3 prompts
    log.info("\n── Sample Prompts ──")
    with open(SPLITS / "train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            rec = json.loads(line)
            log.info(f"\n  [{i+1}] {rec['text'][:200]}...")
            active = [k for k, v in rec["labels"].items() if v]
            log.info(f"       Labels: {active}")

    # Save split summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log.info(f"\nLabel prevalence across splits:")
    log.info(f"  {'Label':<25} {'Train':>8} {'Val':>8} {'Test':>8}")
    log.info("  " + "-" * 52)
    for col in LABEL_COLS:
        if col in summary_df.columns:
            row = summary_df.set_index("split")[col]
            log.info(f"  {col:<25} "
                     f"{row.get('train',0):>8.1%} "
                     f"{row.get('val',0):>8.1%} "
                     f"{row.get('test',0):>8.1%}")

    log.info(f"\nSaved split summary -> {summary_path}")
    log.info(f"JSONL files -> data/splits/")
    log.info("\nDone.")


if __name__ == "__main__":
    main()