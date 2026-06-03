"""
risk_models.py
--------------
Validated clinical risk model equations embedded as callable Python functions.
Each function takes a patient feature dictionary and returns a probability (0-1)
or score that drives label assignment in the synthetic data generator.

Sources:
    - Framingham Risk Score: JAMA 2001, Circulation 2008
    - ACC/AHA ASCVD: JACC 2013 (Goff et al.)
    - FINDRISC: Diabetologia 2003 (Lindström & Tuomilehto)
    - FRAX: WHO/NOGG 2008 (Kanis et al.)
    - STOP-BANG: Anesthesiology 2008 (Chung et al.)
    - CKD-EPI: AJKD 2009 (Levey et al.) — 2021 race-free update
    - PHQ-9: JGIM 2001 (Kroenke et al.)
    - JNC8/ACC-AHA 2017: Hypertension guidelines
"""

import math
import numpy as np
from typing import Dict, Any

# ── Type alias ─────────────────────────────────────────────────────────────
Patient = Dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_log(x: float) -> float:
    """Natural log with guard against zero/negative."""
    return math.log(max(x, 1e-10))


def _sigmoid(x: float) -> float:
    """Logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


# ══════════════════════════════════════════════════════════════════════════
# 1. FRAMINGHAM RISK SCORE (2008)
#    10-year CVD risk
#    Source: Circulation 2008;117:743-753
# ══════════════════════════════════════════════════════════════════════════

# Sex-specific coefficients [log_age, log_tc, log_hdl,
#                             log_sbp_treated, log_sbp_untreated,
#                             smoker, diabetes, baseline_survival, mean_coeff_sum]
_FRAMINGHAM_COEFFS = {
    "M": {
        "log_age":           3.06117,
        "log_tc":            1.12370,
        "log_hdl":          -0.93263,
        "log_sbp_treated":   1.99881,
        "log_sbp_untreated": 1.93303,
        "smoker":            0.65451,
        "diabetes":          0.57367,
        "baseline_survival": 0.88936,
        "mean_coeff_sum":    23.9802,
    },
    "F": {
        "log_age":           2.32888,
        "log_tc":            1.20904,
        "log_hdl":          -0.70833,
        "log_sbp_treated":   2.76157,
        "log_sbp_untreated": 2.82263,
        "smoker":            0.52873,
        "diabetes":          0.69154,
        "baseline_survival": 0.95012,
        "mean_coeff_sum":    26.1931,
    },
}

def framingham_cvd_risk(p: Patient) -> float:
    """
    Compute 10-year CVD risk using Framingham 2008 equations.

    Required keys: age, sex (M/F), total_cholesterol, hdl_cholesterol,
                   sbp, bp_treated (bool), smoker (bool), diabetes (bool)
    Returns: probability 0-1
    """
    sex = p.get("sex", "M")
    c   = _FRAMINGHAM_COEFFS.get(sex, _FRAMINGHAM_COEFFS["M"])

    age  = _clamp(float(p.get("age", 50)), 30, 79)
    tc   = _clamp(float(p.get("total_cholesterol", 200)), 130, 320)
    hdl  = _clamp(float(p.get("hdl_cholesterol", 50)), 20, 100)
    sbp  = _clamp(float(p.get("sbp", 120)), 90, 200)
    trt  = bool(p.get("bp_treated", False))
    smk  = bool(p.get("smoker", False))
    dm   = bool(p.get("diabetes", False))

    score = (
        c["log_age"]           * _safe_log(age)
      + c["log_tc"]            * _safe_log(tc)
      + c["log_hdl"]           * _safe_log(hdl)
      + (c["log_sbp_treated"]  * _safe_log(sbp) if trt
         else c["log_sbp_untreated"] * _safe_log(sbp))
      + c["smoker"]            * int(smk)
      + c["diabetes"]          * int(dm)
    )

    risk_10yr = 1.0 - c["baseline_survival"] ** math.exp(
        score - c["mean_coeff_sum"])
    return _clamp(risk_10yr, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 2. ACC/AHA ASCVD POOLED COHORT EQUATIONS (2013)
#    10-year atherosclerotic CVD risk
#    Source: JACC 2013;63:2935-2959 (Goff et al.)
# ══════════════════════════════════════════════════════════════════════════

_ASCVD_COEFFS = {
    ("white", "M"): {
        "ln_age":          12.344,  "ln_tc":         11.853,
        "ln_age_ln_tc":    -2.664,  "ln_hdl":        -7.990,
        "ln_age_ln_hdl":    1.769,  "ln_sbp_trt":     1.797,
        "ln_age_sbp_trt":   0.0,    "ln_sbp_untrt":   1.764,
        "ln_age_sbp_untrt": 0.0,    "smoker":         7.837,
        "ln_age_smoker":   -1.795,  "diabetes":       0.661,
        "baseline":         0.9144, "mean_sum":      61.18,
    },
    ("white", "F"): {
        "ln_age":          -7.574,  "ln_tc":          4.185,
        "ln_age_ln_tc":     0.661,  "ln_hdl":         0.0,
        "ln_age_ln_hdl":   -0.0,    "ln_sbp_trt":     2.019,
        "ln_age_sbp_trt":   0.0,    "ln_sbp_untrt":   1.957,
        "ln_age_sbp_untrt": 0.0,    "smoker":         7.574,
        "ln_age_smoker":   -1.665,  "diabetes":       0.661,
        "baseline":         0.9665, "mean_sum":      -29.799,
    },
    ("black", "M"): {
        "ln_age":           2.469,  "ln_tc":          0.302,
        "ln_age_ln_tc":     0.0,    "ln_hdl":        -0.307,
        "ln_age_ln_hdl":    0.0,    "ln_sbp_trt":     1.916,
        "ln_age_sbp_trt":   0.0,    "ln_sbp_untrt":   1.809,
        "ln_age_sbp_untrt": 0.0,    "smoker":         0.549,
        "ln_age_smoker":    0.0,    "diabetes":       0.645,
        "baseline":         0.8954, "mean_sum":      19.54,
    },
    ("black", "F"): {
        "ln_age":          17.1141, "ln_tc":          0.9396,
        "ln_age_ln_tc":     0.0,    "ln_hdl":        -18.920,
        "ln_age_ln_hdl":    4.475,  "ln_sbp_trt":    29.291,
        "ln_age_sbp_trt":  -6.432,  "ln_sbp_untrt":  27.819,
        "ln_age_sbp_untrt":-6.087,  "smoker":         0.873,
        "ln_age_smoker":    0.0,    "diabetes":       0.874,
        "baseline":         0.9533, "mean_sum":      86.61,
    },
}

def ascvd_risk(p: Patient) -> float:
    """
    ACC/AHA 2013 Pooled Cohort Equations for 10-yr ASCVD risk.

    Required keys: age, sex (M/F), race (white/black/other),
                   total_cholesterol, hdl_cholesterol, sbp,
                   bp_treated (bool), smoker (bool), diabetes (bool)
    Returns: probability 0-1
    """
    sex  = p.get("sex", "M")
    race = p.get("race", "white")
    # Non-white/non-black defaults to white equations per guideline
    race_key = race if race in ("white", "black") else "white"
    key = (race_key, sex)
    c   = _ASCVD_COEFFS.get(key, _ASCVD_COEFFS[("white", "M")])

    age  = _clamp(float(p.get("age", 50)), 40, 79)
    tc   = _clamp(float(p.get("total_cholesterol", 200)), 130, 320)
    hdl  = _clamp(float(p.get("hdl_cholesterol", 50)), 20, 100)
    sbp  = _clamp(float(p.get("sbp", 120)), 90, 200)
    trt  = bool(p.get("bp_treated", False))
    smk  = bool(p.get("smoker", False))
    dm   = bool(p.get("diabetes", False))

    la   = _safe_log(age)
    ltc  = _safe_log(tc)
    lhdl = _safe_log(hdl)
    lsbp = _safe_log(sbp)

    score = (
        c["ln_age"]            * la
      + c["ln_tc"]             * ltc
      + c["ln_age_ln_tc"]      * la * ltc
      + c["ln_hdl"]            * lhdl
      + c["ln_age_ln_hdl"]     * la * lhdl
      + (c["ln_sbp_trt"]       * lsbp
       + c["ln_age_sbp_trt"]   * la * lsbp if trt else
         c["ln_sbp_untrt"]     * lsbp
       + c["ln_age_sbp_untrt"] * la * lsbp)
      + c["smoker"]            * int(smk)
      + c["ln_age_smoker"]     * la * int(smk)
      + c["diabetes"]          * int(dm)
    )

    risk_10yr = 1.0 - c["baseline"] ** math.exp(score - c["mean_sum"])
    return _clamp(risk_10yr, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 3. FINDRISC — Finnish Diabetes Risk Score
#    10-year Type 2 Diabetes risk
#    Source: Diabetologia 2003;46:1070-1078
# ══════════════════════════════════════════════════════════════════════════

def findrisc_t2dm_risk(p: Patient) -> float:
    """
    FINDRISC score → T2DM probability.

    Required keys: age, bmi, waist_cm, sex (M/F),
                   physical_activity_low (bool), diet_low_veg (bool),
                   bp_med (bool), high_glucose_history (bool),
                   family_history_dm (bool: first-degree),
                   family_history_dm_2nd (bool: second-degree)
    Returns: probability 0-1
    """
    score = 0
    age = float(p.get("age", 40))
    if   age < 45:  score += 0
    elif age < 55:  score += 2
    elif age < 65:  score += 3
    else:           score += 4

    bmi = float(p.get("bmi", 25))
    if   bmi < 25:  score += 0
    elif bmi < 30:  score += 1
    else:           score += 3

    sex   = p.get("sex", "M")
    waist = float(p.get("waist_cm", 90))
    if sex == "M":
        if   waist < 94:   score += 0
        elif waist < 102:  score += 3
        else:              score += 4
    else:
        if   waist < 80:   score += 0
        elif waist < 88:   score += 3
        else:              score += 4

    if p.get("physical_activity_low", False):   score += 2
    if p.get("diet_low_veg", False):            score += 1
    if p.get("bp_med", False):                  score += 2
    if p.get("high_glucose_history", False):    score += 5

    fam = p.get("family_history_dm", False)
    fam2 = p.get("family_history_dm_2nd", False)
    if fam:         score += 5
    elif fam2:      score += 3

    # Convert score to approximate 10-yr probability
    # Based on published FINDRISC risk tables
    score_to_prob = {
        0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01,
        5: 0.02, 6: 0.02, 7: 0.03, 8: 0.04, 9: 0.06,
        10: 0.06, 11: 0.09, 12: 0.09, 13: 0.09, 14: 0.17,
        15: 0.17, 16: 0.17, 17: 0.33, 18: 0.33, 19: 0.33,
        20: 0.50, 21: 0.50, 22: 0.50, 23: 0.50, 24: 0.50,
        25: 0.50, 26: 0.50,
    }
    score = min(score, 26)
    return score_to_prob.get(score, 0.50)


# ══════════════════════════════════════════════════════════════════════════
# 4. CKD-EPI eGFR (2021 Race-Free Equation)
#    Chronic Kidney Disease staging
#    Source: NEJM 2021 — race-free CKD-EPI update
# ══════════════════════════════════════════════════════════════════════════

def ckd_epi_egfr(p: Patient) -> float:
    """
    2021 CKD-EPI creatinine equation (race-free per ASN/NKF guidance).

    Required keys: creatinine (mg/dL), age, sex (M/F)
    Returns: eGFR in mL/min/1.73m²
    """
    cr  = _clamp(float(p.get("creatinine", 0.9)), 0.1, 20.0)
    age = _clamp(float(p.get("age", 50)), 18, 110)
    sex = p.get("sex", "M")

    if sex == "F":
        kappa, alpha = 0.7, -0.241
    else:
        kappa, alpha = 0.9, -0.302

    ratio = cr / kappa
    egfr  = (142
             * min(ratio, 1.0) ** alpha
             * max(ratio, 1.0) ** (-1.200)
             * 0.9938 ** age
             * (1.012 if sex == "F" else 1.0))
    return round(egfr, 1)


def ckd_label(p: Patient) -> bool:
    """
    CKD label: True if eGFR < 60 (stages G3a-G5) per KDIGO criteria.
    """
    egfr = ckd_epi_egfr(p)
    return egfr < 60.0


# ══════════════════════════════════════════════════════════════════════════
# 5. STOP-BANG — Obstructive Sleep Apnea Risk
#    Source: Anesthesiology 2008;108:812-821
# ══════════════════════════════════════════════════════════════════════════

def stop_bang_score(p: Patient) -> int:
    """
    STOP-BANG score (0-8). Score ≥ 3 = high OSA risk.

    Required keys: snoring (bool), tired (bool), observed_apnea (bool),
                   sbp (float or bp_high bool), bmi (float),
                   age (float), neck_cm (float), sex (M/F)
    Returns: integer score 0-8
    """
    score = 0
    if p.get("snoring", False):                          score += 1
    if p.get("tired", False):                            score += 1
    if p.get("observed_apnea", False):                   score += 1
    if float(p.get("sbp", 110)) > 140 or \
       p.get("bp_high", False):                          score += 1
    if float(p.get("bmi", 25)) > 35:                    score += 1
    if float(p.get("age", 40)) > 50:                    score += 1
    if float(p.get("neck_cm", 37)) > 40:                score += 1
    if p.get("sex", "F") == "M":                        score += 1
    return score


def osa_risk(p: Patient) -> float:
    """
    OSA probability from STOP-BANG score.
    Score ≥ 3 maps to ~26% population prevalence per NHANES.
    """
    score = stop_bang_score(p)
    # Published sensitivity/specificity mapping
    score_prob = {
        0: 0.03, 1: 0.06, 2: 0.12,
        3: 0.26, 4: 0.36, 5: 0.49,
        6: 0.62, 7: 0.75, 8: 0.83,
    }
    return score_prob.get(score, 0.26)


# ══════════════════════════════════════════════════════════════════════════
# 6. HYPERTENSION LABEL (ACC/AHA 2017)
#    SBP ≥ 130 or DBP ≥ 80
# ══════════════════════════════════════════════════════════════════════════

def hypertension_label(p: Patient) -> bool:
    """
    ACC/AHA 2017: Hypertension defined as SBP ≥ 130 OR DBP ≥ 80.
    Already on antihypertensives also counts.
    """
    sbp = float(p.get("sbp", 110))
    dbp = float(p.get("dbp", 70))
    on_meds = bool(p.get("bp_treated", False))
    return sbp >= 130 or dbp >= 80 or on_meds


# ══════════════════════════════════════════════════════════════════════════
# 7. OBESITY LABEL (CDC BMI Classification)
# ══════════════════════════════════════════════════════════════════════════

def obesity_class(p: Patient) -> str:
    """
    CDC BMI classification.
    Returns: 'normal', 'overweight', 'obese_1', 'obese_2', 'obese_3'
    """
    bmi = float(p.get("bmi", 25))
    if   bmi < 18.5:  return "underweight"
    elif bmi < 25.0:  return "normal"
    elif bmi < 30.0:  return "overweight"
    elif bmi < 35.0:  return "obese_1"
    elif bmi < 40.0:  return "obese_2"
    else:             return "obese_3"


# ══════════════════════════════════════════════════════════════════════════
# 8. DIABETES LABEL (ADA Diagnostic Criteria)
# ══════════════════════════════════════════════════════════════════════════

def diabetes_label(p: Patient) -> str:
    """
    ADA 2023 diagnostic criteria.
    Returns: 'normal', 'prediabetes', 'diabetes'
    """
    hba1c = float(p.get("hba1c", 5.0))
    fpg   = float(p.get("fasting_glucose", 90))
    dx    = bool(p.get("diagnosed_diabetes", False))

    if dx or hba1c >= 6.5 or fpg >= 126:
        return "diabetes"
    elif hba1c >= 5.7 or fpg >= 100:
        return "prediabetes"
    else:
        return "normal"


# ══════════════════════════════════════════════════════════════════════════
# 9. CHARLSON COMORBIDITY INDEX
#    1-year mortality risk proxy
#    Source: J Chronic Dis 1987;40:373-383
# ══════════════════════════════════════════════════════════════════════════

def charlson_index(p: Patient) -> int:
    """
    Charlson Comorbidity Index — weighted sum of 17 conditions + age.
    Used as comorbidity burden feature, not a label.

    Required keys: binary flags for each condition (see below)
    Returns: integer score (higher = greater 1-yr mortality risk)
    """
    score = 0

    # Age contribution
    age = float(p.get("age", 40))
    if   age < 50:  score += 0
    elif age < 60:  score += 1
    elif age < 70:  score += 2
    else:           score += 3

    # 1-point conditions
    one_pt = [
        "myocardial_infarction", "congestive_heart_failure",
        "peripheral_vascular_disease", "cerebrovascular_disease",
        "dementia", "copd", "connective_tissue_disease",
        "peptic_ulcer", "mild_liver_disease",
        "diabetes_uncomplicated",
    ]
    for cond in one_pt:
        if p.get(cond, False):
            score += 1

    # 2-point conditions
    two_pt = [
        "hemiplegia", "moderate_severe_renal_disease",
        "diabetes_with_end_organ_damage",
        "any_tumor", "leukemia", "lymphoma",
    ]
    for cond in two_pt:
        if p.get(cond, False):
            score += 2

    # 3-point: moderate/severe liver disease
    if p.get("moderate_severe_liver_disease", False):
        score += 3

    # 6-point conditions
    six_pt = ["metastatic_solid_tumor", "aids"]
    for cond in six_pt:
        if p.get(cond, False):
            score += 6

    return score


# ══════════════════════════════════════════════════════════════════════════
# CONVENIENCE: RUN ALL MODELS ON A PATIENT RECORD
# ══════════════════════════════════════════════════════════════════════════

def evaluate_all_risk_models(p: Patient) -> Dict[str, Any]:
    """
    Run all risk models on a patient dict and return a results dict.
    Used by the synthetic data generator for label assignment.
    """
    return {
        "framingham_cvd_10yr":  framingham_cvd_risk(p),
        "ascvd_10yr":           ascvd_risk(p),
        "findrisc_t2dm_10yr":   findrisc_t2dm_risk(p),
        "egfr":                 ckd_epi_egfr(p),
        "ckd_label":            ckd_label(p),
        "stop_bang_score":      stop_bang_score(p),
        "osa_risk":             osa_risk(p),
        "hypertension":         hypertension_label(p),
        "obesity_class":        obesity_class(p),
        "diabetes_status":      diabetes_label(p),
        "charlson_index":       charlson_index(p),
    }


# ══════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test patient: 58-year-old Black male, hypertensive, diabetic, smoker
    test_patient = {
        "age": 58, "sex": "M", "race": "black",
        "total_cholesterol": 210, "hdl_cholesterol": 38,
        "sbp": 148, "dbp": 92,
        "bp_treated": True, "smoker": True,
        "diabetes": True, "diagnosed_diabetes": True,
        "bmi": 31.4, "waist_cm": 102,
        "hba1c": 7.8, "fasting_glucose": 162,
        "creatinine": 1.3,
        "physical_activity_low": True,
        "family_history_dm": True,
        "snoring": True, "tired": True,
        "neck_cm": 42,
    }

    print("=" * 55)
    print("Risk Model Self-Test — 58yo Black Male, HTN+DM+Smoker")
    print("=" * 55)

    results = evaluate_all_risk_models(test_patient)
    for model, value in results.items():
        if isinstance(value, float):
            print(f"  {model:<28} {value:.4f}")
        else:
            print(f"  {model:<28} {value}")

    print("\nExpected ranges:")
    print("  framingham_cvd_10yr    > 0.20  (high-risk patient)")
    print("  ascvd_10yr             > 0.15  (high-risk patient)")
    print("  findrisc_t2dm_10yr     > 0.30  (score should be high)")
    print("  egfr                   < 75    (creatinine 1.3, age 58)")
    print("  hypertension           True    (SBP 148)")
    print("  diabetes_status        diabetes (HbA1c 7.8)")
    print("  obesity_class          obese_1  (BMI 31.4)")