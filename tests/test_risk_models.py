"""
Reference-patient tests for clinical risk equations.

Each test verifies that a risk equation reproduces a worked example from the
source publication or an authoritative online calculator, within a documented
tolerance.

Purpose: catch coefficient errors, transposed terms, and missing components
before any downstream regeneration or training.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generator.risk_models import (
    framingham_cvd_risk,
    ascvd_risk,
    findrisc_t2dm_risk,
    ckd_epi_egfr,
    stop_bang_score,
    hypertension_label,
    obesity_class,
    diabetes_label,
    charlson_index,
)


# ══════════════════════════════════════════════════════════════════════════
# ASCVD REFERENCE PATIENTS
# Source: ACC ASCVD Risk Estimator Plus (tools.acc.org) worked examples
# ══════════════════════════════════════════════════════════════════════════

def test_ascvd_white_female_reference():
    """
    White female, age 55, non-smoker, no diabetes, no BP treatment,
    TC 213, HDL 50, SBP 120.
    Published 10-year ASCVD risk: ~2.1%
    """
    p = {
        "age": 55, "sex": "F", "race": "white",
        "total_cholesterol": 213, "hdl_cholesterol": 50,
        "sbp": 120, "bp_treated": False,
        "smoker": False, "diabetes": False,
    }
    risk = ascvd_risk(p)
    assert 0.01 <= risk <= 0.05, \
        f"White female low-risk: expected ~2%, got {risk:.4f}"


def test_ascvd_white_female_high_risk():
    """
    White female, age 65, smoker, diabetic, on BP meds,
    TC 250, HDL 40, SBP 150.
    Published 10-year ASCVD risk: ~25-35%
    """
    p = {
        "age": 65, "sex": "F", "race": "white",
        "total_cholesterol": 250, "hdl_cholesterol": 40,
        "sbp": 150, "bp_treated": True,
        "smoker": True, "diabetes": True,
    }
    risk = ascvd_risk(p)
    assert 0.20 <= risk <= 0.45, \
        f"White female high-risk: expected 20-45%, got {risk:.4f}"


def test_ascvd_black_male_reference():
    """
    Black male, age 55, non-smoker, no diabetes, no BP treatment,
    TC 213, HDL 50, SBP 120.
    Published 10-year ASCVD risk: ~4-6%
    """
    p = {
        "age": 55, "sex": "M", "race": "black",
        "total_cholesterol": 213, "hdl_cholesterol": 50,
        "sbp": 120, "bp_treated": False,
        "smoker": False, "diabetes": False,
    }
    risk = ascvd_risk(p)
    assert 0.02 <= risk <= 0.10, \
        f"Black male reference: expected 4-6%, got {risk:.4f}"


def test_ascvd_white_male_reference():
    """
    White male, age 55, non-smoker, no diabetes, no BP treatment,
    TC 213, HDL 50, SBP 120.
    Published 10-year ASCVD risk: ~5-6%
    """
    p = {
        "age": 55, "sex": "M", "race": "white",
        "total_cholesterol": 213, "hdl_cholesterol": 50,
        "sbp": 120, "bp_treated": False,
        "smoker": False, "diabetes": False,
    }
    risk = ascvd_risk(p)
    assert 0.03 <= risk <= 0.10, \
        f"White male reference: expected 5-6%, got {risk:.4f}"


# ══════════════════════════════════════════════════════════════════════════
# FRAMINGHAM REFERENCE PATIENTS
# Source: Circulation 2008;117:743-753 worked examples
# ══════════════════════════════════════════════════════════════════════════

def test_framingham_female_treated_hypertensive():
    """
    Female, age 55, treated hypertensive, non-smoker, no diabetes,
    TC 250, HDL 50, SBP 130 (on medication).
    Published 10-year CVD risk: ~6-8%
    """
    p = {
        "age": 55, "sex": "F",
        "total_cholesterol": 250, "hdl_cholesterol": 50,
        "sbp": 130, "bp_treated": True,
        "smoker": False, "diabetes": False,
    }
    risk = framingham_cvd_risk(p)
    assert 0.03 <= risk <= 0.12, \
        f"Female treated HTN: expected 6-8%, got {risk:.4f}"


def test_framingham_male_reference():
    """
    Male, age 55, non-smoker, no diabetes, no BP treatment,
    TC 210, HDL 45, SBP 130.
    Published 10-year CVD risk: ~10-12%
    """
    p = {
        "age": 55, "sex": "M",
        "total_cholesterol": 210, "hdl_cholesterol": 45,
        "sbp": 130, "bp_treated": False,
        "smoker": False, "diabetes": False,
    }
    risk = framingham_cvd_risk(p)
    assert 0.06 <= risk <= 0.16, \
        f"Male reference: expected 10-12%, got {risk:.4f}"


def test_framingham_female_low_risk():
    """
    Female, age 45, non-smoker, no diabetes, no BP treatment,
    TC 180, HDL 60, SBP 110.
    Published 10-year CVD risk: <2%
    """
    p = {
        "age": 45, "sex": "F",
        "total_cholesterol": 180, "hdl_cholesterol": 60,
        "sbp": 110, "bp_treated": False,
        "smoker": False, "diabetes": False,
    }
    risk = framingham_cvd_risk(p)
    assert risk < 0.03, \
        f"Female low-risk: expected <2%, got {risk:.4f}"


# ══════════════════════════════════════════════════════════════════════════
# CKD-EPI 2021 REFERENCE PATIENTS
# Source: NEJM 2021 supplementary tables
# ══════════════════════════════════════════════════════════════════════════

def test_ckd_epi_normal_male():
    """
    Male, age 40, creatinine 1.0 mg/dL.
    Expected eGFR: ~99 mL/min/1.73m²
    """
    p = {"age": 40, "sex": "M", "creatinine": 1.0}
    egfr = ckd_epi_egfr(p)
    assert 90 <= egfr <= 110, \
        f"Normal male eGFR: expected ~99, got {egfr}"


def test_ckd_epi_normal_female():
    """
    Female, age 40, creatinine 0.8 mg/dL.
    Expected eGFR: ~95 mL/min/1.73m²
    """
    p = {"age": 40, "sex": "F", "creatinine": 0.8}
    egfr = ckd_epi_egfr(p)
    assert 85 <= egfr <= 105, \
        f"Normal female eGFR: expected ~95, got {egfr}"


def test_ckd_epi_ckd_stage_3():
    """
    Male, age 65, creatinine 1.8 mg/dL.
    Expected eGFR: ~40 mL/min/1.73m² (CKD stage 3b)
    """
    p = {"age": 65, "sex": "M", "creatinine": 1.8}
    egfr = ckd_epi_egfr(p)
    assert 30 <= egfr <= 50, \
        f"CKD stage 3 male: expected ~40, got {egfr}"


# ══════════════════════════════════════════════════════════════════════════
# FINDRISC BAND MAPPING
# Source: Lindström & Tuomilehto Diabetes Care 2003;26:725-731
# ══════════════════════════════════════════════════════════════════════════

def test_findrisc_very_low_risk():
    """Young, lean, active, no family history → score <7 → 1% risk"""
    p = {
        "age": 30, "bmi": 22, "waist_cm": 75, "sex": "F",
        "physical_activity_low": False, "diet_low_veg": False,
        "bp_med": False, "high_glucose_history": False,
        "family_history_dm": False, "family_history_dm_2nd": False,
    }
    prob = findrisc_t2dm_risk(p)
    assert prob == 0.01, f"Very low: expected 1%, got {prob}"


def test_findrisc_high_risk():
    """Older, obese, sedentary, family history → high score → 33% risk"""
    p = {
        "age": 60, "bmi": 33, "waist_cm": 105, "sex": "M",
        "physical_activity_low": True, "diet_low_veg": True,
        "bp_med": True, "high_glucose_history": False,
        "family_history_dm": True, "family_history_dm_2nd": False,
    }
    prob = findrisc_t2dm_risk(p)
    assert prob == 0.33, f"High risk: expected 33%, got {prob}"


# ══════════════════════════════════════════════════════════════════════════
# STOP-BANG
# Source: Anesthesiology 2008;108:812-821
# ══════════════════════════════════════════════════════════════════════════

def test_stopbang_no_risk():
    """All negatives → score 0"""
    p = {
        "snoring": False, "tired": False, "observed_apnea": False,
        "sbp": 110, "bmi": 22, "age": 30, "neck_cm": 35, "sex": "F",
    }
    assert stop_bang_score(p) == 0


def test_stopbang_high_risk():
    """All positives → score 8"""
    p = {
        "snoring": True, "tired": True, "observed_apnea": True,
        "sbp": 150, "bmi": 40, "age": 60, "neck_cm": 45, "sex": "M",
    }
    assert stop_bang_score(p) == 8


# ══════════════════════════════════════════════════════════════════════════
# HYPERTENSION (ACC/AHA 2017)
# ══════════════════════════════════════════════════════════════════════════

def test_hypertension_normal_bp():
    """SBP 115, DBP 75, no meds → False"""
    p = {"sbp": 115, "dbp": 75, "bp_treated": False}
    assert hypertension_label(p) is False


def test_hypertension_elevated_sbp():
    """SBP 135, DBP 78, no meds → True (SBP threshold)"""
    p = {"sbp": 135, "dbp": 78, "bp_treated": False}
    assert hypertension_label(p) is True


def test_hypertension_elevated_dbp():
    """SBP 125, DBP 82, no meds → True (DBP threshold)"""
    p = {"sbp": 125, "dbp": 82, "bp_treated": False}
    assert hypertension_label(p) is True


def test_hypertension_on_medication():
    """Normal BP but on meds → True"""
    p = {"sbp": 118, "dbp": 74, "bp_treated": True}
    assert hypertension_label(p) is True


# ══════════════════════════════════════════════════════════════════════════
# DIABETES (ADA)
# ══════════════════════════════════════════════════════════════════════════

def test_diabetes_normal():
    p = {"hba1c": 5.2, "fasting_glucose": 92}
    assert diabetes_label(p) == "normal"


def test_diabetes_prediabetes_hba1c():
    p = {"hba1c": 6.0, "fasting_glucose": 95}
    assert diabetes_label(p) == "prediabetes"


def test_diabetes_prediabetes_fpg():
    p = {"hba1c": 5.5, "fasting_glucose": 115}
    assert diabetes_label(p) == "prediabetes"


def test_diabetes_diagnosis_hba1c():
    p = {"hba1c": 7.2, "fasting_glucose": 120}
    assert diabetes_label(p) == "diabetes"


def test_diabetes_diagnosis_fpg():
    p = {"hba1c": 6.0, "fasting_glucose": 140}
    assert diabetes_label(p) == "diabetes"


# ══════════════════════════════════════════════════════════════════════════
# BMI CLASSIFICATION (CDC)
# ══════════════════════════════════════════════════════════════════════════

def test_bmi_normal():
    assert obesity_class({"bmi": 22}) == "normal"


def test_bmi_overweight():
    assert obesity_class({"bmi": 27}) == "overweight"


def test_bmi_class_1_obesity():
    assert obesity_class({"bmi": 32}) == "obese_1"


def test_bmi_class_3_obesity():
    assert obesity_class({"bmi": 42}) == "obese_3"


# ══════════════════════════════════════════════════════════════════════════
# CHARLSON COMORBIDITY INDEX
# Source: Charlson et al. J Chronic Dis 1987;40:373-383
# ══════════════════════════════════════════════════════════════════════════

def test_charlson_healthy_young():
    """Age <50, no conditions → 0"""
    p = {"age": 40}
    assert charlson_index(p) == 0


def test_charlson_age_only():
    """Age 55 no conditions → 1 (age tier)"""
    assert charlson_index({"age": 55}) == 1
    assert charlson_index({"age": 65}) == 2
    assert charlson_index({"age": 75}) == 3
    assert charlson_index({"age": 85}) == 4  # ≥80 tier


def test_charlson_diabetes_uncomplicated():
    """Age 55, DM uncomplicated → 1 + 1 = 2"""
    p = {"age": 55, "diabetes_uncomplicated": True}
    assert charlson_index(p) == 2


def test_charlson_metastatic_cancer():
    """Age 70, metastatic tumor → 3 (age 70-79) + 6 (mets) = 9"""
    p = {"age": 70, "metastatic_solid_tumor": True}
    assert charlson_index(p) == 9

# ══════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import traceback

    test_funcs = [
        (name, func) for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func)
    ]

    passed, failed = 0, []
    for name, func in test_funcs:
        try:
            func()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}")
            print(f"     {e}")
            failed.append(name)
        except Exception as e:
            print(f"  💥 {name} — {type(e).__name__}: {e}")
            failed.append(name)

    print()
    print(f"  Passed: {passed}/{len(test_funcs)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        sys.exit(1)
    print("  All tests passed ✅")