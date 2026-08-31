# Pre-Remediation Baseline Numbers

**Date frozen:** 2026-08-30
**Git tag:** pre-remediation-baseline
**Outputs snapshot:** outputs_pre_fix/

## Headline Performance Metrics (Pre-Remediation)

| Metric | Value |
|---|---|
| SLM (Bio_ClinicalBERT) macro AUROC | 0.9857 |
| XGBoost macro AUROC | 0.819 |
| Logistic Regression macro AUROC | 0.779 |

## Per-Label Bio_ClinicalBERT AUROC (Pre-Remediation)

| Label | AUROC |
|---|---|
| Hypertension | 0.999 |
| Type 2 Diabetes | 0.997 |
| CVD Risk | 0.986 |
| Chronic Kidney Disease | 0.993 |
| Sleep Apnea (OSA) | 0.986 |
| Depression/Anxiety | 0.975 |
| COPD | 0.980 |
| Metabolic Syndrome | 0.991 |
| Hypothyroidism | 0.978 |
| Prediabetes | 0.995 |
| Colorectal Cancer | 0.962 |

## Label Prevalence Calibration (Pre-Remediation)

| Label | Synthetic | Benchmark | Delta |
|---|---|---|---|
| Hypertension | 48.0% | 47.0% | +1.0% |
| Type 2 Diabetes | 14.3% | 11.0% | +3.3% |
| Obesity | 43.8% | 42.0% | +1.8% |
| CVD Risk | 14.7% | 12.0% | +2.7% |
| CKD | 12.4% | 15.0% | -2.6% |
| OSA | 23.5% | 26.0% | -2.5% |
| Depression | 10.7% | 8.0% | +2.7% |
| COPD | 7.8% | 6.0% | +1.8% |
| Metabolic Syndrome | 36.0% | 33.0% | +3.0% |
| Hypothyroidism | 7.3% | 5.0% | +2.3% |
| Prediabetes | 34.0% | 38.0% | -4.0% |
| Colorectal Cancer | 3.7% | 4.5% | -0.8% |

## Fairness Audit (Pre-Remediation)

- Pass rate: 78.8% (26/33 subgroup checks)
- CVD Risk sex gap: 19.8 percentage points
- Colorectal Cancer race gap: 12.8 pp

## Power Analysis (Pre-Remediation)

- Bio_ClinicalBERT saturates at ~1,000 records (0.9861 AUROC)
- Depression AUROC at 1k records: 0.975
- Hypothyroidism AUROC at 1k records: 0.978

## Notes

These numbers are preserved as the "before" state for comparison against
post-remediation results. Known issues being remediated:

1. Label text leaking into prompts via serialize_prompts.py
2. ASCVD white-female coefficients incorrect
3. Framingham female SBP treated/untreated transposed
4. Metabolic syndrome uses non-ATP-III definition
5. Circular labels not handled in baselines

See P0 Remediation Runbook for full detail.
