# Metabolic Syndrome Label — Design Decision

**Date:** 2026-08-30
**Decision:** Option A — Rename the label; keep current logic; document honestly.
**Rationale:** P0 scope is limited to making existing work honest, not adding new features.

## Background

The current implementation was labeled as "ATP III Metabolic Syndrome" but does not
match the published ATP III criteria (NCEP ATP III 2001, updated 2005).

## Comparison

| Criterion | Published ATP III | Current Implementation |
|---|---|---|
| Waist circumference | ≥ 102 cm (M), ≥ 88 cm (F) | ≥ 98 cm (M), ≥ 84 cm (F) |
| Fasting glucose | ≥ 100 mg/dL | ≥ 100 mg/dL ✓ |
| HDL cholesterol | < 40 mg/dL (M), < 50 mg/dL (F) | < 40 mg/dL (M), < 50 mg/dL (F) ✓ |
| Blood pressure | ≥ 130/85 mmHg | ≥ 130/85 mmHg ✓ |
| Triglycerides | ≥ 150 mg/dL | **Total cholesterol ≥ 220 mg/dL** (substituted) |

## Decision

Rename the label from "Metabolic Syndrome (ATP III)" to a name that reflects what
it actually measures. The current logic is retained.

**New name:** `metabolic_risk_composite`
**Display name:** "Metabolic Risk Composite"

## Documentation Requirement

Anywhere this label appears in the codebase, generator, or report, it must be described
as a composite metabolic risk score using proxy criteria (not published ATP III), with
the exact definition stated.

## Path to True ATP III (Future P1 Work)

Implementing published ATP III requires:

1. Add triglycerides to `configs/feature_schema.yaml`
2. Add triglycerides row/column to all 12 correlation matrices in `configs/correlation_matrix.yaml`
3. Fit triglycerides distribution parameters from NHANES BIOPRO module
4. Update waist thresholds to 102/88 cm
5. Replace total_cholesterol substitution with triglycerides ≥ 150 mg/dL
6. Regenerate 100k dataset with new feature vector
7. Retrain all downstream models

Estimated effort: 1 full day of work. Not in scope for P0.

## Impact on Existing Results

Renaming does not require regeneration. Results are unchanged; only naming and
documentation are updated. The prevalence of ~36% observed in the current
synthetic data is representative of what the current logic produces — not what
published ATP III would produce.
