# GoEMed Synthetic Health Data & SLM

Synthetic patient data generation pipeline and Bio_ClinicalBERT fine-tuning
for multi-label health condition prediction.

## Setup
```zsh
conda activate goemed
```

## Project Structure
- `data/` — raw sources, processed features, train/val/test splits
- `src/generator/` — Gaussian copula synthetic data engine
- `src/models/` — SLM fine-tuning and baseline ML models
- `src/validation/` — statistical fidelity tests (KS, MMD, prevalence)
- `src/api/` — FastAPI inference endpoint
- `configs/` — YAML distribution parameters and risk model coefficients
- `infra/terraform/` — AWS infrastructure as code
- `notebooks/` — exploration and validation notebooks
- `tests/` — unit tests for all generator modules
