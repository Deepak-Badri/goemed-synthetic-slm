# GoEMed Synthetic Health Data & Small Language Model

Privacy-preserving health risk pipeline built during a Summer 2026 internship with GoEMed.com — a synthetic patient generator calibrated to federal health survey data, feeding a fine-tuned Bio_ClinicalBERT model, deployed as a live inference API on AWS ECS Fargate.

**No real patient records were used at any stage of training.** The model learns from statistically realistic synthetic data calibrated to NHANES, BRFSS, and SEER — avoiding HIPAA exposure and eliminating the multi-month data-use-agreement barrier that blocks most direct-pay healthcare platforms from building risk intelligence.

## What This Project Does

For each user profile received by the API, the system returns probability scores across 11 clinically relevant conditions (Hypertension, Type 2 Diabetes, CVD Risk, CKD, Sleep Apnea, Depression, COPD, Metabolic Risk Composite, Hypothyroidism, Prediabetes, and Colorectal Cancer). These risk signals are intended to power personalized procedure recommendations on the GoEMed marketplace — not to serve as clinical diagnoses.

## Pipeline Architecture

```
NHANES + BRFSS + SEER + Census
             ↓
Synthetic Patient Generator (Gaussian copula, 100k records)
             ↓
    Prompt Serialization (5 templates)
             ↓
   Bio_ClinicalBERT Fine-Tuning (110M params)
             ↓
      AWS ECS Fargate + Auto CI/CD
             ↓
     Live REST API — 690ms inference
```

## Data Sources

| Source | Coverage | Purpose |
|---|---|---|
| NHANES 2017–2020 & 2021–2023 | 17,846 adults | Distribution fitting, correlation matrices |
| BRFSS 2024 | 457,670 respondents | Binary feature calibration |
| NCI SEER 2018–2022 | 50 states | Colorectal cancer stratified rates |
| Census ACS 2022 | US population | Demographic sampling weights |
| AHA 2023 Statistical Update | Nationwide | Label prevalence validation targets |

## Methods

**Synthetic data generation:** Demographics sampled from Census weights; clinical features sampled jointly via Gaussian copula across 12 race-sex strata to preserve real correlation structure (BMI ↔ blood pressure, HbA1c ↔ fasting glucose, etc.); binary features from BRFSS rates; labels assigned via nine validated clinical risk equations (Framingham 2008, ACC/AHA ASCVD 2013, FINDRISC 2003, CKD-EPI 2021, STOP-BANG 2008, JNC8/ACC-AHA 2017, ADA diagnostic criteria, CDC BMI classification, Charlson Comorbidity Index).

**Validation:** Four independent statistical tests — Kolmogorov-Smirnov (marginals), Frobenius norm (correlation structure), propensity score AUROC (indistinguishability from real data), Hotelling T² (multivariate mean equality). Label prevalences calibrated to within ±5 percentage points of published epidemiological benchmarks across all 12 labels.

**Modeling:** XGBoost tabular baselines and Bio_ClinicalBERT (110M parameter model pre-trained on 880M words of MIMIC III clinical notes) fine-tuned for multi-label classification. SHAP TreeSHAP for interpretability. Fairness audit across 33 demographic subgroups (5 races × 2 sexes × 4 age groups).

## Results

| Metric | Value | Notes |
|---|---|---|
| Frobenius norm (correlation structure) | 1.917 | Threshold < 2.0 |
| Propensity AUROC (real vs. synthetic) | 0.586 | Threshold < 0.65 (ideal ~0.5) |
| Hotelling T² (normalized) | 0.020 | Threshold < 1.0 |
| Label prevalence calibration | 12/12 labels within ±5% | |
| XGBoost macro AUROC (learned labels) | 0.9032 | |
| Bio_ClinicalBERT macro AUROC | 0.6788 | |
| SHAP clinical alignment | 72.4% | Top features match published risk factors |
| Fairness audit pass rate | 87.9% | 4/33 flagged gaps, all clinically justified |
| API inference latency | 690ms | 0.5 vCPU, 1GB RAM |
| Monthly deployment cost | ~$20 | ECS Fargate + S3 + CloudWatch |

XGBoost outperforms Bio_ClinicalBERT on this specific dataset — a result consistent with recent literature: gradient-boosted trees remain competitive or superior to language models when input features are structured numeric clinical values with well-defined thresholds, since tabular models can access these thresholds directly rather than recovering them from text paraphrase.

## Deployment

Live REST API on AWS ECS Fargate:

```bash
POST http://[api-ip]:8000/predict
Content-Type: application/json

{
  "age": 58, "sex": "M", "race": "nh_black",
  "bmi": 31.4, "sbp": 148, "dbp": 92,
  "hba1c": 7.2, "fasting_glucose": 142
}

→ Returns 11 condition probabilities + risk summary + top risk factors
```

Full stack:
- **Storage:** S3 for model weights (413MB) and synthetic dataset
- **Container:** Docker image on ECR, linux/amd64 built for Fargate
- **Compute:** ECS Fargate serverless (0.5 vCPU / 1GB RAM)
- **Security:** IAM least-privilege role, VPC security groups
- **CI/CD:** GitHub Actions auto-deploy on push to main (4-min pipeline)
- **Observability:** CloudWatch logs + custom dashboard

## Repository Structure

```
goemed-synthetic-slm/
├── src/
│   ├── generator/       # Synthetic patient generation
│   ├── models/          # Baseline ML + SLM training
│   ├── validation/      # Statistical validation suite
│   └── api/             # FastAPI inference server
├── tests/               # 32 reference-patient + regression tests
├── configs/             # Feature schemas, correlation matrices
├── notebooks/           # Exploratory data analysis
├── data/                # Federal survey data (gitignored)
├── outputs/
│   ├── reports/         # Executive summary, final report, HTML
│   ├── figures/         # Validation plots, SHAP charts
│   └── checkpoints/     # Bio_ClinicalBERT weights (gitignored)
├── infra/               # AWS deployment configs and scripts
├── docs/                # Design decisions, methodology notes
├── .github/workflows/   # CI/CD pipeline (auto-deploy on push)
├── Dockerfile
├── requirements.txt
└── requirements-api.txt # Inference container dependencies
```

## Reports

Three complementary deliverables in `outputs/reports/`:
- `GoEMed_Executive_Summary_Deepak_Badri_Summer2026.docx` — 2-page overview
- `GoEMed_Project_Summary_Deepak_Badri_Summer2026.docx` — 8-page walkthrough
- `GoEMed_Internship_Final_Report_Deepak_Badri_Summer2026.docx` — detailed technical report
- `GoEMed_Internship_Report.html` — self-contained HTML report with embedded diagrams

## Validation Scope

This project's evaluation addresses one question — is the synthetic data statistically valid? — thoroughly. All performance metrics were computed on a held-out synthetic test set. The complementary question — does the model predict outcomes accurately in real patients? — is a natural extension and the highest-priority v2 work item. See Section 9 of the final report for the proposed rigorous validation protocol (NHANES cycle split, calibration alongside discrimination, direct comparison against published equations, negative control experiment).

## What's Next

- Rigorous v2 validation study using NHANES cycle-split protocol
- Geographic health risk intelligence — state-specific SEER cancer multipliers integrated into generator (Cancer Alley focus)
- Iron deficiency anemia and heart failure labels via NHANES CBC data
- Hybrid inference — XGBoost for threshold conditions, SLM for complex ones
- Production HTTPS deployment with ALB, ACM cert, ECS Auto Scaling

## Coursework Connections

- **Stat 309/310** — MLE distribution fitting, Cramér-Rao bounds, power-law learning curves
- **Applied Multivariate Analysis** — Gaussian copula, PSD verification, Hotelling T² test
- **Math 321/322** — Convergence properties of the copula sampler
- **Intro to Number Theory** — Foundation for planned quasi-random Sobol sampling
- **Math Methods in Data Science** — Kernel-based MMD as distributional similarity metric

## Tech Stack

**Languages:** Python 3.11, Bash  
**ML/Data:** PyTorch, Transformers (HuggingFace), XGBoost, scikit-learn, SHAP, pandas, NumPy  
**Statistics:** SciPy, statsmodels  
**Infrastructure:** AWS (S3, ECR, ECS Fargate, IAM, CloudWatch), Docker, GitHub Actions  
**API:** FastAPI, Uvicorn  
**Frontend (companion demo):** React

## Acknowledgments

Supervisor: **Vasan Purighalla**, GoEMed.com  
Institution: **University of Wisconsin–Madison**, Mathematics & Statistics  
Period: Summer 2026

## Disclaimer

This system produces **risk signals for informational purposes only**. It is not a clinical diagnosis and is not intended to replace consultation with a qualified healthcare provider. The model was trained on synthetic data and has not been validated against real patient outcomes.
