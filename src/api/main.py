"""
main.py
-------
FastAPI inference endpoint for GoEMed Health SLM.

Endpoints:
    GET  /health     — health check for ALB
    POST /predict    — patient JSON → condition probabilities
    GET  /labels     — list of supported condition labels
    GET  /version    — model version info
"""

import os
import json
import logging
import time
import boto3
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/model")
S3_BUCKET   = os.getenv("MODEL_S3_BUCKET", "goemed-model-artifacts")
S3_PREFIX   = os.getenv(
    "MODEL_S3_PREFIX",
    "checkpoints/bio_clinicalbert_goemed")
MODEL_VERSION = os.getenv("MODEL_VERSION", "bio-clinicalbert-goemed-v1")
MAX_LENGTH  = 128

LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

LABEL_DISPLAY = {
    "hypertension":       "Hypertension",
    "diabetes":           "Type 2 Diabetes",
    "cvd_risk":           "Cardiovascular Disease Risk",
    "ckd":                "Chronic Kidney Disease",
    "osa":                "Obstructive Sleep Apnea",
    "depression":         "Depression / Anxiety",
    "copd":               "COPD / Asthma",
    "metabolic_syndrome": "Metabolic Syndrome",
    "hypothyroidism":     "Hypothyroidism",
    "prediabetes":        "Prediabetes",
    "colorectal_cancer":  "Colorectal Cancer Risk",
}

RISK_THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════

class PatientInput(BaseModel):
    age:                  float = Field(..., ge=18, le=110,
                                        description="Age in years")
    sex:                  str   = Field(..., pattern="^[MF]$",
                                        description="M or F")
    race:                 Optional[str] = Field(
                              None,
                              description="nh_white/nh_black/hispanic/"
                                          "nh_asian/other_multiracial")
    bmi:                  Optional[float] = Field(None, ge=10, le=80)
    sbp:                  Optional[float] = Field(None, ge=70, le=250,
                              description="Systolic BP (mmHg)")
    dbp:                  Optional[float] = Field(None, ge=40, le=150,
                              description="Diastolic BP (mmHg)")
    total_cholesterol:    Optional[float] = Field(None, ge=50, le=500)
    hdl_cholesterol:      Optional[float] = Field(None, ge=10, le=150)
    hba1c:                Optional[float] = Field(None, ge=3.0, le=20.0)
    fasting_glucose:      Optional[float] = Field(None, ge=40, le=600)
    creatinine:           Optional[float] = Field(None, ge=0.1, le=20.0)
    smoker:               Optional[bool]  = None
    pack_years:           Optional[float] = Field(None, ge=0)
    physical_activity_low:Optional[bool]  = None
    bp_treated:           Optional[bool]  = None
    family_history_dm:    Optional[bool]  = None
    family_history_cvd:   Optional[bool]  = None
    snoring:              Optional[bool]  = None
    alcohol_use:          Optional[bool]  = None
    waist_cm:             Optional[float] = Field(None, ge=40, le=200)


class ConditionPrediction(BaseModel):
    label:       str
    display_name:str
    probability: float
    risk_level:  str   # low / moderate / high


class PredictionResponse(BaseModel):
    predictions:     Dict[str, ConditionPrediction]
    risk_summary:    str
    top_risk_factors:list
    model_version:   str
    inference_time_ms:float
    disclaimer:      str


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════

def download_model_from_s3(local_path: str):
    """Download model from S3 if not present locally."""
    local = Path(local_path)
    if local.exists() and (local / "model.safetensors").exists():
        log.info(f"Model found locally at {local_path}")
        return

    log.info(f"Downloading model from s3://{S3_BUCKET}/{S3_PREFIX}/...")
    local.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key      = obj["Key"]
            filename = key.replace(S3_PREFIX + "/", "")
            if not filename:
                continue
            dest = local / filename
            log.info(f"  Downloading {filename}...")
            s3.download_file(S3_BUCKET, key, str(dest))

    log.info("Model download complete")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Global model state ─────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None


def load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return

    download_model_from_s3(MODEL_PATH)

    log.info(f"Loading model from {MODEL_PATH}...")
    _device    = get_device()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(LABEL_COLS),
    ).to(_device)
    _model.eval()
    log.info(f"Model loaded on {_device}")


# ══════════════════════════════════════════════════════════════════════════
# PROMPT SERIALIZATION
# ══════════════════════════════════════════════════════════════════════════

def patient_to_prompt(p: PatientInput) -> str:
    """Convert patient input to natural language prompt."""
    sex_str  = "male" if p.sex == "M" else "female"
    age_str  = f"{int(p.age)}-year-old"

    race_map = {
        "nh_white":          "White",
        "nh_black":          "Black",
        "hispanic":          "Hispanic",
        "nh_asian":          "Asian",
        "other_multiracial": "Multiracial",
    }
    race_str = race_map.get(p.race or "", "")

    parts = [f"Patient: {age_str} {race_str} {sex_str}.".strip()]

    if p.bmi is not None:
        parts.append(f"BMI: {p.bmi:.1f} kg/m².")
    if p.sbp is not None and p.dbp is not None:
        parts.append(f"Blood pressure: {int(p.sbp)}/{int(p.dbp)} mmHg.")
    elif p.sbp is not None:
        parts.append(f"Systolic BP: {int(p.sbp)} mmHg.")
    if p.total_cholesterol is not None:
        parts.append(f"Total cholesterol: {int(p.total_cholesterol)} mg/dL.")
    if p.hdl_cholesterol is not None:
        parts.append(f"HDL cholesterol: {int(p.hdl_cholesterol)} mg/dL.")
    if p.hba1c is not None:
        parts.append(f"HbA1c: {p.hba1c:.1f}%.")
    if p.fasting_glucose is not None:
        parts.append(f"Fasting glucose: {int(p.fasting_glucose)} mg/dL.")
    if p.creatinine is not None:
        parts.append(f"Creatinine: {p.creatinine:.2f} mg/dL.")
    if p.smoker is not None:
        smoke_str = "current smoker" if p.smoker else "non-smoker"
        if p.pack_years and p.pack_years > 0:
            smoke_str += f" ({int(p.pack_years)} pack-years)"
        parts.append(f"Smoking: {smoke_str}.")
    if p.physical_activity_low is not None:
        act = "low" if p.physical_activity_low else "moderate to high"
        parts.append(f"Physical activity: {act}.")
    if p.bp_treated is not None:
        parts.append(
            f"BP treatment: {'yes' if p.bp_treated else 'no'}.")
    if p.family_history_dm is not None or p.family_history_cvd is not None:
        fh_parts = []
        if p.family_history_dm:
            fh_parts.append("diabetes")
        if p.family_history_cvd:
            fh_parts.append("CVD")
        if fh_parts:
            parts.append(
                f"Family history: {', '.join(fh_parts)}.")
    if p.snoring:
        parts.append("Snoring: yes.")
    if p.alcohol_use:
        parts.append("Alcohol use: yes.")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════

def predict(patient: PatientInput) -> PredictionResponse:
    load_model()
    start = time.time()

    prompt = patient_to_prompt(patient)
    log.info(f"Prompt: {prompt[:100]}...")

    enc = _tokenizer(
        prompt,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    with torch.no_grad():
        logits = _model(**enc).logits

    probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Build predictions
    predictions = {}
    for i, label in enumerate(LABEL_COLS):
        prob = float(probs[i])
        if prob >= 0.70:
            risk = "high"
        elif prob >= 0.40:
            risk = "moderate"
        else:
            risk = "low"

        predictions[label] = ConditionPrediction(
            label=label,
            display_name=LABEL_DISPLAY[label],
            probability=round(prob, 4),
            risk_level=risk,
        )

    # Risk summary
    high_risk = [l for l, p in predictions.items()
                 if p.risk_level == "high"]
    if len(high_risk) >= 3:
        risk_summary = "high"
    elif len(high_risk) >= 1:
        risk_summary = "moderate"
    else:
        risk_summary = "low"

    # Top risk factors (highest probability conditions)
    top_risk = sorted(
        predictions.items(),
        key=lambda x: x[1].probability,
        reverse=True
    )[:3]
    top_risk_factors = [p.display_name for _, p in top_risk
                        if p.probability >= 0.40]

    inference_ms = (time.time() - start) * 1000

    return PredictionResponse(
        predictions=predictions,
        risk_summary=risk_summary,
        top_risk_factors=top_risk_factors,
        model_version=MODEL_VERSION,
        inference_time_ms=round(inference_ms, 1),
        disclaimer=(
            "This is a risk signal for informational purposes only. "
            "It is not a clinical diagnosis. Always consult a "
            "qualified healthcare provider."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="GoEMed Health SLM API",
    description="Multi-label health condition prediction from patient profiles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    load_model()


@app.get("/health")
async def health():
    """ALB health check endpoint."""
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.get("/version")
async def version():
    """Model version info."""
    return {
        "model":        MODEL_VERSION,
        "base_model":   "emilyalsentzer/Bio_ClinicalBERT",
        "n_labels":     len(LABEL_COLS),
        "labels":       LABEL_COLS,
        "macro_auroc":  0.9857,
    }


@app.get("/labels")
async def labels():
    """List supported condition labels."""
    return {
        "labels": [
            {"key": k, "display": v}
            for k, v in LABEL_DISPLAY.items()
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(patient: PatientInput):
    """
    Predict health condition risks from patient profile.

    Accepts a patient feature vector and returns probability scores
    for 11 health conditions with risk level classifications.
    """
    try:
        return predict(patient)
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)