"""
train_slm.py
------------
Fine-tunes Bio_ClinicalBERT on synthetic patient JSONL data
for multi-label health condition prediction.

Architecture:
    - Base: emilyalsentzer/Bio_ClinicalBERT (BERT-Base, 110M params)
    - Head: Linear(768 -> 12) + Sigmoid per label
    - Loss: Binary Cross-Entropy with class-frequency inverse weighting
    - Optimizer: AdamW with cosine LR schedule
    - Device: MPS (Apple M2)

Output:
    - outputs/checkpoints/bio_clinicalbert_goemed/
    - outputs/reports/slm_evaluation.csv
    - outputs/figures/slm_auroc.png
    - outputs/figures/slm_learning_curves.png
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

# ── MPS fallback ───────────────────────────────────────────────────────────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
SPLITS_DIR   = REPO_ROOT / "data/splits"
CHECKPOINTS  = REPO_ROOT / "outputs/checkpoints/bio_clinicalbert_goemed"
FIGURES      = REPO_ROOT / "outputs/figures"
REPORTS      = REPO_ROOT / "outputs/reports"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_ID    = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH  = 128
BATCH_SIZE  = 8
EPOCHS      = 5
LR          = 2e-5
WARMUP_FRAC = 0.1
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

LABEL_DISPLAY = {
    "hypertension":       "Hypertension",
    "diabetes":           "Diabetes",
    "cvd_risk":           "CVD Risk",
    "ckd":                "CKD",
    "osa":                "Sleep Apnea",
    "depression":         "Depression",
    "copd":               "COPD",
    "metabolic_syndrome": "Metabolic Syndrome",
    "hypothyroidism":     "Hypothyroidism",
    "prediabetes":        "Prediabetes",
    "colorectal_cancer":  "Colorectal Cancer",
}


# ══════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ══════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        log.info("Device: MPS (Apple Silicon GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        log.info("Device: CUDA GPU")
        return torch.device("cuda")
    else:
        log.info("Device: CPU")
        return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class PatientDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer,
                 max_length: int, max_records: int = None):
        self.records = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                rec = json.loads(line)
                self.records.append(rec)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        text   = rec["text"]
        labels = rec["labels"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_vector = torch.tensor(
            [float(labels.get(col, 0)) for col in LABEL_COLS],
            dtype=torch.float32,
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         label_vector,
        }

# ══════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════

def compute_class_weights(train_path: Path) -> torch.Tensor:
    """Compute inverse frequency weights for imbalanced labels."""
    label_sums = np.zeros(len(LABEL_COLS))
    total      = 0

    with open(train_path) as f:
        for line in f:
            rec = json.loads(line)
            for i, col in enumerate(LABEL_COLS):
                label_sums[i] += float(rec["labels"].get(col, 0))
            total += 1

    prevalences  = label_sums / total
    # pos_weight = neg_count / pos_count for BCEWithLogitsLoss
    pos_weights  = (1 - prevalences) / np.maximum(prevalences, 1e-6)
    pos_weights  = np.clip(pos_weights, 1.0, 20.0)  # cap at 20x

    log.info("Class weights (pos_weight per label):")
    for col, pw, prev in zip(LABEL_COLS, pos_weights, prevalences):
        log.info(f"  {col:<25} prevalence={prev:.1%}  "
                 f"pos_weight={pw:.2f}")

    return torch.tensor(pos_weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(
    model, loader, optimizer, scheduler,
    loss_fn, device, epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        loss   = loss_fn(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            log.info(f"  Epoch {epoch} [{batch_idx+1}/{n_batches}] "
                     f"loss={loss.item():.4f}")

    return total_loss / n_batches


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def evaluate(
    model, loader, loss_fn, device,
) -> Dict:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            loss   = loss_fn(logits, labels)
            total_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.sigmoid(torch.tensor(logits)).numpy()

    # Per-label metrics
    results = {}
    for i, col in enumerate(LABEL_COLS):
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            auroc = roc_auc_score(y_true, y_prob)
        except Exception:
            auroc = float("nan")

        results[col] = {
            "auroc": auroc,
            "f1":    f1_score(y_true, y_pred, zero_division=0),
            "brier": brier_score_loss(y_true, y_prob),
        }

    macro_auroc = np.nanmean([v["auroc"] for v in results.values()])
    macro_f1    = np.mean([v["f1"] for v in results.values()])

    return {
        "loss":        total_loss / len(loader),
        "macro_auroc": macro_auroc,
        "macro_f1":    macro_f1,
        "per_label":   results,
        "probs":       probs,
        "labels":      labels,
    }


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: List[Dict]):
    epochs = list(range(1, len(history) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(epochs, [h["train_loss"] for h in history],
                 "b-o", label="Train")
    axes[0].plot(epochs, [h["val_loss"] for h in history],
                 "r-o", label="Val")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_macro_auroc"] for h in history],
                 "g-o")
    axes[1].set_title("Val Macro AUROC", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.5, 1.0)

    axes[2].plot(epochs, [h["val_macro_f1"] for h in history],
                 "m-o")
    axes[2].set_title("Val Macro F1", fontweight="bold")
    axes[2].set_xlabel("Epoch")

    plt.suptitle("Bio_ClinicalBERT Training Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "slm_training_curves.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/slm_training_curves.png")


def plot_auroc_comparison(slm_results: Dict, baseline_path: Path):
    """Compare SLM AUROC vs XGBoost baseline per label."""
    baseline = pd.read_csv(baseline_path)
    xgb = baseline[baseline["model"] == "XGBoost"].set_index("label")

    labels  = LABEL_COLS
    slm_aurocs = [slm_results["per_label"][l]["auroc"] for l in labels]
    xgb_aurocs = [xgb.loc[l, "auroc"] if l in xgb.index
                  else 0 for l in labels]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width/2, xgb_aurocs, width,
           label="XGBoost (baseline)", color="#378ADD", alpha=0.85)
    ax.bar(x + width/2, slm_aurocs, width,
           label="Bio_ClinicalBERT", color="#1D9E75", alpha=0.85)

    ax.axhline(0.7, color="gray", linestyle="--",
               linewidth=1, label="0.70 threshold")
    ax.set_xlabel("Condition Label")
    ax.set_ylabel("AUROC")
    ax.set_title("SLM vs XGBoost Baseline — AUROC by Condition",
                 fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABEL_DISPLAY.get(l, l) for l in labels],
        rotation=30, ha="right"
    )
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "slm_vs_baseline_auroc.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/slm_vs_baseline_auroc.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Bio_ClinicalBERT Fine-Tuning — GoEMed Health SLM")
    log.info("=" * 55)

    device = get_device()

    # ── Load tokenizer and model ───────────────────────────────────────
    log.info(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )
    model = model.to(device)
    log.info(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Datasets and loaders ──────────────────────────────────────────
    log.info("\nLoading datasets...")
    # Limit dataset size for M2 8GB RAM
    # Power analysis will train on multiple sizes
    TRAIN_SIZE = 20_000
    VAL_SIZE   = 4_000
    TEST_SIZE  = 4_000

    train_ds = PatientDataset(
        SPLITS_DIR / "train.jsonl", tokenizer, MAX_LENGTH,
        max_records=TRAIN_SIZE)
    val_ds   = PatientDataset(
        SPLITS_DIR / "val.jsonl",   tokenizer, MAX_LENGTH,
        max_records=VAL_SIZE)
    test_ds  = PatientDataset(
        SPLITS_DIR / "test.jsonl",  tokenizer, MAX_LENGTH,
        max_records=TEST_SIZE)

    log.info(f"Train: {len(train_ds):,}  "
             f"Val: {len(val_ds):,}  "
             f"Test: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0)
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=0)
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────────────
    log.info("\nComputing class weights...")
    class_weights = compute_class_weights(
        SPLITS_DIR / "train.jsonl").to(device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # ── Optimizer and scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01)

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_FRAC)
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    log.info(f"\nTraining config:")
    log.info(f"  Epochs:        {EPOCHS}")
    log.info(f"  Batch size:    {BATCH_SIZE}")
    log.info(f"  LR:            {LR}")
    log.info(f"  Total steps:   {total_steps:,}")
    log.info(f"  Warmup steps:  {warmup_steps:,}")
    log.info(f"  Max length:    {MAX_LENGTH}")
    log.info(f"  Labels:        {len(LABEL_COLS)}")

    # ── Training loop ─────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("Starting training...")
    log.info("=" * 55)

    history      = []
    best_auroc   = 0.0
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        log.info(f"\n── Epoch {epoch}/{EPOCHS} ──")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch)

        val_results = evaluate(model, val_loader, loss_fn, device)

        log.info(f"  Train loss:     {train_loss:.4f}")
        log.info(f"  Val loss:       {val_results['loss']:.4f}")
        log.info(f"  Val macro AUROC:{val_results['macro_auroc']:.4f}")
        log.info(f"  Val macro F1:   {val_results['macro_f1']:.4f}")

        history.append({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "val_loss":       val_results["loss"],
            "val_macro_auroc":val_results["macro_auroc"],
            "val_macro_f1":   val_results["macro_f1"],
        })

        # Save best model
        if val_results["macro_auroc"] > best_auroc:
            best_auroc  = val_results["macro_auroc"]
            best_epoch  = epoch
            model.save_pretrained(CHECKPOINTS)
            tokenizer.save_pretrained(CHECKPOINTS)
            log.info(f"  ✅ New best model saved "
                     f"(AUROC={best_auroc:.4f})")

    log.info(f"\nTraining complete. Best epoch: {best_epoch} "
             f"(Val AUROC={best_auroc:.4f})")

    # ── Final evaluation on test set ──────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("Final Evaluation — Test Set")
    log.info("=" * 55)

    # Reload best model
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINTS,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)

    test_results = evaluate(model, test_loader, loss_fn, device)

    log.info(f"\n  {'Label':<25} {'AUROC':>7} {'F1':>7} {'Brier':>7}")
    log.info("  " + "-" * 48)
    for col in LABEL_COLS:
        r = test_results["per_label"][col]
        log.info(f"  {col:<25} {r['auroc']:>7.4f} "
                 f"{r['f1']:>7.4f} {r['brier']:>7.4f}")

    macro_auroc = test_results["macro_auroc"]
    macro_f1    = test_results["macro_f1"]
    log.info(f"  {'MACRO AVERAGE':<25} {macro_auroc:>7.4f} "
             f"{macro_f1:>7.4f}")

    # ── Save results ──────────────────────────────────────────────────
    rows = []
    for col in LABEL_COLS:
        r = test_results["per_label"][col]
        rows.append({
            "model": "Bio_ClinicalBERT",
            "label": col,
            "auroc": round(r["auroc"], 4),
            "f1":    round(r["f1"], 4),
            "brier": round(r["brier"], 4),
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(REPORTS / "slm_evaluation.csv", index=False)

    # ── Plots ─────────────────────────────────────────────────────────
    plot_training_curves(history)
    plot_auroc_comparison(
        test_results, REPORTS / "baseline_results.csv")

    log.info(f"\nSaved -> outputs/reports/slm_evaluation.csv")
    log.info(f"Saved -> outputs/checkpoints/bio_clinicalbert_goemed/")
    log.info("\nDone.")


if __name__ == "__main__":
    main()