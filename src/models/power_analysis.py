"""
power_analysis.py
-----------------
Power Analysis — Learning Curves for Bio_ClinicalBERT and XGBoost.

Answers the proposal's key research question:
"What is the minimum dataset size required for an SLM to achieve
clinically meaningful predictive performance?"

Method (Stat 309/310):
    - Train models on increasing dataset sizes: 1k, 2k, 5k, 10k, 20k
    - Record macro AUROC at each size
    - Fit a learning curve model: AUROC(n) = a - b * n^(-c)
    - Find inflection point where marginal gain < 0.005 AUROC
    - That inflection point = minimum clinically meaningful dataset size

Output:
    - outputs/figures/power_analysis_curves.png
    - outputs/figures/power_analysis_fitted.png
    - outputs/reports/power_analysis_results.csv
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).resolve().parents[2]
SPLITS_DIR  = REPO_ROOT / "data/splits"
CHECKPOINTS = REPO_ROOT / "outputs/checkpoints/bio_clinicalbert_goemed"
PROCESSED   = REPO_ROOT / "data/processed"
FIGURES     = REPO_ROOT / "outputs/figures"
REPORTS     = REPO_ROOT / "outputs/reports"
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_COLS = [
    "hypertension", "diabetes", "cvd_risk", "ckd",
    "osa", "depression", "copd", "metabolic_syndrome",
    "hypothyroidism", "prediabetes", "colorectal_cancer",
]

# Dataset sizes for learning curve
TRAIN_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000]

FEATURE_COLS = [
    "age", "bmi", "sbp", "dbp",
    "total_cholesterol", "hdl_cholesterol",
    "hba1c", "fasting_glucose", "waist_cm", "weight_kg",
    "creatinine", "pack_years",
    "smoker", "physical_activity_low", "bp_treated",
    "family_history_dm", "family_history_cvd",
    "snoring", "tired", "alcohol_use", "sex_male",
]


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class PatientDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer,
                 max_length: int = 128, max_records: int = None):
        self.records = []
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                self.records.append(json.loads(line))
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec      = self.records[idx]
        encoding = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_vector = torch.tensor(
            [float(rec["labels"].get(col, 0)) for col in LABEL_COLS],
            dtype=torch.float32,
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         label_vector,
        }


# ══════════════════════════════════════════════════════════════════════════
# XGBOOST LEARNING CURVE
# ══════════════════════════════════════════════════════════════════════════

def xgboost_learning_curve(train_sizes: List[int]) -> List[Dict]:
    """Train XGBoost at each dataset size and record macro AUROC."""
    log.info("\n── XGBoost Learning Curve ──")

    df = pd.read_parquet(PROCESSED / "synthetic_patients.parquet")
    df["sex_male"] = (df["sex"] == "M").astype(int)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in feat_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_test  = 10_000
    test_df = df.iloc[:n_test]
    train_pool = df.iloc[n_test:]

    X_test  = test_df[feat_cols].fillna(test_df[feat_cols].median())
    Y_test  = test_df[LABEL_COLS].astype(int)

    results = []
    for n in train_sizes:
        train_df = train_pool.iloc[:n]
        X_train  = train_df[feat_cols].fillna(train_df[feat_cols].median())
        Y_train  = train_df[LABEL_COLS].astype(int)

        aurocs = []
        for label in LABEL_COLS:
            pos_weight = (Y_train[label] == 0).sum() / \
                         max((Y_train[label] == 1).sum(), 1)
            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=pos_weight,
                eval_metric="auc",
                random_state=SEED,
                verbosity=0,
            )
            clf.fit(X_train, Y_train[label])
            probs = clf.predict_proba(X_test)[:, 1]
            try:
                aurocs.append(roc_auc_score(Y_test[label], probs))
            except Exception:
                pass

        macro_auroc = np.mean(aurocs)
        results.append({
            "model":       "XGBoost",
            "train_size":  n,
            "macro_auroc": round(macro_auroc, 4),
        })
        log.info(f"  n={n:>7,}  macro AUROC={macro_auroc:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# SLM LEARNING CURVE
# ══════════════════════════════════════════════════════════════════════════

def slm_learning_curve(train_sizes: List[int]) -> List[Dict]:
    """Fine-tune Bio_ClinicalBERT at each dataset size."""
    log.info("\n── Bio_ClinicalBERT Learning Curve ──")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS)

    # Fixed val and test sets
    val_ds  = PatientDataset(
        SPLITS_DIR / "val.jsonl", tokenizer, max_records=2_000)
    test_ds = PatientDataset(
        SPLITS_DIR / "test.jsonl", tokenizer, max_records=2_000)
    val_loader  = DataLoader(val_ds,  batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    results = []

    for n in train_sizes:
        log.info(f"\n  Training SLM with n={n:,} records...")

        # Fresh model for each size
        model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINTS,
            num_labels=len(LABEL_COLS),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        ).to(device)

        train_ds = PatientDataset(
            SPLITS_DIR / "train.jsonl", tokenizer,
            max_records=n)
        train_loader = DataLoader(
            train_ds, batch_size=8, shuffle=True)

        # Class weights
        label_sums = np.zeros(len(LABEL_COLS))
        for rec in train_ds.records:
            for i, col in enumerate(LABEL_COLS):
                label_sums[i] += float(rec["labels"].get(col, 0))
        prevalences  = label_sums / len(train_ds)
        pos_weights  = np.clip(
            (1 - prevalences) / np.maximum(prevalences, 1e-6),
            1.0, 20.0)
        class_weights = torch.tensor(
            pos_weights, dtype=torch.float32).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

        # Short training — 2 epochs for learning curve
        epochs        = 2
        total_steps   = len(train_loader) * epochs
        optimizer     = torch.optim.AdamW(
            model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler     = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        # Evaluate on test set
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs        = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = torch.sigmoid(torch.tensor(logits)).numpy()

        aurocs = []
        for i in range(len(LABEL_COLS)):
            try:
                aurocs.append(
                    roc_auc_score(labels[:, i], probs[:, i]))
            except Exception:
                pass

        macro_auroc = np.mean(aurocs)
        results.append({
            "model":       "Bio_ClinicalBERT",
            "train_size":  n,
            "macro_auroc": round(macro_auroc, 4),
        })
        log.info(f"  n={n:>7,}  macro AUROC={macro_auroc:.4f}")

        # Free memory
        del model
        torch.mps.empty_cache() if hasattr(torch, "mps") else None

    return results


# ══════════════════════════════════════════════════════════════════════════
# LEARNING CURVE FITTING (Stat 309/310)
# ══════════════════════════════════════════════════════════════════════════

def fit_learning_curve(
    sizes: np.ndarray,
    aurocs: np.ndarray,
    model_name: str,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit power-law learning curve: AUROC(n) = a - b * n^(-c)

    This is the standard learning curve model from statistical
    learning theory. Parameters:
        a = asymptotic performance (n → ∞)
        b = initial performance gap
        c = learning rate exponent

    From Stat 309/310: use nonlinear least squares (scipy curve_fit)
    which minimizes sum of squared residuals.
    """
    def power_law(n, a, b, c):
        return a - b * np.power(n, -c)

    try:
        popt, pcov = curve_fit(
            power_law, sizes, aurocs,
            p0=[0.99, 0.5, 0.5],
            bounds=([0.5, 0, 0.01], [1.0, 2.0, 2.0]),
            maxfev=5000,
        )
        a, b, c = popt
        perr    = np.sqrt(np.diag(pcov))

        log.info(f"\n  {model_name} fitted curve: "
                 f"AUROC(n) = {a:.4f} - {b:.4f} * n^(-{c:.4f})")
        log.info(f"  Asymptotic AUROC (n→∞): {a:.4f}")
        log.info(f"  Parameter std errors: "
                 f"a±{perr[0]:.4f}, b±{perr[1]:.4f}, c±{perr[2]:.4f}")

        # Find minimum n where marginal gain < 0.005
        n_range     = np.arange(500, 100_001, 100)
        auroc_curve = power_law(n_range, *popt)
        marginal    = np.diff(auroc_curve)
        try:
            min_n_idx = np.where(marginal < 0.005)[0][0]
            min_n     = int(n_range[min_n_idx])
            log.info(f"  Minimum dataset size "
                     f"(marginal gain < 0.005): n={min_n:,}")
        except IndexError:
            min_n = int(sizes[-1])

        return popt, pcov, min_n

    except Exception as e:
        log.warning(f"  Curve fitting failed: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_learning_curves(results_df: pd.DataFrame):
    """Plot empirical learning curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "XGBoost":          "#378ADD",
        "Bio_ClinicalBERT": "#1D9E75",
    }

    for ax_idx, (ax, title) in enumerate(zip(
            axes, ["Learning Curves", "Performance Gap"])):

        if ax_idx == 0:
            for model, grp in results_df.groupby("model"):
                ax.plot(
                    grp["train_size"], grp["macro_auroc"],
                    "o-", color=colors[model],
                    label=model, linewidth=2, markersize=6,
                )
            ax.set_xlabel("Training Set Size", fontsize=11)
            ax.set_ylabel("Macro AUROC", fontsize=11)
            ax.set_title("Learning Curves — Macro AUROC vs Dataset Size",
                         fontweight="bold")
            ax.set_xscale("log")
            ax.set_ylim(0.5, 1.05)
            ax.axhline(0.80, color="gray", linestyle="--",
                       linewidth=1, label="0.80 threshold")
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Performance gap between SLM and XGBoost
            xgb_df  = results_df[
                results_df["model"] == "XGBoost"].set_index(
                "train_size")["macro_auroc"]
            slm_df  = results_df[
                results_df["model"] == "Bio_ClinicalBERT"].set_index(
                "train_size")["macro_auroc"]
            common  = xgb_df.index.intersection(slm_df.index)
            gap     = slm_df[common] - xgb_df[common]

            ax.bar(range(len(common)),
                   gap.values, color="#D85A30", alpha=0.8)
            ax.set_xticks(range(len(common)))
            ax.set_xticklabels(
                [f"{n:,}" for n in common], rotation=30)
            ax.set_xlabel("Training Set Size", fontsize=11)
            ax.set_ylabel("AUROC Gap (SLM − XGBoost)", fontsize=11)
            ax.set_title("Bio_ClinicalBERT Advantage over XGBoost",
                         fontweight="bold")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Power Analysis — Minimum Dataset Size for Clinical SLM",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "power_analysis_curves.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/power_analysis_curves.png")


def plot_fitted_curves(
    results_df: pd.DataFrame,
    fits: Dict,
):
    """Plot fitted power-law curves with asymptote and minimum n."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = {
        "XGBoost":          "#378ADD",
        "Bio_ClinicalBERT": "#1D9E75",
    }
    n_smooth = np.linspace(500, 25_000, 300)

    def power_law(n, a, b, c):
        return a - b * np.power(n, -c)

    for model, (popt, pcov, min_n) in fits.items():
        grp = results_df[results_df["model"] == model]
        color = colors[model]

        # Empirical points
        ax.scatter(grp["train_size"], grp["macro_auroc"],
                   color=color, s=80, zorder=5)

        if popt is not None:
            # Fitted curve
            fitted = power_law(n_smooth, *popt)
            ax.plot(n_smooth, fitted, color=color,
                    linewidth=2, label=f"{model} (fitted)",
                    linestyle="--")

            # Asymptote
            ax.axhline(popt[0], color=color, linewidth=0.8,
                       linestyle=":", alpha=0.6,
                       label=f"{model} asymptote={popt[0]:.3f}")

            # Minimum n marker
            if min_n and min_n <= 25_000:
                ax.axvline(min_n, color=color, linewidth=1.0,
                           linestyle="-.", alpha=0.7)
                ax.text(min_n, 0.52,
                        f"min n={min_n:,}",
                        color=color, fontsize=8,
                        rotation=90, va="bottom")

    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel("Macro AUROC", fontsize=11)
    ax.set_title(
        "Fitted Learning Curves — Power Law Model\n"
        "AUROC(n) = a − b·n^(−c)",
        fontweight="bold", fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "power_analysis_fitted.png",
                dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved -> outputs/figures/power_analysis_fitted.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 55)
    log.info("Power Analysis — Learning Curves")
    log.info("(Stat 309/310: nonlinear least squares curve fitting)")
    log.info("=" * 55)

    # XGBoost learning curve — fast
    xgb_results = xgboost_learning_curve(TRAIN_SIZES)

    # SLM learning curve — slower
    slm_results = slm_learning_curve(TRAIN_SIZES)

    # Combine
    all_results = xgb_results + slm_results
    results_df  = pd.DataFrame(all_results)

    log.info("\n── Learning Curve Summary ──")
    log.info(f"\n{results_df.pivot(index='train_size', columns='model', values='macro_auroc').to_string()}")

    # Fit power-law curves
    log.info("\n── Power-Law Curve Fitting (Stat 309/310) ──")
    fits = {}
    for model in ["XGBoost", "Bio_ClinicalBERT"]:
        grp = results_df[results_df["model"] == model]
        if len(grp) >= 3:
            popt, pcov, min_n = fit_learning_curve(
                grp["train_size"].values.astype(float),
                grp["macro_auroc"].values,
                model,
            )
            fits[model] = (popt, pcov, min_n)

    # Plots
    plot_learning_curves(results_df)
    plot_fitted_curves(results_df, fits)

    # Save results
    results_df.to_csv(
        REPORTS / "power_analysis_results.csv", index=False)

    log.info(f"\nSaved -> outputs/reports/power_analysis_results.csv")
    log.info(f"Saved -> outputs/figures/power_analysis_curves.png")
    log.info(f"Saved -> outputs/figures/power_analysis_fitted.png")
    log.info("\nDone.")


if __name__ == "__main__":
    main()