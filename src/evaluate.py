import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from .data import FIGURES_DIR, LABELS, MAJOR_LABELS, RESULTS_DIR, SEED

EXPERIMENT_LOG = RESULTS_DIR / "experiments.csv"


def align_proba(proba, classes):
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)
    aligned = np.column_stack([proba[:, classes.index(label)] for label in LABELS])
    return aligned / aligned.sum(axis=1, keepdims=True)


def labels_from_proba(proba):
    return np.array(LABELS)[np.argmax(proba, axis=1)]


def macro_f1(y_true, y_pred, labels=LABELS):
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def bootstrap_macro_f1(y_true, y_pred, n_boot=1000, seed=SEED):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(macro_f1(y_true[idx], y_pred[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def multiclass_log_loss(y_true, proba, eps=1e-15):
    idx = np.array([LABELS.index(y) for y in y_true])
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(idx)), idx], eps, 1))))


def compute_metrics(y_true, proba):
    y_pred = labels_from_proba(proba)
    low, high = bootstrap_macro_f1(y_true, y_pred)
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    metrics = {
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci_low": low,
        "macro_f1_ci_high": high,
        "macro_f1_major": macro_f1(y_true, y_pred, MAJOR_LABELS),
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": multiclass_log_loss(y_true, proba),
    }
    metrics.update({f"f1_{label}": float(score) for label, score in zip(LABELS, per_class)})
    return {k: float(v) for k, v in metrics.items()}, y_pred


def evaluate(name, val_df, proba, save=True):
    y_true = val_df["category"].to_numpy()
    metrics, y_pred = compute_metrics(y_true, proba)
    print(f"{name}: macro-F1 {metrics['macro_f1']:.3f} "
          f"(95% CI {metrics['macro_f1_ci_low']:.3f}-{metrics['macro_f1_ci_high']:.3f}), "
          f"accuracy {metrics['accuracy']:.3f}, log loss {metrics['log_loss']:.3f}")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    if save:
        (RESULTS_DIR / f"{name}_metrics.json").write_text(json.dumps({"model": name, **metrics}, indent=2))
        preds = pd.DataFrame({"id": val_df["id"].to_numpy(), "category": y_true, "predicted": y_pred})
        for i, label in enumerate(LABELS):
            preds[f"p_{label}"] = proba[:, i]
        preds.to_csv(RESULTS_DIR / f"{name}_val_predictions.csv", index=False)
    return metrics, y_pred


def plot_confusion(y_true, y_pred, title, ax=None, save_as=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    if save_as:
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / save_as, dpi=150)
    return ax


def plot_history(history, title, save_as=None):
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(hist["epoch"], hist["train_loss"], marker="o", label="train")
    axes[0].plot(hist["epoch"], hist["dev_loss"], marker="o", label="dev")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["dev_macro_f1"], marker="o", color="#C44E52")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dev macro-F1")
    fig.suptitle(title)
    plt.tight_layout()
    if save_as:
        plt.savefig(FIGURES_DIR / save_as, dpi=150)
    plt.show()


def log_experiment(model, experiment, change, selection_score, val_metrics, train_time_s=None):
    row = {
        "model": model,
        "experiment": experiment,
        "change": change,
        "selection_macro_f1": round(float(selection_score), 4),
        "val_macro_f1": round(val_metrics["macro_f1"], 4),
        "val_macro_f1_major": round(val_metrics["macro_f1_major"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_log_loss": round(val_metrics["log_loss"], 4),
        "train_time_s": None if train_time_s is None else round(float(train_time_s), 1),
    }
    new = pd.DataFrame([row])
    if EXPERIMENT_LOG.exists():
        log = pd.read_csv(EXPERIMENT_LOG)
        log = log[~((log["model"] == model) & (log["experiment"] == experiment))]
        new = pd.concat([log, new], ignore_index=True) if len(log) else new
    new.to_csv(EXPERIMENT_LOG, index=False)
    return pd.DataFrame([row])


def load_experiments(model=None):
    log = pd.read_csv(EXPERIMENT_LOG)
    return log if model is None else log[log["model"] == model].reset_index(drop=True)


def load_final_results(names):
    metrics = pd.DataFrame([json.loads((RESULTS_DIR / f"{n}_metrics.json").read_text()) for n in names])
    preds = {n: pd.read_csv(RESULTS_DIR / f"{n}_val_predictions.csv") for n in names}
    return metrics.set_index("model"), preds
