#!/usr/bin/env python3
"""Evaluation script for the QRIntel empirical risk model.

Performs:
1. **5-fold stratified cross-validation** on the training partition.
2. **Held-out test-set evaluation** using the paper's optimised weights.

Metrics reported for every fold and on the test set:
    Accuracy, Precision, Recall, F1-score, MCC, ROC-AUC.

Typical usage
-------------
::

    python -m evaluation.evaluate \\
        --features evaluation/features.csv \\
        --split    data/train_test_split_indices.json

Outputs
-------
* ``evaluation/cv_results.json``   – per-fold and aggregated CV metrics,
  plus held-out test metrics.
* Formatted console table.

Paper reference values
~~~~~~~~~~~~~~~~~~~~~~
* w_l = 0.5353, w_d = 0.1053, w_r = 0.1749, w_ssl = 0.1845
* threshold = 30.06
* Train CV F1 = 0.9935 ± 0.0042, Test F1 = 0.9946, MCC = 0.9712
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-reported optimal configuration
# ---------------------------------------------------------------------------
PAPER_WEIGHTS: dict[str, float] = {
    "w_l": 0.5353,
    "w_d": 0.1053,
    "w_r": 0.1749,
    "w_ssl": 0.1845,
}
PAPER_THRESHOLD: float = 30.06

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_CSV: str = "evaluation/features.csv"
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_SPLIT_JSON: str = "data/train_test_split_indices.json"
DEFAULT_CONFIG_JSON: str = "evaluation/config.json"
DEFAULT_OUTPUT_JSON: str = "evaluation/cv_results.json"
DEFAULT_N_FOLDS: int = 5


# ==========================================================================#
# Data loading                                                              #
# ==========================================================================#

def _load_features_csv(path: str) -> list[dict[str, Any]]:
    """Load feature rows from CSV."""
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "url": row["url"],
                "label": int(row["label"]),
                "lexical_score": float(row["lexical_score"]),
                "dns_score": float(row["dns_score"]),
                "redirect_score": float(row["redirect_score"]),
                "ssl_score": float(row["ssl_score"]),
            })
    return rows


def _load_features_sqlite(path: str) -> list[dict[str, Any]]:
    """Load feature rows from SQLite."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT url, label, lexical_score, dns_score, "
        "redirect_score, ssl_score FROM features ORDER BY url"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_features(
    csv_path: str | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load features from CSV or SQLite (CSV preferred).

    Raises
    ------
    FileNotFoundError
        If neither source file exists.
    """
    if csv_path and Path(csv_path).exists():
        logger.info("Loading features from %s", csv_path)
        return _load_features_csv(csv_path)
    if db_path and Path(db_path).exists():
        logger.info("Loading features from %s", db_path)
        return _load_features_sqlite(db_path)
    raise FileNotFoundError(
        f"Feature source not found (CSV={csv_path}, DB={db_path})."
    )


def load_split_indices(path: str) -> dict[str, list[int]]:
    """Load train/test index arrays from JSON."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        "train_indices": data["train_indices"],
        "test_indices": data["test_indices"],
    }


def load_config(path: str) -> dict[str, float]:
    """Load weights and threshold from a config JSON.

    Falls back to the paper-reported values if the file does not exist.

    Parameters
    ----------
    path:
        Path to ``config.json``.

    Returns
    -------
    dict
        Keys: ``w_l``, ``w_d``, ``w_r``, ``w_ssl``, ``threshold``.
    """
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        logger.info("Loaded config from %s", path)
        return {
            "w_l": float(cfg["w_l"]),
            "w_d": float(cfg["w_d"]),
            "w_r": float(cfg["w_r"]),
            "w_ssl": float(cfg["w_ssl"]),
            "threshold": float(cfg["threshold"]),
        }
    logger.info("Config not found at %s – using paper defaults.", path)
    return {**PAPER_WEIGHTS, "threshold": PAPER_THRESHOLD}


# ==========================================================================#
# Scoring helpers                                                           #
# ==========================================================================#

def compute_scores(
    features: list[dict[str, Any]],
    indices: list[int] | np.ndarray,
    *,
    w_l: float,
    w_d: float,
    w_r: float,
    w_ssl: float,
) -> np.ndarray:
    """Compute the empirical risk score (Eq. 1) for a subset of rows.

    Parameters
    ----------
    features:
        Full feature matrix.
    indices:
        Row indices to score.
    w_l, w_d, w_r, w_ssl:
        Module weights.

    Returns
    -------
    numpy.ndarray
        Risk scores in ``[0, 100]``.
    """
    subset = [features[i] for i in indices]
    s_l = np.array([r["lexical_score"] for r in subset])
    s_d = np.array([r["dns_score"] for r in subset])
    s_r = np.array([r["redirect_score"] for r in subset])
    s_ssl = np.array([r["ssl_score"] for r in subset])
    return 100.0 * (w_l * s_l + w_d * s_d + w_r * s_r + w_ssl * s_ssl)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute the full evaluation metric suite.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    scores:
        Continuous risk scores (for ROC-AUC).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``mcc``,
        ``roc_auc``.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0.0)), 4
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, zero_division=0.0)), 4
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, zero_division=0.0)), 4
        ),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "roc_auc": round(
            float(roc_auc_score(y_true, scores)), 4
        ),
    }


# ==========================================================================#
# Cross-validation                                                          #
# ==========================================================================#

def run_cross_validation(
    features: list[dict[str, Any]],
    train_indices: list[int],
    *,
    config: dict[str, float],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run stratified K-fold cross-validation on the training partition.

    Parameters
    ----------
    features:
        Full feature matrix.
    train_indices:
        Indices into *features* for the training partition.
    config:
        Dict with keys ``w_l``, ``w_d``, ``w_r``, ``w_ssl``, ``threshold``.
    n_folds:
        Number of CV folds.
    seed:
        Random state for reproducibility.

    Returns
    -------
    list[dict]
        One metric dict per fold.
    """
    w_l = config["w_l"]
    w_d = config["w_d"]
    w_r = config["w_r"]
    w_ssl = config["w_ssl"]
    threshold = config["threshold"]

    train_data = [features[i] for i in train_indices]
    y_all = np.array([r["label"] for r in train_data])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results: list[dict[str, float]] = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_data, y_all), 1):
        # Map local val_idx back to global indices
        global_val = [train_indices[i] for i in val_idx]
        scores = compute_scores(
            features, global_val, w_l=w_l, w_d=w_d, w_r=w_r, w_ssl=w_ssl
        )
        y_true = np.array([features[i]["label"] for i in global_val])
        y_pred = (scores >= threshold).astype(int)

        metrics = evaluate_predictions(y_true, y_pred, scores)
        fold_results.append(metrics)

        logger.info(
            "Fold %d/%d  F1=%.4f  MCC=%.4f  AUC=%.4f",
            fold_idx, n_folds, metrics["f1"], metrics["mcc"], metrics["roc_auc"],
        )

    return fold_results


# ==========================================================================#
# Held-out test evaluation                                                  #
# ==========================================================================#

def run_test_evaluation(
    features: list[dict[str, Any]],
    test_indices: list[int],
    *,
    config: dict[str, float],
) -> dict[str, float]:
    """Evaluate on the held-out test partition.

    Parameters
    ----------
    features:
        Full feature matrix.
    test_indices:
        Indices into *features* for the test partition.
    config:
        Dict with keys ``w_l``, ``w_d``, ``w_r``, ``w_ssl``, ``threshold``.

    Returns
    -------
    dict
        Full metric suite.
    """
    scores = compute_scores(
        features,
        test_indices,
        w_l=config["w_l"],
        w_d=config["w_d"],
        w_r=config["w_r"],
        w_ssl=config["w_ssl"],
    )
    y_true = np.array([features[i]["label"] for i in test_indices])
    y_pred = (scores >= config["threshold"]).astype(int)

    return evaluate_predictions(y_true, y_pred, scores)


# ==========================================================================#
# Results formatting                                                        #
# ==========================================================================#

def _print_results_table(
    cv_folds: list[dict[str, float]],
    cv_summary: dict[str, Any],
    test_metrics: dict[str, float],
) -> None:
    """Print a nicely formatted console table of results."""
    metrics_keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
    header_labels = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"]

    sep = "+" + "+".join(["-" * 16] + ["-" * 12] * len(header_labels)) + "+"
    header = (
        "| {:^14s} |".format("Partition")
        + "|".join(f" {h:^10s} " for h in header_labels)
        + "|"
    )

    print("\n" + "=" * len(sep))
    print("  QRIntel Evaluation Results")
    print("=" * len(sep))
    print(sep)
    print(header)
    print(sep)

    for i, fold in enumerate(cv_folds, 1):
        row = "| {:^14s} |".format(f"CV Fold {i}")
        row += "|".join(f" {fold[k]:^10.4f} " for k in metrics_keys)
        row += "|"
        print(row)

    print(sep)

    # CV mean ± std
    cv_row = "| {:^14s} |".format("CV Mean±Std")
    for k in metrics_keys:
        mean = cv_summary[f"{k}_mean"]
        std = cv_summary[f"{k}_std"]
        cv_row += f" {mean:.4f}±{std:.4f}".ljust(12) + "|"
    print(cv_row)

    print(sep)

    # Test set
    test_row = "| {:^14s} |".format("Test Set")
    test_row += "|".join(f" {test_metrics[k]:^10.4f} " for k in metrics_keys)
    test_row += "|"
    print(test_row)

    print(sep + "\n")


# ==========================================================================#
# Main driver                                                               #
# ==========================================================================#

def run_evaluation(
    features: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    *,
    config: dict[str, float],
    n_folds: int = DEFAULT_N_FOLDS,
    output_path: str = DEFAULT_OUTPUT_JSON,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full evaluation pipeline and save results.

    Parameters
    ----------
    features:
        Complete feature matrix.
    train_indices:
        Training partition indices.
    test_indices:
        Test partition indices.
    config:
        Weights and threshold configuration.
    n_folds:
        Number of CV folds.
    output_path:
        Destination JSON for results.
    seed:
        Random state.

    Returns
    -------
    dict
        Complete results including ``cv_folds``, ``cv_summary``,
        and ``test_metrics``.
    """
    logger.info(
        "Config: w_l=%.4f w_d=%.4f w_r=%.4f w_ssl=%.4f threshold=%.2f",
        config["w_l"], config["w_d"], config["w_r"],
        config["w_ssl"], config["threshold"],
    )

    # --- Cross-validation -----------------------------------------------
    logger.info("Running %d-fold stratified CV on training set (%d URLs)...",
                n_folds, len(train_indices))
    cv_folds = run_cross_validation(
        features, train_indices, config=config, n_folds=n_folds, seed=seed,
    )

    # Aggregate CV metrics
    metrics_keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
    cv_summary: dict[str, Any] = {}
    for k in metrics_keys:
        values = [fold[k] for fold in cv_folds]
        cv_summary[f"{k}_mean"] = round(float(np.mean(values)), 4)
        cv_summary[f"{k}_std"] = round(float(np.std(values)), 4)

    logger.info(
        "CV Summary  F1=%.4f±%.4f  MCC=%.4f±%.4f",
        cv_summary["f1_mean"], cv_summary["f1_std"],
        cv_summary["mcc_mean"], cv_summary["mcc_std"],
    )

    # --- Held-out test --------------------------------------------------
    logger.info("Evaluating on held-out test set (%d URLs)...", len(test_indices))
    test_metrics = run_test_evaluation(
        features, test_indices, config=config
    )
    logger.info(
        "Test  F1=%.4f  MCC=%.4f  AUC=%.4f",
        test_metrics["f1"], test_metrics["mcc"], test_metrics["roc_auc"],
    )

    # --- Print table ----------------------------------------------------
    _print_results_table(cv_folds, cv_summary, test_metrics)

    # --- Persist --------------------------------------------------------
    results: dict[str, Any] = {
        "config": config,
        "cv_folds": cv_folds,
        "cv_summary": cv_summary,
        "test_metrics": test_metrics,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to %s", output_path)

    return results


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate the QRIntel risk model (CV + held-out test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features", "-f",
        default=DEFAULT_FEATURES_CSV,
        help="Path to features CSV.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to SQLite feature cache (fallback).",
    )
    parser.add_argument(
        "--split", "-s",
        default=DEFAULT_SPLIT_JSON,
        help="Path to train_test_split_indices.json.",
    )
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_JSON,
        help="Path to config.json (weights + threshold).",
    )
    parser.add_argument(
        "--folds", "-k",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the evaluation script."""
    args = _build_parser().parse_args(argv)

    features = load_features(csv_path=args.features, db_path=args.db)
    splits = load_split_indices(args.split)
    config = load_config(args.config)

    run_evaluation(
        features=features,
        train_indices=splits["train_indices"],
        test_indices=splits["test_indices"],
        config=config,
        n_folds=args.folds,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
