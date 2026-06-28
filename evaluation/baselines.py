#!/usr/bin/env python3
"""Baseline comparison script for the QRIntel evaluation suite.

Implements five baseline classifiers on the same feature matrix and
train/test split used by the QRIntel optimised model, enabling a
fair apples-to-apples comparison:

1. **Naive Lexical** – predict malicious iff ``s_l >= 0.5``.
2. **Equal Weights**  – each module gets ``w = 0.25``; threshold from
   paper (30.06).
3. **Logistic Regression** – scikit-learn ``LogisticRegression`` on the
   four feature columns.
4. **Random Forest** – 100 trees, scikit-learn ``RandomForestClassifier``.
5. **Gradient Boosting** – 100 trees, scikit-learn
   ``GradientBoostingClassifier``.

All ML models are trained on the training split and evaluated on the
held-out test split.  The same six metrics as ``evaluate.py`` are
reported.

Typical usage
-------------
::

    python -m evaluation.baselines \\
        --features evaluation/features.csv \\
        --split    data/train_test_split_indices.json

Outputs
-------
* ``evaluation/baseline_results.json``
* Formatted console table.
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
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
# Paper defaults
# ---------------------------------------------------------------------------
PAPER_THRESHOLD: float = 30.06
PAPER_WEIGHTS: dict[str, float] = {
    "w_l": 0.5353,
    "w_d": 0.1053,
    "w_r": 0.1749,
    "w_ssl": 0.1845,
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_CSV: str = "evaluation/features.csv"
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_SPLIT_JSON: str = "data/train_test_split_indices.json"
DEFAULT_CONFIG_JSON: str = "evaluation/config.json"
DEFAULT_OUTPUT_JSON: str = "evaluation/baseline_results.json"
SEED: int = 42

# Feature column order used by all ML baselines
FEATURE_COLS: list[str] = [
    "lexical_score", "dns_score", "redirect_score", "ssl_score",
]


# ==========================================================================#
# Data loading helpers                                                      #
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
    """Load features from CSV or SQLite.

    Raises
    ------
    FileNotFoundError
        If neither source exists.
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
    """Load config, falling back to paper defaults."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return {
            "w_l": float(cfg["w_l"]),
            "w_d": float(cfg["w_d"]),
            "w_r": float(cfg["w_r"]),
            "w_ssl": float(cfg["w_ssl"]),
            "threshold": float(cfg["threshold"]),
        }
    return {**PAPER_WEIGHTS, "threshold": PAPER_THRESHOLD}


# ==========================================================================#
# Evaluation helper                                                         #
# ==========================================================================#

def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute the standard six-metric suite.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted binary labels.
    y_score:
        Continuous scores / probabilities for ROC-AUC.

    Returns
    -------
    dict
    """
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.0

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
        "roc_auc": round(auc, 4),
    }


# ==========================================================================#
# Feature matrix helpers                                                    #
# ==========================================================================#

def _build_X(
    features: list[dict[str, Any]],
    indices: list[int],
) -> np.ndarray:
    """Build a ``(n, 4)`` feature matrix for the given indices."""
    return np.array([
        [features[i][c] for c in FEATURE_COLS]
        for i in indices
    ])


def _build_y(
    features: list[dict[str, Any]],
    indices: list[int],
) -> np.ndarray:
    """Build a label vector for the given indices."""
    return np.array([features[i]["label"] for i in indices])


# ==========================================================================#
# Baseline implementations                                                  #
# ==========================================================================#

def baseline_naive_lexical(
    features: list[dict[str, Any]],
    test_indices: list[int],
) -> dict[str, Any]:
    """Naive Lexical baseline: predict malicious iff ``s_l >= 0.5``.

    This baseline uses **only** the lexical score with a hard threshold
    of 0.5, ignoring all other modules.

    Parameters
    ----------
    features:
        Full feature matrix.
    test_indices:
        Test partition indices.

    Returns
    -------
    dict
        Method name and metrics.
    """
    y_true = _build_y(features, test_indices)
    scores = np.array([features[i]["lexical_score"] for i in test_indices])
    y_pred = (scores >= 0.5).astype(int)

    metrics = _evaluate(y_true, y_pred, scores)
    logger.info("Naive Lexical      F1=%.4f  MCC=%.4f", metrics["f1"], metrics["mcc"])
    return {"method": "Naive Lexical (s_l >= 0.5)", "metrics": metrics}


def baseline_equal_weights(
    features: list[dict[str, Any]],
    test_indices: list[int],
    threshold: float = PAPER_THRESHOLD,
) -> dict[str, Any]:
    """Equal Weights baseline: ``w_i = 0.25`` for all modules.

    Uses the optimised threshold from the paper for fair comparison.

    Parameters
    ----------
    features:
        Full feature matrix.
    test_indices:
        Test partition indices.
    threshold:
        Decision threshold (default: paper value 30.06).

    Returns
    -------
    dict
        Method name and metrics.
    """
    y_true = _build_y(features, test_indices)
    X_test = _build_X(features, test_indices)
    scores = 100.0 * np.mean(X_test, axis=1)  # equal weights = mean × 100
    y_pred = (scores >= threshold).astype(int)

    metrics = _evaluate(y_true, y_pred, scores)
    logger.info("Equal Weights      F1=%.4f  MCC=%.4f", metrics["f1"], metrics["mcc"])
    return {"method": "Equal Weights (Optimized Threshold)", "metrics": metrics}


def baseline_logistic_regression(
    features: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
) -> dict[str, Any]:
    """Logistic Regression baseline on the four feature columns.

    Parameters
    ----------
    features:
        Full feature matrix.
    train_indices:
        Training partition indices.
    test_indices:
        Test partition indices.

    Returns
    -------
    dict
        Method name and metrics.
    """
    X_train = _build_X(features, train_indices)
    y_train = _build_y(features, train_indices)
    X_test = _build_X(features, test_indices)
    y_true = _build_y(features, test_indices)

    clf = LogisticRegression(random_state=SEED, max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_true, y_pred, y_score)
    logger.info("Logistic Regression F1=%.4f  MCC=%.4f", metrics["f1"], metrics["mcc"])
    return {"method": "Logistic Regression", "metrics": metrics}


def baseline_random_forest(
    features: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    n_estimators: int = 100,
) -> dict[str, Any]:
    """Random Forest baseline (100 trees).

    Parameters
    ----------
    features:
        Full feature matrix.
    train_indices:
        Training partition indices.
    test_indices:
        Test partition indices.
    n_estimators:
        Number of trees.

    Returns
    -------
    dict
        Method name and metrics.
    """
    X_train = _build_X(features, train_indices)
    y_train = _build_y(features, train_indices)
    X_test = _build_X(features, test_indices)
    y_true = _build_y(features, test_indices)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_true, y_pred, y_score)
    logger.info(
        "Random Forest      F1=%.4f  MCC=%.4f",
        metrics["f1"], metrics["mcc"],
    )
    return {
        "method": f"Random Forest ({n_estimators} trees)",
        "metrics": metrics,
    }


def baseline_gradient_boosting(
    features: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    n_estimators: int = 100,
) -> dict[str, Any]:
    """Gradient Boosting baseline (100 trees).

    Parameters
    ----------
    features:
        Full feature matrix.
    train_indices:
        Training partition indices.
    test_indices:
        Test partition indices.
    n_estimators:
        Number of boosting rounds.

    Returns
    -------
    dict
        Method name and metrics.
    """
    X_train = _build_X(features, train_indices)
    y_train = _build_y(features, train_indices)
    X_test = _build_X(features, test_indices)
    y_true = _build_y(features, test_indices)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=SEED
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    metrics = _evaluate(y_true, y_pred, y_score)
    logger.info(
        "Gradient Boosting  F1=%.4f  MCC=%.4f",
        metrics["f1"], metrics["mcc"],
    )
    return {
        "method": f"Gradient Boosting ({n_estimators} trees)",
        "metrics": metrics,
    }


# ==========================================================================#
# QRIntel optimised (ours) — for inclusion in the comparison table          #
# ==========================================================================#

def qrintel_optimized(
    features: list[dict[str, Any]],
    test_indices: list[int],
    config: dict[str, float],
) -> dict[str, Any]:
    """QRIntel optimised model (ours) on the test set.

    Included so the baseline table mirrors Table I in the manuscript.

    Parameters
    ----------
    features:
        Full feature matrix.
    test_indices:
        Test partition indices.
    config:
        Optimised weights and threshold.

    Returns
    -------
    dict
        Method name and metrics.
    """
    y_true = _build_y(features, test_indices)
    X_test = _build_X(features, test_indices)
    scores = 100.0 * (
        config["w_l"] * X_test[:, 0]
        + config["w_d"] * X_test[:, 1]
        + config["w_r"] * X_test[:, 2]
        + config["w_ssl"] * X_test[:, 3]
    )
    y_pred = (scores >= config["threshold"]).astype(int)

    metrics = _evaluate(y_true, y_pred, scores)
    logger.info(
        "QRIntel Optimized  F1=%.4f  MCC=%.4f", metrics["f1"], metrics["mcc"]
    )
    return {"method": "QRIntel Optimized (Ours)", "metrics": metrics}


# ==========================================================================#
# Results formatting                                                        #
# ==========================================================================#

def _print_baseline_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc"]
    headers = ["Accuracy", "Precision", "Recall", "F1", "MCC", "ROC-AUC"]

    col_w = 10
    name_w = 38
    sep = (
        "+" + "-" * name_w
        + "+" + (("-" * col_w + "+") * len(headers))
    )
    header_row = (
        "| {:^{nw}s}|".format("Method", nw=name_w - 1)
        + "|".join(f"{h:^{col_w}s}" for h in headers)
        + "|"
    )

    print("\n" + "=" * len(sep))
    print("  TABLE I: Baseline Comparison on Held-Out Test Set")
    print("=" * len(sep))
    print(sep)
    print(header_row)
    print(sep)

    for entry in results:
        name = entry["method"]
        m = entry["metrics"]
        row = "| {:^{nw}s}|".format(name, nw=name_w - 1)
        row += "|".join(f"{m[k]:^{col_w}.4f}" for k in metric_keys)
        row += "|"
        print(row)

    print(sep + "\n")


# ==========================================================================#
# Main driver                                                               #
# ==========================================================================#

def run_baselines(
    features: list[dict[str, Any]],
    train_indices: list[int],
    test_indices: list[int],
    *,
    config: dict[str, float],
    output_path: str = DEFAULT_OUTPUT_JSON,
) -> list[dict[str, Any]]:
    """Execute all baselines and the QRIntel model, then save results.

    Parameters
    ----------
    features:
        Complete feature matrix.
    train_indices:
        Training partition indices.
    test_indices:
        Test partition indices.
    config:
        QRIntel optimised weights and threshold.
    output_path:
        Destination JSON.

    Returns
    -------
    list[dict]
        One entry per method with ``method`` and ``metrics`` keys.
    """
    logger.info(
        "Running baselines  (train=%d, test=%d)",
        len(train_indices), len(test_indices),
    )

    results: list[dict[str, Any]] = [
        baseline_naive_lexical(features, test_indices),
        baseline_equal_weights(features, test_indices, config["threshold"]),
        baseline_logistic_regression(features, train_indices, test_indices),
        baseline_random_forest(features, train_indices, test_indices),
        baseline_gradient_boosting(features, train_indices, test_indices),
        qrintel_optimized(features, test_indices, config),
    ]

    _print_baseline_table(results)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Baseline results saved to %s", output_path)

    return results


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="QRIntel baseline comparison on the held-out test set.",
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
        "--output", "-o",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the baseline comparison script."""
    args = _build_parser().parse_args(argv)

    features = load_features(csv_path=args.features, db_path=args.db)
    splits = load_split_indices(args.split)
    config = load_config(args.config)

    run_baselines(
        features=features,
        train_indices=splits["train_indices"],
        test_indices=splits["test_indices"],
        config=config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
