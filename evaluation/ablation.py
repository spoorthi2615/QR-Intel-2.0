#!/usr/bin/env python3
"""Frozen-weight ablation study for the QRIntel empirical risk model.

For each of the four Stage-1 intelligence modules (lexical, DNS, redirect,
SSL), this script **zeros out** the module's weight while keeping all other
weights and the decision threshold **fixed** at their optimised values.
Crucially, the remaining weights are **not re-normalised** — this
isolates the contribution of each module without introducing confounding
weight redistribution effects.

Typical usage
-------------
::

    python -m evaluation.ablation \\
        --features evaluation/features.csv \\
        --split    data/train_test_split_indices.json

Paper reference values
~~~~~~~~~~~~~~~~~~~~~~
* w_l = 0.5353, w_d = 0.1053, w_r = 0.1749, w_ssl = 0.1845
* threshold = 30.06 (held constant across all ablations)

Expected results (from the manuscript):

======== ======== ==========
Removed   F1      ΔF1
======== ======== ==========
Lexical   0.0000  −0.9946
DNS       0.9919  −0.0027
Redirect  0.9946   0.0000
SSL       0.9946   0.0000
======== ======== ==========

Outputs
-------
* ``evaluation/ablation_results.json``
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

# Module definitions — maps a human-readable name to its weight key
MODULE_KEYS: dict[str, str] = {
    "lexical": "w_l",
    "dns": "w_d",
    "redirect": "w_r",
    "ssl": "w_ssl",
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_CSV: str = "evaluation/features.csv"
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_SPLIT_JSON: str = "data/train_test_split_indices.json"
DEFAULT_CONFIG_JSON: str = "evaluation/config.json"
DEFAULT_OUTPUT_JSON: str = "evaluation/ablation_results.json"


# ==========================================================================#
# Data loading helpers (shared pattern)                                     #
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
    logger.info("Config not found – using paper defaults.")
    return {**PAPER_WEIGHTS, "threshold": PAPER_THRESHOLD}


# ==========================================================================#
# Scoring and evaluation                                                    #
# ==========================================================================#

def _compute_scores(
    features: list[dict[str, Any]],
    indices: list[int],
    *,
    w_l: float,
    w_d: float,
    w_r: float,
    w_ssl: float,
) -> np.ndarray:
    """Vectorised empirical risk scores (Eq. 1) for a subset of rows."""
    subset = [features[i] for i in indices]
    s_l = np.array([r["lexical_score"] for r in subset])
    s_d = np.array([r["dns_score"] for r in subset])
    s_r = np.array([r["redirect_score"] for r in subset])
    s_ssl = np.array([r["ssl_score"] for r in subset])
    return 100.0 * (w_l * s_l + w_d * s_d + w_r * s_r + w_ssl * s_ssl)


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Compute the full metric suite."""
    # Guard against AUC failure when all predictions are one class
    try:
        auc = float(roc_auc_score(y_true, scores))
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
# Ablation engine                                                           #
# ==========================================================================#

def run_ablation(
    features: list[dict[str, Any]],
    test_indices: list[int],
    *,
    config: dict[str, float],
    output_path: str = DEFAULT_OUTPUT_JSON,
) -> dict[str, Any]:
    """Execute the frozen-weight ablation study.

    For each module, zero out its weight and evaluate on the test set.
    The threshold is **always** held fixed at ``config["threshold"]``
    (paper value: 30.06).  Remaining weights are **not** re-normalised.

    Parameters
    ----------
    features:
        Complete feature matrix.
    test_indices:
        Test partition indices.
    config:
        Optimised weights and threshold.
    output_path:
        Destination JSON.

    Returns
    -------
    dict
        Keys: ``baseline``, ``ablations`` (list), ``summary``.
    """
    threshold = config["threshold"]
    base_weights = {k: config[k] for k in ("w_l", "w_d", "w_r", "w_ssl")}

    y_true = np.array([features[i]["label"] for i in test_indices])

    # --- Baseline (full model) ------------------------------------------
    baseline_scores = _compute_scores(
        features, test_indices, **base_weights
    )
    baseline_pred = (baseline_scores >= threshold).astype(int)
    baseline_metrics = _evaluate(y_true, baseline_pred, baseline_scores)

    logger.info("Baseline (full model)  F1=%.4f", baseline_metrics["f1"])

    # --- Per-module ablation --------------------------------------------
    ablations: list[dict[str, Any]] = []

    for module_name, weight_key in MODULE_KEYS.items():
        # Create ablated weight set (zero out this module)
        ablated_weights = dict(base_weights)
        ablated_weights[weight_key] = 0.0
        # Do NOT re-normalise – frozen weights

        ablated_scores = _compute_scores(
            features, test_indices, **ablated_weights
        )
        ablated_pred = (ablated_scores >= threshold).astype(int)
        ablated_metrics = _evaluate(y_true, ablated_pred, ablated_scores)

        delta_f1 = round(ablated_metrics["f1"] - baseline_metrics["f1"], 4)

        ablations.append({
            "removed_module": module_name,
            "weight_key": weight_key,
            "original_weight": base_weights[weight_key],
            "metrics": ablated_metrics,
            "delta_f1": delta_f1,
        })

        logger.info(
            "Without %-10s  F1=%.4f  ΔF1=%+.4f",
            module_name,
            ablated_metrics["f1"],
            delta_f1,
        )

    # --- Summary --------------------------------------------------------
    summary = {
        entry["removed_module"]: entry["delta_f1"]
        for entry in ablations
    }

    # --- Pretty-print ---------------------------------------------------
    _print_ablation_table(baseline_metrics, ablations)

    # --- Persist --------------------------------------------------------
    results: dict[str, Any] = {
        "config": config,
        "threshold_fixed": threshold,
        "baseline": baseline_metrics,
        "ablations": ablations,
        "summary_delta_f1": summary,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Ablation results saved to %s", output_path)

    return results


def _print_ablation_table(
    baseline: dict[str, float],
    ablations: list[dict[str, Any]],
) -> None:
    """Print a formatted ablation results table to stdout."""
    sep = "+" + "-" * 18 + "+" + "-" * 10 + "+" + "-" * 12 + "+"
    print("\n" + "=" * 44)
    print("  Ablation Study Results")
    print("=" * 44)
    print(sep)
    print(f"| {'Condition':^16s} | {'F1':^8s} | {'ΔF1':^10s} |")
    print(sep)
    print(f"| {'Full model':^16s} | {baseline['f1']:^8.4f} | {'—':^10s} |")
    print(sep)

    for entry in ablations:
        name = f"– {entry['removed_module']}"
        f1_val = entry["metrics"]["f1"]
        delta = entry["delta_f1"]
        sign = "+" if delta >= 0 else ""
        print(f"| {name:^16s} | {f1_val:^8.4f} | {sign}{delta:>8.4f} |")

    print(sep + "\n")


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="QRIntel frozen-weight ablation study.",
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
    """CLI entry-point for the ablation study."""
    args = _build_parser().parse_args(argv)

    features = load_features(csv_path=args.features, db_path=args.db)
    splits = load_split_indices(args.split)
    config = load_config(args.config)

    # Force threshold to paper value (30.06) for frozen-weight ablation
    config["threshold"] = PAPER_THRESHOLD

    run_ablation(
        features=features,
        test_indices=splits["test_indices"],
        config=config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
