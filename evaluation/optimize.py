#!/usr/bin/env python3
"""Optuna hyperparameter optimisation for the QRIntel empirical risk model.

This script searches for the optimal Stage-1 module weights
(w_l, w_d, w_r, w_ssl) and decision threshold that maximise the
F1-score on the **training partition only**, preventing any data leakage
from the held-out test set.

The search space is defined as:

* Each weight ``w_i ∈ [0.0, 1.0]``, then **normalised** so ``∑w_i = 1``.
* Threshold ``∈ [10.0, 90.0]``.

The Tree-structured Parzen Estimator (TPE) sampler drives 200 trials by
default.

Typical usage
-------------
::

    python -m evaluation.optimize \\
        --features evaluation/features.csv \\
        --split    data/train_test_split_indices.json \\
        --trials   200

Outputs
-------
* ``evaluation/optimization_results.json`` – per-trial history.
* ``evaluation/config.json`` – best weights + threshold.
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
import optuna
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence Optuna's verbose trial logging (we log our own summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_CSV: str = "evaluation/features.csv"
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_SPLIT_JSON: str = "data/train_test_split_indices.json"
DEFAULT_N_TRIALS: int = 200
DEFAULT_OUTPUT_RESULTS: str = "evaluation/optimization_results.json"
DEFAULT_OUTPUT_CONFIG: str = "evaluation/config.json"


# ==========================================================================#
# Data loading helpers                                                      #
# ==========================================================================#

def load_features_csv(csv_path: str) -> list[dict[str, Any]]:
    """Load the feature matrix from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV.  Expected columns: ``url``, ``label``,
        ``lexical_score``, ``dns_score``, ``redirect_score``, ``ssl_score``.

    Returns
    -------
    list[dict]
    """
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
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


def load_features_sqlite(db_path: str) -> list[dict[str, Any]]:
    """Load features from the SQLite cache database.

    Parameters
    ----------
    db_path:
        Path to the ``qrintel_features.db`` file.

    Returns
    -------
    list[dict]
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT url, label, lexical_score, dns_score,
               redirect_score, ssl_score
        FROM features ORDER BY url
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_features(
    csv_path: str | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Load features from CSV *or* SQLite, preferring CSV if both are given.

    Parameters
    ----------
    csv_path:
        Path to features CSV.
    db_path:
        Path to SQLite cache.

    Returns
    -------
    list[dict]

    Raises
    ------
    FileNotFoundError
        If neither source exists.
    """
    if csv_path and Path(csv_path).exists():
        logger.info("Loading features from CSV: %s", csv_path)
        return load_features_csv(csv_path)
    if db_path and Path(db_path).exists():
        logger.info("Loading features from SQLite: %s", db_path)
        return load_features_sqlite(db_path)
    raise FileNotFoundError(
        f"No feature source found (tried CSV={csv_path}, DB={db_path})."
    )


def load_split_indices(json_path: str) -> dict[str, list[int]]:
    """Load train / test index arrays from JSON.

    Parameters
    ----------
    json_path:
        Path to ``train_test_split_indices.json``.  Expected keys:
        ``"train_indices"`` and ``"test_indices"``.

    Returns
    -------
    dict with ``"train_indices"`` and ``"test_indices"`` lists.
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {
        "train_indices": data["train_indices"],
        "test_indices": data["test_indices"],
    }


# ==========================================================================#
# Scoring function                                                          #
# ==========================================================================#

def compute_empirical_score(
    lexical: float,
    dns: float,
    redirect: float,
    ssl: float,
    *,
    w_l: float,
    w_d: float,
    w_r: float,
    w_ssl: float,
) -> float:
    """Compute the QRIntel empirical risk score (Eq. 1 from the paper).

    Parameters
    ----------
    lexical, dns, redirect, ssl:
        Individual module scores in ``[0, 1]``.
    w_l, w_d, w_r, w_ssl:
        Normalised module weights (should sum to 1).

    Returns
    -------
    float
        Risk score in ``[0, 100]``.
    """
    return 100.0 * (
        w_l * lexical + w_d * dns + w_r * redirect + w_ssl * ssl
    )


# ==========================================================================#
# Optuna objective                                                          #
# ==========================================================================#

def make_objective(
    features: list[dict[str, Any]],
    train_indices: list[int],
):
    """Return an Optuna objective closure bound to the training data.

    Parameters
    ----------
    features:
        Full feature matrix (list of dicts).
    train_indices:
        Indices into *features* for the training partition.

    Returns
    -------
    callable
        An ``objective(trial)`` function suitable for ``study.optimize``.
    """
    # Pre-extract numpy arrays for speed
    train = [features[i] for i in train_indices]
    y_true = np.array([r["label"] for r in train])
    s_l = np.array([r["lexical_score"] for r in train])
    s_d = np.array([r["dns_score"] for r in train])
    s_r = np.array([r["redirect_score"] for r in train])
    s_ssl = np.array([r["ssl_score"] for r in train])

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximise F1-score on the training partition."""
        # --- Sample raw weights and normalise --------------------------
        raw_wl = trial.suggest_float("raw_w_l", 0.0, 1.0)
        raw_wd = trial.suggest_float("raw_w_d", 0.0, 1.0)
        raw_wr = trial.suggest_float("raw_w_r", 0.0, 1.0)
        raw_wssl = trial.suggest_float("raw_w_ssl", 0.0, 1.0)

        total = raw_wl + raw_wd + raw_wr + raw_wssl
        if total == 0:
            return 0.0  # degenerate trial
        w_l = raw_wl / total
        w_d = raw_wd / total
        w_r = raw_wr / total
        w_ssl = raw_wssl / total

        # --- Sample threshold ------------------------------------------
        threshold = trial.suggest_float("threshold", 10.0, 90.0)

        # --- Vectorised score computation ------------------------------
        scores = 100.0 * (w_l * s_l + w_d * s_d + w_r * s_r + w_ssl * s_ssl)
        y_pred = (scores >= threshold).astype(int)

        return float(f1_score(y_true, y_pred, zero_division=0.0))

    return objective


# ==========================================================================#
# Main optimisation driver                                                  #
# ==========================================================================#

def run_optimization(
    features: list[dict[str, Any]],
    train_indices: list[int],
    n_trials: int = DEFAULT_N_TRIALS,
    *,
    output_results: str = DEFAULT_OUTPUT_RESULTS,
    output_config: str = DEFAULT_OUTPUT_CONFIG,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the Optuna TPE optimisation and persist results.

    Parameters
    ----------
    features:
        Complete feature matrix.
    train_indices:
        Indices for the training partition.
    n_trials:
        Number of Optuna trials.
    output_results:
        Path for the full trial-history JSON.
    output_config:
        Path for the best-configuration JSON.
    seed:
        Random seed for the TPE sampler.

    Returns
    -------
    dict
        Best configuration with keys ``w_l``, ``w_d``, ``w_r``, ``w_ssl``,
        ``threshold``, and ``best_f1``.
    """
    logger.info("Starting Optuna optimization (%d trials, seed=%d)", n_trials, seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="qrintel_weight_optimization",
    )

    objective = make_objective(features, train_indices)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # --- Extract best normalised weights --------------------------------
    best = study.best_trial
    raw = {
        "raw_w_l": best.params["raw_w_l"],
        "raw_w_d": best.params["raw_w_d"],
        "raw_w_r": best.params["raw_w_r"],
        "raw_w_ssl": best.params["raw_w_ssl"],
    }
    total = sum(raw.values())
    best_config: dict[str, Any] = {
        "w_l": round(raw["raw_w_l"] / total, 4),
        "w_d": round(raw["raw_w_d"] / total, 4),
        "w_r": round(raw["raw_w_r"] / total, 4),
        "w_ssl": round(raw["raw_w_ssl"] / total, 4),
        "threshold": round(best.params["threshold"], 2),
        "best_f1": round(best.value, 6),
    }

    logger.info("Best F1 = %.6f", best_config["best_f1"])
    logger.info(
        "Weights  w_l=%.4f  w_d=%.4f  w_r=%.4f  w_ssl=%.4f  threshold=%.2f",
        best_config["w_l"],
        best_config["w_d"],
        best_config["w_r"],
        best_config["w_ssl"],
        best_config["threshold"],
    )

    # --- Save per-trial history -----------------------------------------
    trial_history = []
    for t in study.trials:
        raw_total = (
            t.params["raw_w_l"]
            + t.params["raw_w_d"]
            + t.params["raw_w_r"]
            + t.params["raw_w_ssl"]
        )
        if raw_total == 0:
            raw_total = 1.0  # prevent division by zero for degenerate trial
        trial_history.append({
            "number": t.number,
            "f1": t.value,
            "w_l": round(t.params["raw_w_l"] / raw_total, 4),
            "w_d": round(t.params["raw_w_d"] / raw_total, 4),
            "w_r": round(t.params["raw_w_r"] / raw_total, 4),
            "w_ssl": round(t.params["raw_w_ssl"] / raw_total, 4),
            "threshold": round(t.params["threshold"], 2),
        })

    os.makedirs(os.path.dirname(output_results) or ".", exist_ok=True)
    with open(output_results, "w", encoding="utf-8") as fh:
        json.dump(
            {"best": best_config, "trials": trial_history},
            fh,
            indent=2,
        )
    logger.info("Trial history saved to %s", output_results)

    # --- Save best config -----------------------------------------------
    os.makedirs(os.path.dirname(output_config) or ".", exist_ok=True)
    with open(output_config, "w", encoding="utf-8") as fh:
        json.dump(best_config, fh, indent=2)
    logger.info("Best config saved to %s", output_config)

    return best_config


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the optimisation CLI."""
    parser = argparse.ArgumentParser(
        description="Optuna weight / threshold optimisation for QRIntel.",
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
        help="Path to SQLite feature cache (fallback if CSV missing).",
    )
    parser.add_argument(
        "--split", "-s",
        default=DEFAULT_SPLIT_JSON,
        help="Path to train_test_split_indices.json.",
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TPE sampler.",
    )
    parser.add_argument(
        "--output-results",
        default=DEFAULT_OUTPUT_RESULTS,
        help="Path for full trial-history JSON.",
    )
    parser.add_argument(
        "--output-config",
        default=DEFAULT_OUTPUT_CONFIG,
        help="Path for best-config JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the Optuna optimisation script."""
    args = _build_parser().parse_args(argv)

    features = load_features(csv_path=args.features, db_path=args.db)
    splits = load_split_indices(args.split)

    run_optimization(
        features=features,
        train_indices=splits["train_indices"],
        n_trials=args.trials,
        output_results=args.output_results,
        output_config=args.output_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
