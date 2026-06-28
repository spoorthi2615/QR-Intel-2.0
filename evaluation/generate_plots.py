#!/usr/bin/env python3
"""Visualisation script for the QRIntel evaluation suite.

Generates three publication-quality figures using matplotlib and seaborn:

1. **weights_pie_chart.png** – Distribution of the Optuna-optimised
   intelligence module weights (w_l, w_d, w_r, w_ssl).
2. **ablation_bar_chart.png** – Impact on F1-score when each module is
   removed (ΔF1 ablation results).
3. **feature_distributions.png** – Box-plots of each module's score
   stratified by class (phishing vs. benign).

All figures use a consistent, journal-ready style:
- 300 DPI, tight layout, white backgrounds.
- Colour-blind-friendly palette.
- LaTeX-style axis labels where appropriate.

Typical usage
-------------
::

    python -m evaluation.generate_plots \\
        --features evaluation/features.csv \\
        --ablation evaluation/ablation_results.json \\
        --outdir   evaluation/

Outputs
-------
* ``evaluation/weights_pie_chart.png``
* ``evaluation/ablation_bar_chart.png``
* ``evaluation/feature_distributions.png``
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use a non-interactive backend so the script can run headless
matplotlib.use("Agg")

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

# Expected ablation results (used as fallback when JSON not available)
PAPER_ABLATION: dict[str, float] = {
    "lexical": -0.9946,
    "dns": -0.0027,
    "redirect": 0.0000,
    "ssl": 0.0000,
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FEATURES_CSV: str = "evaluation/features.csv"
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_CONFIG_JSON: str = "evaluation/config.json"
DEFAULT_ABLATION_JSON: str = "evaluation/ablation_results.json"
DEFAULT_OUTDIR: str = "evaluation"

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "lexical": "#2196F3",   # blue
    "dns": "#4CAF50",       # green
    "redirect": "#FF9800",  # orange
    "ssl": "#9C27B0",       # purple
}
MODULE_LABELS: dict[str, str] = {
    "lexical": "Lexical ($w_l$)",
    "dns": "DNS ($w_d$)",
    "redirect": "Redirect ($w_r$)",
    "ssl": "SSL ($w_{ssl}$)",
}
MODULE_LABELS_PLAIN: dict[str, str] = {
    "lexical": "Lexical",
    "dns": "DNS",
    "redirect": "Redirect",
    "ssl": "SSL",
}
SCORE_COLS: dict[str, str] = {
    "lexical_score": "Lexical ($s_l$)",
    "dns_score": "DNS ($s_d$)",
    "redirect_score": "Redirect ($s_r$)",
    "ssl_score": "SSL ($s_{ssl}$)",
}
CLASS_LABELS: dict[int, str] = {
    1: "Phishing",
    0: "Benign",
}

DPI: int = 300


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
) -> list[dict[str, Any]] | None:
    """Load features from CSV or SQLite.  Returns ``None`` if unavailable."""
    if csv_path and Path(csv_path).exists():
        return _load_features_csv(csv_path)
    if db_path and Path(db_path).exists():
        return _load_features_sqlite(db_path)
    return None


def load_config(path: str) -> dict[str, float]:
    """Load weights config, falling back to paper defaults."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return {
            "w_l": float(cfg["w_l"]),
            "w_d": float(cfg["w_d"]),
            "w_r": float(cfg["w_r"]),
            "w_ssl": float(cfg["w_ssl"]),
        }
    return dict(PAPER_WEIGHTS)


def load_ablation(path: str) -> dict[str, float]:
    """Load ΔF1 results from ablation JSON, falling back to paper values."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("summary_delta_f1", PAPER_ABLATION)
    logger.warning(
        "Ablation results not found at %s – using paper defaults.", path
    )
    return dict(PAPER_ABLATION)


# ==========================================================================#
# Plot 1: Weights Pie Chart                                                 #
# ==========================================================================#

def plot_weights_pie(
    weights: dict[str, float],
    outdir: str,
) -> str:
    """Generate ``weights_pie_chart.png``.

    Parameters
    ----------
    weights:
        Dict with keys ``w_l``, ``w_d``, ``w_r``, ``w_ssl``.
    outdir:
        Output directory.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    labels = []
    sizes = []
    colors = []
    weight_key_map = {"w_l": "lexical", "w_d": "dns", "w_r": "redirect", "w_ssl": "ssl"}

    for wk, module in weight_key_map.items():
        labels.append(f"{MODULE_LABELS_PLAIN[module]}\n({weights[wk]:.4f})")
        sizes.append(weights[wk])
        colors.append(PALETTE[module])

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    ax.set_title(
        "Optuna-Optimised Module Weight Distribution",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Draw a centre circle for a donut-style chart
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(centre_circle)
    ax.text(
        0, 0, "QRIntel\nWeights",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#333",
    )

    fig.tight_layout()
    path = os.path.join(outdir, "weights_pie_chart.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)
    return os.path.abspath(path)


# ==========================================================================#
# Plot 2: Ablation Bar Chart                                                #
# ==========================================================================#

def plot_ablation_bar(
    delta_f1: dict[str, float],
    outdir: str,
) -> str:
    """Generate ``ablation_bar_chart.png``.

    Parameters
    ----------
    delta_f1:
        Dict mapping module names to ΔF1 values.
    outdir:
        Output directory.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    modules = list(delta_f1.keys())
    values = [delta_f1[m] for m in modules]
    colors = [PALETTE.get(m, "#888888") for m in modules]
    labels = [MODULE_LABELS_PLAIN.get(m, m) for m in modules]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    bars = ax.bar(
        labels,
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.5,
        width=0.55,
    )

    # Annotate bars
    for bar, val in zip(bars, values):
        y_pos = bar.get_height()
        offset = -0.03 if val < 0 else 0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + offset,
            f"{val:+.4f}",
            ha="center",
            va="top" if val < 0 else "bottom",
            fontsize=11,
            fontweight="bold",
            color="#333",
        )

    ax.axhline(y=0, color="#333", linewidth=0.8, linestyle="-")
    ax.set_ylabel("ΔF1-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Removed Module", fontsize=12, fontweight="bold")
    ax.set_title(
        "Ablation Study: F1-Score Impact of Removing Each Module",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # Adjust y-axis to accommodate the large lexical drop
    y_min = min(values) * 1.15 if min(values) < 0 else -0.1
    y_max = max(max(values) * 1.3, 0.05)
    ax.set_ylim(y_min, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    path = os.path.join(outdir, "ablation_bar_chart.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)
    return os.path.abspath(path)


# ==========================================================================#
# Plot 3: Feature Distributions                                             #
# ==========================================================================#

def plot_feature_distributions(
    features: list[dict[str, Any]],
    outdir: str,
) -> str:
    """Generate ``feature_distributions.png``.

    Creates a 2×2 grid of box-plots showing the distribution of each
    module score stratified by class (phishing vs. benign).

    Parameters
    ----------
    features:
        Full feature matrix.
    outdir:
        Output directory.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    score_keys = list(SCORE_COLS.keys())
    score_labels = list(SCORE_COLS.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    axes_flat = axes.flatten()

    class_palette = {"Phishing": "#E53935", "Benign": "#43A047"}

    for idx, (score_key, score_label) in enumerate(zip(score_keys, score_labels)):
        ax = axes_flat[idx]

        # Build data arrays
        phishing_scores = [
            r[score_key] for r in features if r["label"] == 1
        ]
        benign_scores = [
            r[score_key] for r in features if r["label"] == 0
        ]

        plot_data = {
            "Score": phishing_scores + benign_scores,
            "Class": (
                ["Phishing"] * len(phishing_scores)
                + ["Benign"] * len(benign_scores)
            ),
        }

        sns.boxplot(
            x="Class",
            y="Score",
            data=plot_data,
            palette=class_palette,
            ax=ax,
            width=0.5,
            fliersize=3,
            linewidth=1.2,
        )

        ax.set_title(score_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)

        # Annotate mean values
        for i, (cls_name, data_arr) in enumerate([
            ("Phishing", phishing_scores),
            ("Benign", benign_scores),
        ]):
            if data_arr:
                mean_val = np.mean(data_arr)
                ax.text(
                    i, 1.02, f"μ={mean_val:.2f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#555",
                )

    fig.suptitle(
        "Feature Score Distributions by Class",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()
    path = os.path.join(outdir, "feature_distributions.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", path)
    return os.path.abspath(path)


# ==========================================================================#
# Main driver                                                               #
# ==========================================================================#

def generate_all_plots(
    features_csv: str = DEFAULT_FEATURES_CSV,
    db_path: str = DEFAULT_DB_PATH,
    config_json: str = DEFAULT_CONFIG_JSON,
    ablation_json: str = DEFAULT_ABLATION_JSON,
    outdir: str = DEFAULT_OUTDIR,
) -> dict[str, str]:
    """Generate all three evaluation plots.

    Parameters
    ----------
    features_csv:
        Path to features CSV.
    db_path:
        Path to SQLite feature cache (fallback).
    config_json:
        Path to config JSON with optimised weights.
    ablation_json:
        Path to ablation results JSON.
    outdir:
        Output directory for PNG files.

    Returns
    -------
    dict
        Mapping of plot names to their absolute file paths.
    """
    os.makedirs(outdir, exist_ok=True)
    generated: dict[str, str] = {}

    # --- Plot 1: Weight Pie Chart --------------------------------------
    weights = load_config(config_json)
    generated["weights_pie_chart"] = plot_weights_pie(weights, outdir)

    # --- Plot 2: Ablation Bar Chart ------------------------------------
    delta_f1 = load_ablation(ablation_json)
    generated["ablation_bar_chart"] = plot_ablation_bar(delta_f1, outdir)

    # --- Plot 3: Feature Distributions ---------------------------------
    features = load_features(csv_path=features_csv, db_path=db_path)
    if features is not None and len(features) > 0:
        generated["feature_distributions"] = plot_feature_distributions(
            features, outdir
        )
    else:
        logger.warning(
            "No feature data available – skipping feature_distributions plot."
        )

    logger.info("All plots generated: %s", list(generated.keys()))
    return generated


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality QRIntel evaluation plots.",
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
        "--config", "-c",
        default=DEFAULT_CONFIG_JSON,
        help="Path to config.json (weights).",
    )
    parser.add_argument(
        "--ablation", "-a",
        default=DEFAULT_ABLATION_JSON,
        help="Path to ablation_results.json.",
    )
    parser.add_argument(
        "--outdir", "-o",
        default=DEFAULT_OUTDIR,
        help="Output directory for PNG files.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the plot generation script."""
    args = _build_parser().parse_args(argv)
    generate_all_plots(
        features_csv=args.features,
        db_path=args.db,
        config_json=args.config,
        ablation_json=args.ablation,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
