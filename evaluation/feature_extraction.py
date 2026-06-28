#!/usr/bin/env python3
"""Feature-extraction pipeline for the QRIntel evaluation suite.

This module loads URLs from a CSV dataset, runs each URL through the
four Stage-1 intelligence modules (lexical, DNS, redirect, SSL), caches
every raw result in a local SQLite database, and exports a tidy
feature-matrix CSV ready for downstream optimisation and evaluation.

Typical usage
-------------
::

    # From the repository root:
    python -m evaluation.feature_extraction \\
        --input  data/QRIntel_21k_Dataset.csv \\
        --db     evaluation/qrintel_features.db \\
        --output evaluation/features.csv \\
        --workers 8

CSV input format
~~~~~~~~~~~~~~~~
The input CSV **must** contain at least two columns:

* ``url``   – the raw URL string.
* ``label`` – integer class label (1 = phishing / malicious, 0 = benign).

SQLite cache schema
~~~~~~~~~~~~~~~~~~~
A single ``features`` table stores one row per URL with columns:
``url``, ``label``, ``lexical_score``, ``dns_score``, ``redirect_score``,
``ssl_score``, and ``extracted_at`` (ISO-8601 timestamp).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project-local imports – graceful fallback when core modules are absent
# ---------------------------------------------------------------------------
try:
    from backend.core.lexical_analyzer import lexical_analyzer
except ImportError:  # pragma: no cover
    lexical_analyzer = None  # type: ignore[assignment]

try:
    from backend.core.dns_analyzer import dns_analyzer
except ImportError:  # pragma: no cover
    dns_analyzer = None  # type: ignore[assignment]

try:
    from backend.core.redirect_analyzer import redirect_analyzer
except ImportError:  # pragma: no cover
    redirect_analyzer = None  # type: ignore[assignment]

try:
    from backend.core.ssl_analyzer import ssl_analyzer
except ImportError:  # pragma: no cover
    ssl_analyzer = None  # type: ignore[assignment]

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
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH: str = "evaluation/qrintel_features.db"
DEFAULT_OUTPUT_CSV: str = "evaluation/features.csv"
DEFAULT_WORKERS: int = 8

FEATURES_TABLE_DDL: str = """
CREATE TABLE IF NOT EXISTS features (
    url            TEXT    PRIMARY KEY,
    label          INTEGER NOT NULL,
    lexical_score  REAL,
    dns_score      REAL,
    redirect_score REAL,
    ssl_score      REAL,
    extracted_at   TEXT    NOT NULL
);
"""


# ==========================================================================#
# Helper: safe analyser invocation                                          #
# ==========================================================================#

def _safe_call(
    analyzer_fn: Any | None,
    url: str,
    *,
    module_name: str,
) -> float:
    """Invoke an analyser function and return its normalised ``[0, 1]`` score.

    Parameters
    ----------
    analyzer_fn:
        The analyser callable (e.g. ``lexical_analyzer``).  May be ``None``
        if the corresponding core module is not yet implemented.
    url:
        The URL to analyse.
    module_name:
        Human-readable name used only in log messages.

    Returns
    -------
    float
        Normalised score in ``[0.0, 1.0]``.  Returns ``0.0`` on error.
    """
    if analyzer_fn is None:
        return 0.0
    try:
        result = analyzer_fn(url)
        # Analysers may return a dict with a "score" key, or a plain float.
        if isinstance(result, dict):
            score = float(result.get("score", result.get("risk_score", 0.0)))
        else:
            score = float(result)
        return max(0.0, min(1.0, score))
    except Exception:
        logger.debug("%-10s  ERROR for %s", module_name, url, exc_info=True)
        return 0.0


# ==========================================================================#
# Per-URL extraction                                                        #
# ==========================================================================#

def extract_features_for_url(
    url: str,
    label: int,
) -> dict[str, Any]:
    """Run all four Stage-1 analysers on a single URL.

    Parameters
    ----------
    url:
        The raw URL string.
    label:
        Ground-truth class label (``1`` = malicious, ``0`` = benign).

    Returns
    -------
    dict
        Keys: ``url``, ``label``, ``lexical_score``, ``dns_score``,
        ``redirect_score``, ``ssl_score``, ``extracted_at``.
    """
    return {
        "url": url,
        "label": label,
        "lexical_score": _safe_call(lexical_analyzer, url, module_name="lexical"),
        "dns_score": _safe_call(dns_analyzer, url, module_name="dns"),
        "redirect_score": _safe_call(redirect_analyzer, url, module_name="redirect"),
        "ssl_score": _safe_call(ssl_analyzer, url, module_name="ssl"),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


# ==========================================================================#
# SQLite caching layer                                                      #
# ==========================================================================#

def init_database(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite cache database and return a connection.

    Parameters
    ----------
    db_path:
        Filesystem path to the ``.db`` file.

    Returns
    -------
    sqlite3.Connection
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(FEATURES_TABLE_DDL)
    conn.commit()
    return conn


def url_is_cached(conn: sqlite3.Connection, url: str) -> bool:
    """Return ``True`` if *url* already has a cached feature row."""
    row = conn.execute(
        "SELECT 1 FROM features WHERE url = ? LIMIT 1", (url,)
    ).fetchone()
    return row is not None


def upsert_features(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert or replace a feature row in the cache.

    Parameters
    ----------
    conn:
        Active SQLite connection.
    row:
        Feature dictionary produced by :func:`extract_features_for_url`.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO features
            (url, label, lexical_score, dns_score, redirect_score,
             ssl_score, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["url"],
            row["label"],
            row["lexical_score"],
            row["dns_score"],
            row["redirect_score"],
            row["ssl_score"],
            row["extracted_at"],
        ),
    )


def load_cached_features(db_path: str) -> list[dict[str, Any]]:
    """Load every cached feature row from SQLite as a list of dicts.

    Parameters
    ----------
    db_path:
        Path to the ``.db`` file.

    Returns
    -------
    list[dict]
        Each dict mirrors the ``features`` table columns.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT url, label, lexical_score, dns_score,
               redirect_score, ssl_score
        FROM features
        ORDER BY url
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ==========================================================================#
# Dataset loader                                                            #
# ==========================================================================#

def load_dataset(csv_path: str) -> list[tuple[str, int]]:
    """Read URLs and labels from a two-column CSV.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.  Expected columns: ``url``, ``label``.

    Returns
    -------
    list[tuple[str, int]]
        ``(url, label)`` pairs.

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If the CSV is missing required columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    records: list[tuple[str, int]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        # Accept flexible column naming
        url_col = _find_column(reader.fieldnames, {"url", "URL", "Url"})
        label_col = _find_column(
            reader.fieldnames, {"label", "Label", "class", "Class"}
        )
        for row in reader:
            url = row[url_col].strip()
            if not url:
                continue
            try:
                label = int(row[label_col])
            except (ValueError, TypeError):
                logger.warning("Skipping row with non-integer label: %s", row)
                continue
            records.append((url, label))

    logger.info("Loaded %d URLs from %s", len(records), csv_path)
    return records


def _find_column(fieldnames: list[str], candidates: set[str]) -> str:
    """Return the first fieldname matching any of *candidates*.

    Raises
    ------
    ValueError
        If none of the candidates is present.
    """
    for name in fieldnames:
        if name in candidates:
            return name
    raise ValueError(
        f"CSV must contain one of {candidates}; found {fieldnames}"
    )


# ==========================================================================#
# CSV export                                                                #
# ==========================================================================#

def export_features_csv(
    features: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Write the feature matrix to a CSV file.

    Parameters
    ----------
    features:
        List of feature dicts (see :func:`extract_features_for_url`).
    output_path:
        Destination CSV path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = [
        "url", "label", "lexical_score", "dns_score",
        "redirect_score", "ssl_score",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features)
    logger.info("Exported %d rows to %s", len(features), output_path)


# ==========================================================================#
# Parallel extraction orchestrator                                          #
# ==========================================================================#

def run_extraction(
    csv_path: str,
    db_path: str = DEFAULT_DB_PATH,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    max_workers: int = DEFAULT_WORKERS,
    *,
    skip_cached: bool = True,
) -> list[dict[str, Any]]:
    """Execute the full extraction pipeline.

    Steps
    -----
    1. Load URLs from *csv_path*.
    2. Open (or create) the SQLite cache at *db_path*.
    3. Skip URLs that are already cached (unless *skip_cached* is ``False``).
    4. Extract features for remaining URLs in parallel.
    5. Cache every result to SQLite.
    6. Export the complete feature matrix to *output_csv*.

    Parameters
    ----------
    csv_path:
        Path to the input dataset CSV.
    db_path:
        Path to the SQLite cache database.
    output_csv:
        Destination for the exported feature-matrix CSV.
    max_workers:
        Number of ``ThreadPoolExecutor`` worker threads.
    skip_cached:
        If ``True`` (default), skip URLs that already have cached features.

    Returns
    -------
    list[dict]
        The complete feature matrix (cached + freshly extracted).
    """
    dataset = load_dataset(csv_path)
    conn = init_database(db_path)

    # --- Determine which URLs need extraction --------------------------
    if skip_cached:
        to_extract = [
            (url, label)
            for url, label in dataset
            if not url_is_cached(conn, url)
        ]
        cached_count = len(dataset) - len(to_extract)
        logger.info(
            "Cache hit: %d / %d URLs already extracted", cached_count, len(dataset)
        )
    else:
        to_extract = dataset

    # --- Parallel extraction -------------------------------------------
    total = len(to_extract)
    completed = 0
    start_time = time.monotonic()

    if total > 0:
        logger.info(
            "Starting parallel extraction for %d URLs (%d workers)",
            total,
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(extract_features_for_url, url, label): url
                for url, label in to_extract
            }
            for future in as_completed(future_map):
                url = future_map[future]
                try:
                    row = future.result()
                    upsert_features(conn, row)
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        elapsed = time.monotonic() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(
                            "Progress: %d / %d  (%.1f URLs/s)",
                            completed,
                            total,
                            rate,
                        )
                except Exception:
                    logger.error(
                        "Failed to extract features for %s", url, exc_info=True
                    )
        conn.commit()
        logger.info(
            "Extraction complete: %d URLs in %.1f s",
            completed,
            time.monotonic() - start_time,
        )
    else:
        logger.info("Nothing to extract – all URLs already cached.")

    conn.close()

    # --- Export full matrix from cache ----------------------------------
    all_features = load_cached_features(db_path)
    export_features_csv(all_features, output_csv)
    return all_features


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser for the feature-extraction CLI."""
    parser = argparse.ArgumentParser(
        description="QRIntel feature-extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV (url, label columns).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite feature-cache database.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the exported feature-matrix CSV.",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker threads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract features even if already cached.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the feature-extraction pipeline."""
    args = _build_parser().parse_args(argv)
    run_extraction(
        csv_path=args.input,
        db_path=args.db,
        output_csv=args.output,
        max_workers=args.workers,
        skip_cached=not args.force,
    )


if __name__ == "__main__":
    main()
