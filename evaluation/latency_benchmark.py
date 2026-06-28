#!/usr/bin/env python3
"""Latency benchmarking for the QRIntel lexical analysis module.

Measures the wall-clock execution time of the lexical analyser over a
configurable number of trials, reporting mean, median, 95th-percentile,
and 99th-percentile latencies in milliseconds.

If the core ``lexical_analyzer`` module is not yet available, a
**reference stub** implementation is used that mirrors the Shannon-entropy
and heuristic scoring pipeline described in the paper (O(n) complexity).

Typical usage
-------------
::

    python -m evaluation.latency_benchmark --trials 1000

Outputs
-------
* ``evaluation/latency_results.json``
* Formatted console summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import time
from typing import Any, Callable
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Project-local import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from backend.core.lexical_analyzer import lexical_analyzer as _core_analyzer
except ImportError:
    _core_analyzer = None  # type: ignore[assignment]

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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRIALS: int = 1000
DEFAULT_OUTPUT_JSON: str = "evaluation/latency_results.json"

# A representative set of URLs covering both benign and malicious patterns
# for realistic benchmarking across varied string lengths and structures.
BENCHMARK_URLS: list[str] = [
    "https://www.google.com/search?q=qr+code+security",
    "https://secure-login.paypa1.com/signin/verify?token=abc123&redirect=https://evil.site/steal",
    "https://bit.ly/3xF9kQ",
    "https://www.amazon.co.uk/dp/B0BSHF7C29/ref=cm_sw_r_cp_api_i_dl_ABC123",
    "http://192.168.1.1:8080/admin/login.php?user=admin&pass=admin",
    "https://xn--pypal-4ve.com/login",
    "https://docs.google.com/forms/d/e/1FAIpQLSe_example/viewform",
    "https://www.wikipedia.org/wiki/QR_code",
    "http://free-iphone15-giveaway.xyz/claim?id=8472947&ref=qr_scan",
    "https://cdn.shopify.com/s/files/1/0553/7896/products/item.jpg",
]


# ==========================================================================#
# Reference stub — used when backend.core.lexical_analyzer is unavailable   #
# ==========================================================================#

def _shannon_entropy(text: str) -> float:
    """Compute Shannon entropy H(X) of a character string.

    Implements Eq. 2 from the manuscript:

    .. math::

        H(X) = -\\sum_{i=1}^{n} P(x_i) \\log_2 P(x_i)

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    float
        Entropy in bits.  Returns ``0.0`` for empty strings.
    """
    if not text:
        return 0.0
    length = len(text)
    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _stub_lexical_analyzer(url: str) -> float:
    """Reference-stub lexical analyser for benchmarking.

    Mirrors the O(n) heuristic pipeline described in the paper:
    1. Parse URL components.
    2. Compute Shannon entropy of the full URL.
    3. Count suspicious indicators (IP literals, deep subdomains,
       excessive special characters, high entropy).
    4. Return a normalised ``[0, 1]`` risk score.

    This stub is **intentionally lightweight** to measure pure Python
    overhead without network I/O.

    Parameters
    ----------
    url:
        Raw URL string.

    Returns
    -------
    float
        Risk score in ``[0.0, 1.0]``.
    """
    score = 0.0
    max_score = 5.0  # normalisation denominator

    try:
        parsed = urlparse(url)
    except Exception:
        return 0.5

    hostname = parsed.hostname or ""

    # --- Heuristic 1: URL length ----------------------------------------
    if len(url) > 75:
        score += 1.0
    elif len(url) > 54:
        score += 0.5

    # --- Heuristic 2: Shannon entropy -----------------------------------
    entropy = _shannon_entropy(url)
    if entropy > 4.5:
        score += 1.0
    elif entropy > 3.5:
        score += 0.5

    # --- Heuristic 3: IP address as hostname ----------------------------
    # Simple check for numeric-only hostname segments
    parts = hostname.split(".")
    if all(p.isdigit() for p in parts if p):
        score += 1.0

    # --- Heuristic 4: subdomain depth -----------------------------------
    subdomain_depth = len(parts) - 2 if len(parts) > 2 else 0
    if subdomain_depth >= 3:
        score += 1.0
    elif subdomain_depth >= 1:
        score += 0.3

    # --- Heuristic 5: special character density -------------------------
    special = sum(1 for c in url if c in "@!#$%^&*()=+[]{}|;',<>?")
    if special > 5:
        score += 1.0
    elif special > 2:
        score += 0.5

    return min(score / max_score, 1.0)


# ==========================================================================#
# Benchmarking engine                                                       #
# ==========================================================================#

def benchmark_function(
    func: Callable[[str], Any],
    urls: list[str],
    n_trials: int,
) -> list[float]:
    """Benchmark a callable over multiple URLs and trials.

    Each trial iterates through all *urls*, calling *func* once per URL.
    The reported latency is the **per-call** time (total trial time
    divided by the number of URLs).

    Parameters
    ----------
    func:
        The function to benchmark (e.g. ``lexical_analyzer``).
    urls:
        List of URLs to analyse in each trial.
    n_trials:
        Number of benchmark trials.

    Returns
    -------
    list[float]
        Per-call latencies in **seconds** for each trial.
    """
    latencies: list[float] = []

    # Warm-up pass (excluded from measurements)
    for url in urls:
        func(url)

    for _ in range(n_trials):
        start = time.perf_counter()
        for url in urls:
            func(url)
        elapsed = time.perf_counter() - start
        per_call = elapsed / len(urls)
        latencies.append(per_call)

    return latencies


def compute_statistics(latencies_sec: list[float]) -> dict[str, float]:
    """Compute summary statistics from per-call latencies.

    Parameters
    ----------
    latencies_sec:
        Per-call latencies in seconds.

    Returns
    -------
    dict
        Keys: ``mean_ms``, ``median_ms``, ``p95_ms``, ``p99_ms``,
        ``min_ms``, ``max_ms``, ``stdev_ms``, ``n_trials``.
    """
    ms = [t * 1000.0 for t in latencies_sec]
    ms_sorted = sorted(ms)
    n = len(ms_sorted)

    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)

    return {
        "mean_ms": round(statistics.mean(ms), 4),
        "median_ms": round(statistics.median(ms), 4),
        "p95_ms": round(ms_sorted[min(p95_idx, n - 1)], 4),
        "p99_ms": round(ms_sorted[min(p99_idx, n - 1)], 4),
        "min_ms": round(min(ms), 4),
        "max_ms": round(max(ms), 4),
        "stdev_ms": round(statistics.stdev(ms) if n > 1 else 0.0, 4),
        "n_trials": n,
    }


# ==========================================================================#
# Main driver                                                               #
# ==========================================================================#

def run_benchmark(
    n_trials: int = DEFAULT_N_TRIALS,
    output_path: str = DEFAULT_OUTPUT_JSON,
) -> dict[str, Any]:
    """Execute the latency benchmark and save results.

    Parameters
    ----------
    n_trials:
        Number of measurement trials.
    output_path:
        Destination JSON path.

    Returns
    -------
    dict
        Benchmark results including statistics and configuration.
    """
    # Select the analyser function
    if _core_analyzer is not None:
        analyzer_fn = _core_analyzer
        analyzer_source = "backend.core.lexical_analyzer"
        logger.info("Using core lexical_analyzer from backend.")
    else:
        analyzer_fn = _stub_lexical_analyzer
        analyzer_source = "reference_stub (backend module not available)"
        logger.info(
            "Core lexical_analyzer not found — using reference stub."
        )

    logger.info(
        "Benchmarking %d trials × %d URLs = %d calls",
        n_trials, len(BENCHMARK_URLS), n_trials * len(BENCHMARK_URLS),
    )

    latencies = benchmark_function(analyzer_fn, BENCHMARK_URLS, n_trials)
    stats = compute_statistics(latencies)

    # --- Console output -------------------------------------------------
    print("\n" + "=" * 50)
    print("  QRIntel Lexical Module – Latency Benchmark")
    print("=" * 50)
    print(f"  Analyser:   {analyzer_source}")
    print(f"  Trials:     {n_trials}")
    print(f"  URLs/trial: {len(BENCHMARK_URLS)}")
    print("-" * 50)
    print(f"  Mean:       {stats['mean_ms']:.4f} ms")
    print(f"  Median:     {stats['median_ms']:.4f} ms")
    print(f"  P95:        {stats['p95_ms']:.4f} ms")
    print(f"  P99:        {stats['p99_ms']:.4f} ms")
    print(f"  Min:        {stats['min_ms']:.4f} ms")
    print(f"  Max:        {stats['max_ms']:.4f} ms")
    print(f"  Stdev:      {stats['stdev_ms']:.4f} ms")
    print("=" * 50 + "\n")

    # --- Persist --------------------------------------------------------
    results: dict[str, Any] = {
        "module": "lexical_analyzer",
        "analyzer_source": analyzer_source,
        "n_trials": n_trials,
        "n_urls_per_trial": len(BENCHMARK_URLS),
        "total_calls": n_trials * len(BENCHMARK_URLS),
        "statistics": stats,
        "benchmark_urls": BENCHMARK_URLS,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Latency results saved to %s", output_path)

    return results


# ==========================================================================#
# CLI entry-point                                                           #
# ==========================================================================#

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark latency of the QRIntel lexical analyser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trials", "-n",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of benchmark trials.",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for the latency benchmark."""
    args = _build_parser().parse_args(argv)
    run_benchmark(n_trials=args.trials, output_path=args.output)


if __name__ == "__main__":
    main()
