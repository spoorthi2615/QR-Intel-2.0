"""Two-Stage Risk Orchestrator for the QRIntel pipeline.

This is the central module that fuses all sub-scores into a single
verdict.  It implements the complete scoring formula:

Stage 1 – Empirical Score
--------------------------
    EmpiricalScore = 100 × (w_l·s_l + w_d·s_d + w_r·s_r + w_ssl·s_ssl)

where the weights are sourced from :mod:`backend.config` and each s_*
is the normalised [0, 1] output of the corresponding analyser.

Stage 2 – Contextual Overrides
-------------------------------
    FinalScore = clamp(EmpiricalScore + CanonicalPenalty
                       + BrandPenalty + GraphModifier, 0, 100)

Overrides:

* **Canonical penalty** (−40) – applied when the effective domain matches
  :data:`backend.config.KNOWN_CANONICAL_DOMAINS`.
* **Brand-impersonation penalty** (+90) – applied when the domain is
  within Levenshtein distance [1, 3] of a known brand.
* **Graph modifier** (0–20) – proportional to similarity with
  previously observed malicious domains.

Verdict Thresholds
------------------
    * score < THRESHOLD  → SAFE
    * THRESHOLD ≤ score < 70 → SUSPICIOUS
    * score ≥ 70         → MALICIOUS

Every invocation produces a complete JSON-serialisable **audit log**
that records every sub-score, signal, penalty, and the reasoning chain
for full explainability.
"""

from __future__ import annotations

import datetime
import logging
import re
import uuid
from urllib.parse import urlparse

from backend.config import (
    CANONICAL_PENALTY,
    KNOWN_CANONICAL_DOMAINS,
    THRESHOLD,
    WEIGHT_DNS,
    WEIGHT_LEXICAL,
    WEIGHT_REDIRECT,
    WEIGHT_SSL,
)
from backend.core import (
    brand_analyzer,
    dns_analyzer,
    graph_analyzer,
    lexical_analyzer,
    redirect_analyzer,
    ssl_analyzer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(url: str) -> dict:
    """Run the full two-stage risk analysis on *url*.

    Parameters
    ----------
    url : str
        The URL extracted from a QR code (or supplied directly).

    Returns
    -------
    dict
        A JSON-serialisable dictionary with the following top-level keys:

        * ``scan_id`` (str) – unique identifier for this scan.
        * ``url`` (str) – the analysed URL.
        * ``verdict`` (str) – ``"SAFE"``, ``"SUSPICIOUS"``, or
          ``"MALICIOUS"``.
        * ``risk_score`` (float) – final score in [0, 100].
        * ``empirical_score`` (float) – Stage-1 score before overrides.
        * ``audit_log`` (dict) – full explainability trace.
    """
    scan_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    url = _normalise_url(url)

    # ------------------------------------------------------------------
    # Stage 1 – Sub-score collection
    # ------------------------------------------------------------------
    lexical_result = _safe_analyse("lexical", lexical_analyzer.analyse, url)
    dns_result = _safe_analyse("dns", dns_analyzer.analyse, url)
    redirect_result = _safe_analyse("redirect", redirect_analyzer.analyse, url)
    ssl_result = _safe_analyse("ssl", ssl_analyzer.analyse, url)

    s_l = lexical_result.get("score", 0.0)
    s_d = dns_result.get("score", 0.0)
    s_r = redirect_result.get("score", 0.0)
    s_ssl = ssl_result.get("score", 0.0)

    empirical_score = 100.0 * (
        WEIGHT_LEXICAL * s_l
        + WEIGHT_DNS * s_d
        + WEIGHT_REDIRECT * s_r
        + WEIGHT_SSL * s_ssl
    )

    # ------------------------------------------------------------------
    # Stage 2 – Contextual overrides
    # ------------------------------------------------------------------

    # 2a. Canonical domain check
    hostname = _extract_hostname(url)
    is_canonical = hostname in KNOWN_CANONICAL_DOMAINS
    canonical_adjustment = CANONICAL_PENALTY if is_canonical else 0.0

    # 2b. Brand impersonation
    brand_result = brand_analyzer.analyse(url)
    brand_adjustment = brand_result.get("penalty", 0.0)

    # 2c. Graph intelligence
    signal_profile = {
        "lexical": s_l,
        "dns": s_d,
        "redirect": s_r,
        "ssl": s_ssl,
    }
    # We need a preliminary verdict for the graph module.
    preliminary_verdict = _classify(empirical_score + canonical_adjustment + brand_adjustment)
    graph_result = graph_analyzer.analyse(url, signal_profile, preliminary_verdict)
    graph_adjustment = graph_result.get("risk_modifier", 0.0)

    # Final score
    final_score = _clamp(
        empirical_score + canonical_adjustment + brand_adjustment + graph_adjustment,
        lo=0.0,
        hi=100.0,
    )

    verdict = _classify(final_score)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------
    audit_log = {
        "scan_id": scan_id,
        "timestamp": timestamp,
        "url": url,
        "stage_1": {
            "formula": "100 * (w_l*s_l + w_d*s_d + w_r*s_r + w_ssl*s_ssl)",
            "weights": {
                "w_l": WEIGHT_LEXICAL,
                "w_d": WEIGHT_DNS,
                "w_r": WEIGHT_REDIRECT,
                "w_ssl": WEIGHT_SSL,
            },
            "sub_scores": {
                "s_l": round(s_l, 4),
                "s_d": round(s_d, 4),
                "s_r": round(s_r, 4),
                "s_ssl": round(s_ssl, 4),
            },
            "empirical_score": round(empirical_score, 2),
            "details": {
                "lexical": lexical_result.get("details", {}),
                "dns": dns_result.get("details", {}),
                "redirect": redirect_result.get("details", {}),
                "ssl": ssl_result.get("details", {}),
            },
        },
        "stage_2": {
            "canonical": {
                "is_canonical": is_canonical,
                "adjustment": canonical_adjustment,
            },
            "brand_impersonation": {
                "adjustment": brand_adjustment,
                "details": brand_result.get("details", {}),
            },
            "graph_intelligence": {
                "adjustment": round(graph_adjustment, 2),
                "details": graph_result.get("details", {}),
            },
        },
        "final": {
            "empirical_score": round(empirical_score, 2),
            "canonical_adjustment": canonical_adjustment,
            "brand_adjustment": brand_adjustment,
            "graph_adjustment": round(graph_adjustment, 2),
            "final_score": round(final_score, 2),
            "threshold": THRESHOLD,
            "verdict": verdict,
        },
        "redirect_chain": redirect_result.get("chain", []),
    }

    logger.info(
        "Scan %s complete: url=%s verdict=%s score=%.2f",
        scan_id, url, verdict, final_score,
    )

    return {
        "scan_id": scan_id,
        "url": url,
        "verdict": verdict,
        "risk_score": round(final_score, 2),
        "empirical_score": round(empirical_score, 2),
        "audit_log": audit_log,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify(score: float) -> str:
    """Map a numerical score to a categorical verdict.

    Returns
    -------
    str
        ``"SAFE"`` if *score* < ``THRESHOLD``,
        ``"SUSPICIOUS"`` if ``THRESHOLD`` ≤ *score* < 70,
        ``"MALICIOUS"`` if *score* ≥ 70.
    """
    if score >= 70.0:
        return "MALICIOUS"
    if score >= THRESHOLD:
        return "SUSPICIOUS"
    return "SAFE"


def _normalise_url(url: str) -> str:
    """Ensure *url* carries a scheme."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _extract_hostname(url: str) -> str:
    """Return the hostname (lowercase) from *url*."""
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def _safe_analyse(name: str, fn, url: str) -> dict:
    """Call analyser *fn* with *url*, catching and logging any exception.

    Returns a dict with at least a ``score`` key (defaults to 0.5 on
    failure) so that downstream arithmetic never breaks.
    """
    try:
        return fn(url)
    except Exception:
        logger.exception("Analyser '%s' failed for %s", name, url)
        return {"score": 0.5, "details": {"error": f"{name} analyser failed"}}


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))
