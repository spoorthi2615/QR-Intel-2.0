"""Lexical Intelligence Module (s_l) for the QRIntel risk-scoring pipeline.

This module examines the *textual structure* of a URL—without making any
network requests—to produce a normalised risk sub-score in [0.0, 1.0].

Scoring Methodology
-------------------
Seven independent heuristic signals are computed and combined via a
weighted average:

1. **Shannon Entropy** of the full URL string.
   H(X) = -Σ p(x_i) · log₂(p(x_i))
   High entropy (> 4.0) is common in randomly generated phishing domains.

2. **Structural metrics** – subdomain depth, path depth, query-parameter
   count, and TLD suspiciousness.

3. **Suspicious keyword presence** – tokens such as ``login``, ``verify``,
   ``secure`` that appear disproportionately in phishing URLs.

4. **IP-address host** – use of a raw IPv4/IPv6 address instead of a
   domain name is a strong phishing indicator.

5. **URL length anomaly** – URLs longer than 75 characters are penalised.

6. **Special-character ratios** – elevated ratios of ``@``, ``-``, ``_``,
   ``.`` in the host portion indicate domain obfuscation.

7. **Punycode / IDN detection** – ``xn--`` prefixed labels suggest
   internationalised-domain homograph attacks.

All signals are individually clamped to [0.0, 1.0] before the weighted
sum, and the final score is itself clamped to [0.0, 1.0].
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs

from backend.config import LEXICAL_SUSPICIOUS_KEYWORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        ".tk", ".ml", ".ga", ".cf", ".gq",   # Free TLDs (Freenom)
        ".top", ".xyz", ".buzz", ".club",
        ".info", ".click", ".link", ".work",
        ".surf", ".rest", ".icu",
    }
)

_ENTROPY_CEILING: float = 5.5
"""Entropy values above this are clamped to 1.0."""

_ENTROPY_FLOOR: float = 2.5
"""Entropy values below this are clamped to 0.0."""

_URL_LENGTH_THRESHOLD: int = 75
"""URLs longer than this are considered anomalous."""

_MAX_URL_LENGTH: int = 200
"""Length at which the length-anomaly signal saturates at 1.0."""

# Heuristic weights for the seven signals (sum = 1.0)
_SIGNAL_WEIGHTS: dict[str, float] = {
    "entropy":       0.20,
    "structure":     0.15,
    "keywords":      0.20,
    "ip_address":    0.15,
    "length":        0.10,
    "special_chars": 0.10,
    "punycode":      0.10,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str) -> dict:
    """Compute the lexical risk sub-score for *url*.

    Parameters
    ----------
    url : str
        The URL to analyse.  It need not include a scheme; ``https://``
        will be prepended if absent.

    Returns
    -------
    dict
        A dictionary with the following keys:

        * ``score`` (float) – normalised risk in [0.0, 1.0].
        * ``details`` (dict) – per-signal breakdown for audit logging.
    """
    url = _normalise_url(url)
    parsed = urlparse(url)

    signals: dict[str, float] = {}

    try:
        signals["entropy"] = _shannon_entropy_signal(url)
    except Exception:
        logger.exception("Entropy calculation failed for %s", url)
        signals["entropy"] = 0.5

    try:
        signals["structure"] = _structural_signal(parsed)
    except Exception:
        logger.exception("Structural analysis failed for %s", url)
        signals["structure"] = 0.5

    try:
        signals["keywords"] = _keyword_signal(url)
    except Exception:
        logger.exception("Keyword analysis failed for %s", url)
        signals["keywords"] = 0.0

    try:
        signals["ip_address"] = _ip_address_signal(parsed.hostname or "")
    except Exception:
        logger.exception("IP-address check failed for %s", url)
        signals["ip_address"] = 0.0

    try:
        signals["length"] = _length_signal(url)
    except Exception:
        logger.exception("Length analysis failed for %s", url)
        signals["length"] = 0.0

    try:
        signals["special_chars"] = _special_char_signal(parsed.hostname or "")
    except Exception:
        logger.exception("Special-char analysis failed for %s", url)
        signals["special_chars"] = 0.0

    try:
        signals["punycode"] = _punycode_signal(parsed.hostname or "")
    except Exception:
        logger.exception("Punycode check failed for %s", url)
        signals["punycode"] = 0.0

    # Weighted combination
    score = sum(
        _SIGNAL_WEIGHTS[name] * value for name, value in signals.items()
    )
    score = _clamp(score)

    logger.info("Lexical score for %s: %.4f  signals=%s", url, score, signals)
    return {"score": score, "details": signals}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """Ensure *url* has a scheme so that :func:`urlparse` works correctly."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _shannon_entropy_signal(url: str) -> float:
    """Return a [0, 1] score derived from the Shannon entropy of *url*.

    Shannon entropy is defined as:

        H(X) = -Σ p(xᵢ) · log₂(p(xᵢ))

    where p(xᵢ) is the relative frequency of character xᵢ in the string.
    Uniformly random strings yield high entropy (≈ log₂(alphabet_size)).
    Phishing URLs that use randomised subdomains or base-64 slugs tend to
    exhibit entropy above 4.0, whereas natural language domains sit around
    3.0–3.5.
    """
    if not url:
        return 0.0

    length = len(url)
    counts = Counter(url)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

    # Map entropy linearly from [_ENTROPY_FLOOR, _ENTROPY_CEILING] → [0, 1]
    normalised = (entropy - _ENTROPY_FLOOR) / (_ENTROPY_CEILING - _ENTROPY_FLOOR)
    return _clamp(normalised)


def _structural_signal(parsed) -> float:
    """Score structural complexity of the URL (subdomains, path, params).

    Sub-signals (each in [0, 1], averaged):

    * **Subdomain depth** – number of dots in the hostname, normalised by 5.
    * **Path depth** – number of ``/`` segments, normalised by 6.
    * **Query-parameter count** – number of distinct keys, normalised by 5.
    * **Suspicious TLD** – 1.0 if the TLD is in the free / abused set.
    """
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    subdomain_depth = _clamp(hostname.count(".") / 5.0)
    path_depth = _clamp(len([s for s in path.split("/") if s]) / 6.0)
    param_count = _clamp(len(parse_qs(parsed.query or "")) / 5.0)

    tld_suspicious = 0.0
    for tld in _SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            tld_suspicious = 1.0
            break

    return (subdomain_depth + path_depth + param_count + tld_suspicious) / 4.0


def _keyword_signal(url: str) -> float:
    """Return fraction of suspicious keywords found in *url*.

    The URL is lower-cased and each keyword from
    :data:`backend.config.LEXICAL_SUSPICIOUS_KEYWORDS` is tested for
    substring membership.  The signal is the ratio of matched keywords
    to total keywords, producing a value in [0.0, 1.0].
    """
    url_lower = url.lower()
    matches = sum(1 for kw in LEXICAL_SUSPICIOUS_KEYWORDS if kw in url_lower)
    return _clamp(matches / max(len(LEXICAL_SUSPICIOUS_KEYWORDS), 1))


def _ip_address_signal(hostname: str) -> float:
    """Return 1.0 if *hostname* is a raw IP address, else 0.0.

    Checks both IPv4 dotted-decimal (``192.168.1.1``) and IPv6 bracket
    notation (``[::1]``).
    """
    # IPv4
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname):
        return 1.0
    # IPv6
    if hostname.startswith("[") or ":" in hostname:
        return 1.0
    return 0.0


def _length_signal(url: str) -> float:
    """Return a [0, 1] score based on URL length.

    URLs shorter than ``_URL_LENGTH_THRESHOLD`` score 0.0; those longer
    are scaled linearly up to 1.0 at ``_MAX_URL_LENGTH``.
    """
    length = len(url)
    if length <= _URL_LENGTH_THRESHOLD:
        return 0.0
    return _clamp(
        (length - _URL_LENGTH_THRESHOLD)
        / (_MAX_URL_LENGTH - _URL_LENGTH_THRESHOLD)
    )


def _special_char_signal(hostname: str) -> float:
    """Return the density of obfuscation characters in *hostname*.

    Characters checked: ``@``, ``-``, ``_``, ``.``

    The ratio of these characters to the total hostname length is doubled
    (to amplify the signal) and clamped to [0, 1].
    """
    if not hostname:
        return 0.0
    suspicious_chars = sum(1 for ch in hostname if ch in "@-_.")
    ratio = suspicious_chars / len(hostname)
    return _clamp(ratio * 2.0)


def _punycode_signal(hostname: str) -> float:
    """Return 1.0 if *hostname* contains a Punycode (``xn--``) label.

    Internationalised Domain Names that use look-alike Unicode characters
    (homograph attacks) are encoded with the ``xn--`` ACE prefix.
    """
    if "xn--" in hostname.lower():
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))
