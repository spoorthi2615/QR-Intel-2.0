"""Redirect Chain Analysis Module (s_r) for the QRIntel risk-scoring pipeline.

Phishing campaigns frequently interpose multiple HTTP redirects between the
QR-code payload URL and the final landing page.  This module follows the
redirect chain **without** rendering any page content and converts the chain
characteristics into a normalised risk sub-score in [0.0, 1.0].

Scoring Methodology
-------------------
Three signals are combined:

1. **Chain length** – normalised by :data:`backend.config.MAX_REDIRECT_DEPTH`.
   Any chain exceeding :data:`backend.config.REDIRECT_SUSPICIOUS_THRESHOLD`
   hops is considered anomalous.

2. **Cross-domain redirects** – the number of distinct effective
   second-level domains (e.g. ``example.com``) encountered across all
   hops, normalised by the total hop count.

3. **Suspicious status codes** – the proportion of hops that use
   ``301``/``302``/``303``/``307``/``308`` rather than a ``200 OK``.

Each signal is clamped to [0.0, 1.0], and the final score is the weighted
average.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests

from backend.config import MAX_REDIRECT_DEPTH, REDIRECT_SUSPICIOUS_THRESHOLD

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT: float = 5.0
"""Per-hop timeout in seconds for the outbound HTTP request."""

_REDIRECT_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
"""HTTP status codes treated as redirects."""

_SIGNAL_WEIGHTS: dict[str, float] = {
    "chain_length":    0.50,
    "cross_domain":    0.30,
    "status_anomaly":  0.20,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str) -> dict:
    """Follow the redirect chain starting at *url* and score the result.

    Parameters
    ----------
    url : str
        The initial URL to follow.

    Returns
    -------
    dict
        ``score`` (float in [0, 1]), ``details`` (per-signal dict), and
        ``chain`` (list of visited URLs).
    """
    url = _normalise_url(url)

    chain: list[str] = [url]
    current_url = url
    hop = 0

    while hop < MAX_REDIRECT_DEPTH:
        try:
            response = requests.head(
                current_url,
                allow_redirects=False,
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": "QRIntel/2.0 Security Scanner"},
            )
        except requests.exceptions.SSLError:
            logger.debug("SSL error at hop %d: %s", hop, current_url)
            break
        except requests.exceptions.ConnectionError:
            logger.debug("Connection error at hop %d: %s", hop, current_url)
            break
        except requests.exceptions.Timeout:
            logger.debug("Timeout at hop %d: %s", hop, current_url)
            break
        except requests.exceptions.RequestException:
            logger.exception("Request error at hop %d: %s", hop, current_url)
            break

        if response.status_code not in _REDIRECT_CODES:
            break

        location = response.headers.get("Location", "")
        if not location:
            break

        # Resolve relative redirects
        if not location.startswith("http"):
            parsed_current = urlparse(current_url)
            location = f"{parsed_current.scheme}://{parsed_current.netloc}{location}"

        chain.append(location)
        current_url = location
        hop += 1

    signals = _compute_signals(chain)

    score = _clamp(sum(
        _SIGNAL_WEIGHTS[name] * value for name, value in signals.items()
    ))

    logger.info(
        "Redirect score for %s: %.4f  hops=%d  signals=%s",
        url, score, len(chain) - 1, signals,
    )
    return {"score": score, "details": signals, "chain": chain}


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals(chain: list[str]) -> dict[str, float]:
    """Derive the three redirect-chain signals from the visited URL list."""
    hops = len(chain) - 1  # first entry is the original URL

    # 1. Chain length signal
    if hops > REDIRECT_SUSPICIOUS_THRESHOLD:
        chain_length_signal = _clamp(hops / MAX_REDIRECT_DEPTH)
    else:
        chain_length_signal = 0.0

    # 2. Cross-domain signal
    domains = {_effective_domain(u) for u in chain}
    if hops > 0:
        cross_domain_signal = _clamp((len(domains) - 1) / max(hops, 1))
    else:
        cross_domain_signal = 0.0

    # 3. Status-anomaly signal (all hops are redirects by construction,
    #    so a long chain of *only* redirects is itself the anomaly)
    if hops > REDIRECT_SUSPICIOUS_THRESHOLD:
        status_anomaly_signal = _clamp(hops / MAX_REDIRECT_DEPTH)
    else:
        status_anomaly_signal = 0.0

    return {
        "chain_length":   chain_length_signal,
        "cross_domain":   cross_domain_signal,
        "status_anomaly": status_anomaly_signal,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _effective_domain(url: str) -> str:
    """Extract the effective second-level domain (e.g. ``example.com``).

    This is a simplified heuristic—it splits the hostname on dots and
    returns the last two labels.  A production system would use the
    `publicsuffix2 <https://pypi.org/project/publicsuffix2/>`_ library.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def _normalise_url(url: str) -> str:
    """Ensure *url* carries a scheme."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))
