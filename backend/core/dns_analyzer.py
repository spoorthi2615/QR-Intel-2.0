"""DNS Intelligence Module (s_d) for the QRIntel risk-scoring pipeline.

This module queries the Domain Name System for the target URL's hostname
and converts the responses into a normalised risk sub-score in [0.0, 1.0].

Scoring Methodology
-------------------
Four signals are combined with equal weight:

1. **A-record existence** – an NXDOMAIN (no A records) is highly
   suspicious and scores 1.0 immediately.

2. **MX-record absence** – legitimate organisations almost always
   publish MX records.  Absence scores 1.0.

3. **Fast-flux detection** – when a single hostname resolves to many
   A records (≥ 5) the domain may use fast-flux hosting to evade
   takedowns.

4. **Bulletproof-hosting ASN** – if the resolved IP falls inside an
   Autonomous System Number known for hosting abuse infrastructure,
   the signal is 1.0.

All DNS queries use a hard timeout (see :data:`backend.config.DNS_QUERY_TIMEOUT`)
so that network stalls never block the scoring pipeline for long.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import dns.resolver
import dns.exception

from backend.config import DNS_QUERY_TIMEOUT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASNs historically associated with bulletproof hosting.
# This is a representative sample; a production deployment would pull from
# a regularly updated threat-intelligence feed.
# ---------------------------------------------------------------------------
_BULLETPROOF_ASNS: frozenset[str] = frozenset(
    {
        "AS16276",   # OVH  (not inherently bad, but frequently abused)
        "AS44477",   # STARK Industries
        "AS48031",   # PE Ivanov Vitaliy
        "AS202425",  # INT-NETWORK
        "AS58061",   # SWIFT-COOP
        "AS9009",    # M247
        "AS62240",   # Clouvider
        "AS60068",   # Datacamp
        "AS49981",   # WorldStream
    }
)

_FAST_FLUX_THRESHOLD: int = 5
"""Number of distinct A records above which fast-flux is suspected."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str) -> dict:
    """Compute the DNS risk sub-score for *url*.

    Parameters
    ----------
    url : str
        The target URL.  Only the hostname is extracted; the rest of the
        URL is ignored.

    Returns
    -------
    dict
        ``score`` (float in [0, 1]) and ``details`` (per-signal dict).
    """
    hostname = _extract_hostname(url)
    if not hostname:
        logger.warning("Could not extract hostname from URL: %s", url)
        return {"score": 0.5, "details": {"error": "no hostname"}}

    signals: dict[str, float] = {}

    # 1. A-record check
    a_records = _query_a_records(hostname)
    if a_records is None:
        # NXDOMAIN or timeout – highly suspicious
        signals["a_record_missing"] = 1.0
        signals["fast_flux"] = 0.0
        signals["mx_missing"] = 1.0
        signals["bulletproof_hosting"] = 0.0
        score = _clamp(_average(signals))
        logger.info("DNS score for %s: %.4f (NXDOMAIN/timeout)", hostname, score)
        return {"score": score, "details": signals}

    signals["a_record_missing"] = 0.0

    # 2. Fast-flux detection
    if len(a_records) >= _FAST_FLUX_THRESHOLD:
        signals["fast_flux"] = _clamp(len(a_records) / 10.0)
    else:
        signals["fast_flux"] = 0.0

    # 3. MX-record check
    signals["mx_missing"] = 0.0 if _has_mx_records(hostname) else 1.0

    # 4. Bulletproof hosting (stub – full implementation requires an IP-to-ASN
    #    lookup service such as Team Cymru or a local MaxMind DB)
    signals["bulletproof_hosting"] = _check_bulletproof_hosting(a_records)

    score = _clamp(_average(signals))
    logger.info(
        "DNS score for %s: %.4f  signals=%s", hostname, score, signals,
    )
    return {"score": score, "details": signals}


# ---------------------------------------------------------------------------
# DNS helpers
# ---------------------------------------------------------------------------

def _extract_hostname(url: str) -> str | None:
    """Return the hostname from *url*, prepending a scheme if absent."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    return parsed.hostname


def _query_a_records(hostname: str) -> list[str] | None:
    """Return a list of IPv4 A-record addresses, or *None* on failure.

    Failures include ``NXDOMAIN``, ``NoAnswer``, and timeouts.
    """
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = DNS_QUERY_TIMEOUT
        answers = resolver.resolve(hostname, "A")
        return [rdata.to_text() for rdata in answers]
    except dns.resolver.NXDOMAIN:
        logger.debug("NXDOMAIN for %s", hostname)
        return None
    except dns.resolver.NoAnswer:
        logger.debug("NoAnswer (A) for %s", hostname)
        return None
    except dns.exception.Timeout:
        logger.debug("DNS timeout (A) for %s", hostname)
        return None
    except Exception:
        logger.exception("Unexpected DNS error (A) for %s", hostname)
        return None


def _has_mx_records(hostname: str) -> bool:
    """Return *True* if *hostname* publishes at least one MX record."""
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = DNS_QUERY_TIMEOUT
        answers = resolver.resolve(hostname, "MX")
        return len(answers) > 0
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.exception.Timeout):
        return False
    except Exception:
        logger.exception("Unexpected DNS error (MX) for %s", hostname)
        return False


def _check_bulletproof_hosting(a_records: list[str]) -> float:
    """Heuristic bulletproof-hosting check.

    A full implementation would perform an IP → ASN lookup (e.g. via
    Team Cymru DNS or a local MaxMind GeoLite2 ASN database) and compare
    the result against ``_BULLETPROOF_ASNS``.

    This implementation uses a lightweight heuristic: private-range IPs
    (RFC 1918 / loopback) are flagged at 0.5 because they should never
    appear in production DNS and may indicate DNS hijacking or testing
    infrastructure.
    """
    for ip in a_records:
        if (
            ip.startswith("10.")
            or ip.startswith("192.168.")
            or ip.startswith("172.16.")
            or ip.startswith("127.")
        ):
            return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _average(signals: dict[str, float]) -> float:
    """Return the arithmetic mean of all signal values."""
    if not signals:
        return 0.0
    return sum(signals.values()) / len(signals)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))
