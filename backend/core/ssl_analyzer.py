"""SSL Certificate Intelligence Module (s_ssl) for the QRIntel pipeline.

This module opens a TLS connection to the target host, extracts the
server certificate, and converts its properties into a normalised risk
sub-score in [0.0, 1.0].

Scoring Methodology
-------------------
Four signals are combined with equal weight:

1. **Self-signed certificate** – if the issuer CN equals the subject CN,
   the certificate is self-signed and scores 1.0.

2. **Certificate age** – certificates younger than
   :data:`backend.config.SSL_YOUNG_CERT_DAYS` score proportionally
   higher.  Freshly minted certificates are common in throwaway phishing
   infrastructure.

3. **Validity-period anomaly** – very short validity windows (< 90 days)
   or absurdly long ones (> 825 days, violating CA/Browser Forum rules)
   are penalised.

4. **Connection failure** – inability to complete the TLS handshake at
   all is itself a risk signal (score 0.8).

All network operations use a hard timeout
(:data:`backend.config.SSL_CONNECT_TIMEOUT`) so that unreachable hosts
never stall the pipeline.
"""

from __future__ import annotations

import datetime
import logging
import re
import socket
import ssl
from urllib.parse import urlparse

from backend.config import SSL_CONNECT_TIMEOUT, SSL_YOUNG_CERT_DAYS

logger = logging.getLogger(__name__)

_SIGNAL_WEIGHTS: dict[str, float] = {
    "self_signed":    0.30,
    "cert_age":       0.30,
    "validity_span":  0.20,
    "issuer_trust":   0.20,
}

# Issuers considered broadly trustworthy (case-insensitive substring match)
_TRUSTED_ISSUERS: tuple[str, ...] = (
    "let's encrypt",
    "digicert",
    "comodo",
    "globalsign",
    "sectigo",
    "godaddy",
    "entrust",
    "geotrust",
    "thawte",
    "verisign",
    "amazon",
    "google trust services",
    "microsoft",
    "cloudflare",
    "baltimore",
    "isrg",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str) -> dict:
    """Compute the SSL/TLS risk sub-score for *url*.

    Parameters
    ----------
    url : str
        The URL whose host's certificate is inspected.

    Returns
    -------
    dict
        ``score`` (float in [0, 1]) and ``details`` (per-signal dict).
    """
    hostname, port = _extract_host_port(url)
    if not hostname:
        logger.warning("Could not extract hostname from URL: %s", url)
        return {"score": 0.5, "details": {"error": "no hostname"}}

    cert = _fetch_certificate(hostname, port)
    if cert is None:
        logger.info("TLS handshake failed for %s:%d", hostname, port)
        return {
            "score": 0.8,
            "details": {
                "self_signed": 0.0,
                "cert_age": 0.0,
                "validity_span": 0.0,
                "issuer_trust": 0.0,
                "connection_failed": True,
            },
        }

    signals = _evaluate_certificate(cert)
    score = _clamp(sum(
        _SIGNAL_WEIGHTS[name] * signals[name] for name in _SIGNAL_WEIGHTS
    ))

    logger.info(
        "SSL score for %s:%d: %.4f  signals=%s", hostname, port, score, signals,
    )
    return {"score": score, "details": signals}


# ---------------------------------------------------------------------------
# Certificate extraction
# ---------------------------------------------------------------------------

def _extract_host_port(url: str) -> tuple[str | None, int]:
    """Return ``(hostname, port)`` from *url*, defaulting to port 443."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or 443
    return hostname, port


def _fetch_certificate(hostname: str, port: int) -> dict | None:
    """Open a TLS connection and return the peer certificate as a dict.

    Returns *None* if the handshake fails for any reason (timeout,
    connection refused, certificate verification error, etc.).
    """
    context = ssl.create_default_context()
    # We intentionally disable verification so that we can *inspect*
    # even invalid / self-signed certificates without raising.
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    try:
        with socket.create_connection(
            (hostname, port), timeout=SSL_CONNECT_TIMEOUT
        ) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as tls:
                return tls.getpeercert(binary_form=False)
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError):
        return None
    except Exception:
        logger.exception(
            "Unexpected error fetching certificate for %s:%d", hostname, port,
        )
        return None


# ---------------------------------------------------------------------------
# Certificate evaluation
# ---------------------------------------------------------------------------

def _evaluate_certificate(cert: dict) -> dict[str, float]:
    """Convert a certificate dict into four [0, 1] risk signals."""
    subject_cn = _get_cn(cert.get("subject", ()))
    issuer_cn = _get_cn(cert.get("issuer", ()))
    issuer_org = _get_org(cert.get("issuer", ()))

    not_before = _parse_cert_date(cert.get("notBefore", ""))
    not_after = _parse_cert_date(cert.get("notAfter", ""))
    now = datetime.datetime.now(tz=datetime.timezone.utc)

    # 1. Self-signed check
    is_self_signed = (
        subject_cn.lower() == issuer_cn.lower() if subject_cn and issuer_cn else False
    )
    self_signed_signal = 1.0 if is_self_signed else 0.0

    # 2. Certificate age (days since issuance)
    cert_age_signal = 0.0
    if not_before:
        age_days = (now - not_before).days
        if age_days < SSL_YOUNG_CERT_DAYS:
            cert_age_signal = _clamp(1.0 - (age_days / SSL_YOUNG_CERT_DAYS))

    # 3. Validity span anomaly
    validity_span_signal = 0.0
    if not_before and not_after:
        span_days = (not_after - not_before).days
        if span_days < 90:
            validity_span_signal = _clamp(1.0 - (span_days / 90.0))
        elif span_days > 825:
            validity_span_signal = _clamp((span_days - 825) / 825.0)

    # 4. Issuer trust
    issuer_trust_signal = _issuer_trust_signal(issuer_cn, issuer_org)

    return {
        "self_signed":   self_signed_signal,
        "cert_age":      cert_age_signal,
        "validity_span": validity_span_signal,
        "issuer_trust":  issuer_trust_signal,
    }


def _issuer_trust_signal(issuer_cn: str, issuer_org: str) -> float:
    """Return 0.0 if the issuer is a well-known CA, else 1.0.

    A case-insensitive substring match against ``_TRUSTED_ISSUERS`` is
    used for both the issuer Common Name and Organisation fields.
    """
    combined = f"{issuer_cn} {issuer_org}".lower()
    for trusted in _TRUSTED_ISSUERS:
        if trusted in combined:
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cn(rdns: tuple) -> str:
    """Extract the Common Name (CN) from an RDN sequence."""
    for rdn in rdns:
        for attr_type, attr_value in rdn:
            if attr_type == "commonName":
                return attr_value
    return ""


def _get_org(rdns: tuple) -> str:
    """Extract the Organization (O) from an RDN sequence."""
    for rdn in rdns:
        for attr_type, attr_value in rdn:
            if attr_type == "organizationName":
                return attr_value
    return ""


def _parse_cert_date(date_str: str) -> datetime.datetime | None:
    """Parse the OpenSSL date format ``'Mon DD HH:MM:SS YYYY GMT'``.

    Returns a timezone-aware UTC datetime, or *None* on failure.
    """
    if not date_str:
        return None
    try:
        dt = datetime.datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")
        return dt.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        logger.debug("Could not parse certificate date: %s", date_str)
        return None


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))
