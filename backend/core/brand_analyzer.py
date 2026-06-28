"""Stage-2 Brand Impersonation Detector for the QRIntel pipeline.

This module computes the `Levenshtein edit distance
<https://en.wikipedia.org/wiki/Levenshtein_distance>`_ between the
second-level domain label of the target URL and every entry in the
known-brands list (:data:`backend.config.KNOWN_BRANDS`).

If the minimum distance falls in the range
[``LEVENSHTEIN_MIN_DISTANCE``, ``LEVENSHTEIN_MAX_DISTANCE``] the domain
is flagged as a likely typosquatting attempt, and the
:data:`backend.config.BRAND_IMPERSONATION_PENALTY` is returned as an
additive modifier to the empirical score.

Levenshtein Distance
--------------------
The Levenshtein distance between two strings *a* and *b* is the minimum
number of single-character edits (insertions, deletions, substitutions)
required to transform *a* into *b*.  It is computed via the classic
Wagner–Fischer dynamic-programming algorithm in O(|a|·|b|) time.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from backend.config import (
    BRAND_IMPERSONATION_PENALTY,
    KNOWN_BRANDS,
    LEVENSHTEIN_MAX_DISTANCE,
    LEVENSHTEIN_MIN_DISTANCE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str) -> dict:
    """Check whether *url* impersonates a known brand.

    Parameters
    ----------
    url : str
        The URL to inspect.

    Returns
    -------
    dict
        ``penalty`` (float) – additive score modifier (0 or
        ``BRAND_IMPERSONATION_PENALTY``), ``details`` (dict) containing
        the closest brand, its edit distance, and the extracted domain
        label.
    """
    domain_label = _extract_domain_label(url)
    if not domain_label:
        return {"penalty": 0.0, "details": {"error": "no domain label"}}

    closest_brand, min_distance = _find_closest_brand(domain_label)

    is_impersonation = (
        LEVENSHTEIN_MIN_DISTANCE <= min_distance <= LEVENSHTEIN_MAX_DISTANCE
    )

    penalty = BRAND_IMPERSONATION_PENALTY if is_impersonation else 0.0

    details = {
        "domain_label": domain_label,
        "closest_brand": closest_brand,
        "levenshtein_distance": min_distance,
        "is_impersonation": is_impersonation,
    }

    logger.info(
        "Brand analysis for %s: closest=%s dist=%d impersonation=%s penalty=%.1f",
        domain_label, closest_brand, min_distance, is_impersonation, penalty,
    )
    return {"penalty": penalty, "details": details}


# ---------------------------------------------------------------------------
# Levenshtein distance (Wagner–Fischer algorithm)
# ---------------------------------------------------------------------------

def levenshtein_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between strings *a* and *b*.

    Uses the standard Wagner–Fischer dynamic-programming approach with
    two-row space optimisation (O(min(|a|, |b|)) memory).

    Parameters
    ----------
    a, b : str
        The two strings to compare.

    Returns
    -------
    int
        The minimum number of single-character edits to transform *a*
        into *b*.

    Examples
    --------
    >>> levenshtein_distance("kitten", "sitting")
    3
    >>> levenshtein_distance("google", "go0gle")
    1
    """
    if len(a) < len(b):
        return levenshtein_distance(b, a)

    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a):
        current_row = [i + 1]
        for j, char_b in enumerate(b):
            # Cost is 0 if characters match, 1 otherwise.
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if char_a == char_b else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_closest_brand(domain_label: str) -> tuple[str, int]:
    """Return ``(brand, distance)`` for the known brand closest to *domain_label*.

    If ``KNOWN_BRANDS`` is empty, returns ``("", len(domain_label))``.
    """
    best_brand = ""
    best_distance = len(domain_label) + 1  # worse than any real distance

    label_lower = domain_label.lower()
    for brand in KNOWN_BRANDS:
        dist = levenshtein_distance(label_lower, brand.lower())
        if dist < best_distance:
            best_distance = dist
            best_brand = brand
        if dist == 0:
            break  # exact match – no need to keep searching

    return best_brand, best_distance


def _extract_domain_label(url: str) -> str:
    """Extract the second-level domain label from *url*.

    For ``https://www.secure-go0gle.com/path`` this returns ``secure-go0gle``.
    For ``https://login.paypa1.com/`` this returns ``paypa1``.
    """
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")

    # Remove common "www" prefix
    if parts and parts[0].lower() == "www":
        parts = parts[1:]

    if len(parts) >= 2:
        return parts[-2]  # second-level label
    if parts:
        return parts[0]
    return ""
