"""Global configuration constants for the QRIntel risk-scoring engine.

This module centralises every tunable parameter so that the rest of the
codebase never hard-codes magic numbers.  The Stage-1 weights were obtained
via Optuna Bayesian hyper-parameter optimisation against a labelled corpus
of ~12 000 phishing and benign URLs.

Stage-1 Empirical Score
-----------------------
    EmpiricalScore = 100 × (w_l·s_l + w_d·s_d + w_r·s_r + w_ssl·s_ssl)

Stage-2 Overrides
-----------------
    * Canonical penalty   – applied when the domain matches a known safe list.
    * Brand impersonation – applied when Levenshtein distance to a known brand
      falls in [1, 3].
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Stage-1 optimised weights  (sum ≈ 1.0)
# ---------------------------------------------------------------------------
WEIGHT_LEXICAL: float = 0.5353
"""Weight for the lexical intelligence sub-score (s_l)."""

WEIGHT_DNS: float = 0.1053
"""Weight for the DNS intelligence sub-score (s_d)."""

WEIGHT_REDIRECT: float = 0.1749
"""Weight for the redirect-chain sub-score (s_r)."""

WEIGHT_SSL: float = 0.1845
"""Weight for the SSL certificate sub-score (s_ssl)."""

# ---------------------------------------------------------------------------
# Stage-1 threshold
# ---------------------------------------------------------------------------
THRESHOLD: float = 30.06
"""Empirical score above which a URL is classified as suspicious or malicious.

Scores in [THRESHOLD, 70) → SUSPICIOUS; scores ≥ 70 → MALICIOUS.
"""

# ---------------------------------------------------------------------------
# Stage-2 penalties / bonuses
# ---------------------------------------------------------------------------
CANONICAL_PENALTY: float = -40.0
"""Score adjustment applied when the domain belongs to ``KNOWN_CANONICAL_DOMAINS``.

A large negative penalty pushes the empirical score downward, reflecting
the very low prior probability that a canonical domain is phishing.
"""

BRAND_IMPERSONATION_PENALTY: float = 90.0
"""Score adjustment applied when the domain's Levenshtein distance to a
known brand is in [1, 3], indicating likely typosquatting.

A positive penalty pushes the score upward, significantly increasing the
chance of a MALICIOUS verdict.
"""

# ---------------------------------------------------------------------------
# Known canonical (safe) domains
# ---------------------------------------------------------------------------
KNOWN_CANONICAL_DOMAINS: frozenset[str] = frozenset(
    {
        "google.com",
        "www.google.com",
        "facebook.com",
        "www.facebook.com",
        "amazon.com",
        "www.amazon.com",
        "apple.com",
        "www.apple.com",
        "microsoft.com",
        "www.microsoft.com",
        "github.com",
        "www.github.com",
        "linkedin.com",
        "www.linkedin.com",
        "twitter.com",
        "www.twitter.com",
        "x.com",
        "www.x.com",
        "youtube.com",
        "www.youtube.com",
        "instagram.com",
        "www.instagram.com",
        "netflix.com",
        "www.netflix.com",
        "paypal.com",
        "www.paypal.com",
        "wikipedia.org",
        "www.wikipedia.org",
        "reddit.com",
        "www.reddit.com",
        "stackoverflow.com",
        "www.stackoverflow.com",
        "dropbox.com",
        "www.dropbox.com",
        "zoom.us",
        "www.zoom.us",
        "slack.com",
        "www.slack.com",
        "whatsapp.com",
        "www.whatsapp.com",
    }
)
"""Domains considered trustworthy.  Membership triggers ``CANONICAL_PENALTY``."""

# ---------------------------------------------------------------------------
# Known brand names (for Levenshtein impersonation check)
# ---------------------------------------------------------------------------
KNOWN_BRANDS: frozenset[str] = frozenset(
    {
        "google",
        "facebook",
        "amazon",
        "apple",
        "microsoft",
        "github",
        "linkedin",
        "twitter",
        "youtube",
        "instagram",
        "netflix",
        "paypal",
        "wikipedia",
        "reddit",
        "dropbox",
        "zoom",
        "slack",
        "whatsapp",
        "spotify",
        "adobe",
        "chase",
        "wellsfargo",
        "bankofamerica",
        "citibank",
        "americanexpress",
    }
)
"""Brand names used by :mod:`backend.core.brand_analyzer` to detect
typosquatting via Levenshtein distance."""

# ---------------------------------------------------------------------------
# Analyser-level tuning
# ---------------------------------------------------------------------------
LEXICAL_SUSPICIOUS_KEYWORDS: frozenset[str] = frozenset(
    {
        "login",
        "verify",
        "secure",
        "account",
        "update",
        "banking",
        "confirm",
        "password",
        "signin",
        "suspend",
        "alert",
        "wallet",
        "authenticate",
    }
)
"""Keywords whose presence in a URL raises the lexical sub-score."""

MAX_REDIRECT_DEPTH: int = 10
"""Hard cap on redirects the redirect analyser will follow."""

REDIRECT_SUSPICIOUS_THRESHOLD: int = 3
"""Redirect chains longer than this count are flagged as suspicious."""

DNS_QUERY_TIMEOUT: float = 4.0
"""Seconds to wait for each DNS query before treating it as a timeout."""

SSL_CONNECT_TIMEOUT: float = 5.0
"""Seconds to wait when opening a TLS connection for certificate extraction."""

SSL_YOUNG_CERT_DAYS: int = 30
"""Certificates younger than this many days raise the SSL sub-score."""

LEVENSHTEIN_MIN_DISTANCE: int = 1
"""Minimum edit distance (inclusive) for brand-impersonation flagging."""

LEVENSHTEIN_MAX_DISTANCE: int = 3
"""Maximum edit distance (inclusive) for brand-impersonation flagging."""

GRAPH_DECAY_LAMBDA: float = 0.05
"""Exponential decay rate (λ) for time-weighted threat-graph edges."""
