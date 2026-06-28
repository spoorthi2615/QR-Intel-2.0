"""Stage-2 Graph Intelligence Module for the QRIntel pipeline.

This module maintains an in-memory *threat graph* where:

* **Nodes** represent analysed URLs (identified by their effective
  second-level domain).
* **Edges** connect two nodes whose threat profiles are similar, weighted
  by the `Jaccard similarity coefficient
  <https://en.wikipedia.org/wiki/Jaccard_index>`_ of their signal
  vectors.

Edge weights decay over time using an exponential function:

    Weight_t = Weight_{t-1} × exp(−λ · Δt)

where *λ* is :data:`backend.config.GRAPH_DECAY_LAMBDA` and *Δt* is the
elapsed time in hours since the edge was last reinforced.

The module exposes a ``risk_modifier`` that quantifies how closely the
current URL's profile resembles known-malicious profiles already in the
graph.

Thread Safety
-------------
All mutations to the shared graph dictionary are serialised via a
:class:`threading.Lock`.  For a multi-process deployment (e.g. Gunicorn
with ``--workers > 1``), replace the in-memory dict with Redis or a
similar external store.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from backend.config import GRAPH_DECAY_LAMBDA

logger = logging.getLogger(__name__)

_graph_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThreatNode:
    """A node in the threat graph representing a single domain.

    Attributes
    ----------
    domain : str
        Effective second-level domain (e.g. ``example.com``).
    profile : dict[str, float]
        Mapping of signal names to their [0, 1] values at the time of
        the most recent scan.
    verdict : str
        Most recent verdict (``SAFE``, ``SUSPICIOUS``, ``MALICIOUS``).
    last_seen : float
        Unix timestamp of the most recent scan.
    """

    domain: str
    profile: dict[str, float] = field(default_factory=dict)
    verdict: str = "SAFE"
    last_seen: float = field(default_factory=time.time)


@dataclass
class ThreatEdge:
    """A weighted, undirected edge between two :class:`ThreatNode` instances.

    Attributes
    ----------
    weight : float
        Jaccard similarity when the edge was created or last reinforced.
    created_at : float
        Unix timestamp of edge creation.
    last_reinforced : float
        Unix timestamp of the most recent reinforcement.
    """

    weight: float
    created_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)


# The in-memory graph stores nodes by domain key and edges by a
# frozenset pair of domain keys.
_nodes: dict[str, ThreatNode] = {}
_edges: dict[frozenset[str], ThreatEdge] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(url: str, signal_profile: dict[str, float], verdict: str) -> dict:
    """Update the threat graph and compute a risk modifier for *url*.

    Parameters
    ----------
    url : str
        The URL that was just scanned.
    signal_profile : dict[str, float]
        The four Stage-1 sub-scores keyed by analyser name (e.g.
        ``{"lexical": 0.72, "dns": 0.3, "redirect": 0.1, "ssl": 0.4}``).
    verdict : str
        The Stage-1 verdict before graph adjustment.

    Returns
    -------
    dict
        ``risk_modifier`` (float) – additive adjustment (positive means
        riskier), and ``details`` (dict) with neighbour information.
    """
    domain = _effective_domain(url)
    now = time.time()

    with _graph_lock:
        # Upsert node
        node = _nodes.get(domain)
        if node is None:
            node = ThreatNode(
                domain=domain, profile=signal_profile,
                verdict=verdict, last_seen=now,
            )
            _nodes[domain] = node
        else:
            node.profile = signal_profile
            node.verdict = verdict
            node.last_seen = now

        # Compare against every other node and maintain edges
        malicious_similarity_sum = 0.0
        malicious_neighbours = 0
        neighbour_details: list[dict] = []

        for other_domain, other_node in _nodes.items():
            if other_domain == domain:
                continue

            similarity = jaccard_similarity(signal_profile, other_node.profile)
            edge_key = frozenset({domain, other_domain})

            if similarity > 0.0:
                existing = _edges.get(edge_key)
                if existing is None:
                    _edges[edge_key] = ThreatEdge(
                        weight=similarity, created_at=now, last_reinforced=now,
                    )
                else:
                    existing.weight = similarity
                    existing.last_reinforced = now

            # Accumulate risk from malicious neighbours
            edge = _edges.get(edge_key)
            if edge and other_node.verdict == "MALICIOUS":
                decayed_weight = _time_decay(edge.weight, edge.last_reinforced, now)
                malicious_similarity_sum += decayed_weight
                malicious_neighbours += 1
                neighbour_details.append({
                    "domain": other_domain,
                    "similarity": round(similarity, 4),
                    "decayed_weight": round(decayed_weight, 4),
                })

    # The risk modifier scales with the number and similarity of
    # malicious neighbours, capped at 20 points.
    risk_modifier = min(malicious_similarity_sum * 10.0, 20.0)

    logger.info(
        "Graph analysis for %s: modifier=%.2f malicious_neighbours=%d",
        domain, risk_modifier, malicious_neighbours,
    )
    return {
        "risk_modifier": risk_modifier,
        "details": {
            "domain": domain,
            "malicious_neighbours": malicious_neighbours,
            "neighbours": neighbour_details,
            "graph_size": len(_nodes),
        },
    }


def jaccard_similarity(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    threshold: float = 0.5,
) -> float:
    """Compute the Jaccard similarity between two threat profiles.

    Each profile is a mapping of signal names to [0, 1] values.  A
    signal is considered *active* if its value ≥ *threshold*.

    The Jaccard index is:

        J(A, B) = |A ∩ B| / |A ∪ B|

    where A and B are the sets of active signals.

    Parameters
    ----------
    profile_a, profile_b : dict[str, float]
        Signal-name → value mappings.
    threshold : float
        Activation threshold (default 0.5).

    Returns
    -------
    float
        Jaccard similarity in [0.0, 1.0].
    """
    set_a = {k for k, v in profile_a.items() if v >= threshold}
    set_b = {k for k, v in profile_b.items() if v >= threshold}

    if not set_a and not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def clear_graph() -> None:
    """Remove all nodes and edges from the threat graph.

    Intended for testing and administrative resets.
    """
    with _graph_lock:
        _nodes.clear()
        _edges.clear()
    logger.info("Threat graph cleared.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_decay(weight: float, last_reinforced: float, now: float) -> float:
    """Apply exponential time decay to an edge weight.

    Formula:
        Weight_t = Weight_{t-1} × exp(−λ × Δt)

    where Δt is measured in **hours**.

    Parameters
    ----------
    weight : float
        The edge weight at ``last_reinforced``.
    last_reinforced : float
        Unix timestamp of the last reinforcement.
    now : float
        Current Unix timestamp.

    Returns
    -------
    float
        The decayed weight (always ≥ 0).
    """
    delta_hours = (now - last_reinforced) / 3600.0
    return weight * math.exp(-GRAPH_DECAY_LAMBDA * delta_hours)


def _effective_domain(url: str) -> str:
    """Extract the effective second-level domain from *url*."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname
