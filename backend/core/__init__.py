"""Core analysis modules for the QRIntel risk-scoring pipeline.

Submodules
----------
lexical_analyzer
    Shannon-entropy and heuristic scoring of URL strings.
dns_analyzer
    DNS record inspection (A / MX / fast-flux detection).
redirect_analyzer
    HTTP redirect-chain follower and anomaly scorer.
ssl_analyzer
    TLS certificate age, issuer-trust, and validity checks.
brand_analyzer
    Levenshtein-distance brand-impersonation detector.
graph_analyzer
    Jaccard-similarity threat-graph with time-decay edges.
risk_engine
    Two-stage orchestrator that fuses all sub-scores into a verdict.
qr_decoder
    pyzbar-backed QR code image decoder.
"""
