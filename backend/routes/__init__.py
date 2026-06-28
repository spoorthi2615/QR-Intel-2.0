"""Flask route blueprints for the QRIntel REST API.

Blueprints
----------
scan_bp
    POST /api/scan – analyse a QR image or raw URL.
health_bp
    GET  /api/health – lightweight liveness / readiness probe.
"""
