"""Health-check route for the QRIntel REST API.

Provides a lightweight ``GET /api/health`` endpoint suitable for
container orchestrators (Kubernetes liveness/readiness probes), load
balancers, and uptime monitors.
"""

from __future__ import annotations

import datetime
import platform

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """Return the current system health status.

    **Request**::

        GET /api/health

    **Response** (200 OK)::

        {
            "status": "healthy",
            "service": "QRIntel",
            "version": "2.0.0",
            "timestamp": "2026-06-28T14:00:00+00:00",
            "platform": "Windows-11-...",
            "python": "3.12.0"
        }

    Returns
    -------
    flask.Response
        JSON payload with status information and a 200 status code.
    """
    payload = {
        "status": "healthy",
        "service": "QRIntel",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    return jsonify(payload), 200
