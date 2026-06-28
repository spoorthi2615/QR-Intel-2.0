"""Flask application factory and entry point for the QRIntel API server.

This module creates the Flask application, configures CORS, registers
all route blueprints, and sets up structured logging.

Running the server
------------------
From the project root::

    python -m backend.app          # development
    gunicorn backend.app:app       # production (Linux / macOS)
    waitress-serve backend.app:app # production (Windows)

The server binds to ``0.0.0.0:5000`` by default.
"""

from __future__ import annotations

import logging
import sys

from flask import Flask
from flask_cors import CORS

from backend.routes.health import health_bp
from backend.routes.scan import scan_bp


def create_app() -> Flask:
    """Application factory that assembles and returns a configured Flask app.

    This follows the `Flask application factory pattern
    <https://flask.palletsprojects.com/en/latest/patterns/appfactories/>`_
    so that tests can create isolated app instances.

    Returns
    -------
    Flask
        A fully configured Flask application ready to serve requests.
    """
    application = Flask(__name__)

    # ---- CORS ----------------------------------------------------------
    # Allow all origins during development.  In production, restrict this
    # to the frontend's domain via the CORS_ORIGINS environment variable.
    CORS(application, resources={r"/api/*": {"origins": "*"}})

    # ---- Blueprints ----------------------------------------------------
    application.register_blueprint(health_bp)
    application.register_blueprint(scan_bp)

    # ---- Logging -------------------------------------------------------
    _configure_logging()

    application.logger.info("QRIntel API server initialised.")

    return application


def _configure_logging() -> None:
    """Set up structured, human-readable logging to *stderr*.

    Log level defaults to ``INFO``.  The format includes the timestamp,
    logger name, level, and message—enough for quick triage without
    overwhelming the console.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if create_app() is called more than once.
    if not root_logger.handlers:
        root_logger.addHandler(handler)


# Module-level app instance for ``flask run`` and WSGI servers.
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
