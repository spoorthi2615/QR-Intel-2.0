"""Scan route for the QRIntel REST API.

Provides ``POST /api/scan`` which accepts either:

* A QR-code image (``multipart/form-data`` with field name ``qr_image``), or
* A raw URL string (JSON body ``{"url": "https://example.com"}``).

The endpoint decodes the QR image (if provided), feeds the extracted URL
into :func:`backend.core.risk_engine.analyze`, and returns the full
verdict, risk score, and audit log as JSON.
"""

from __future__ import annotations

import io
import logging

from flask import Blueprint, jsonify, request

from backend.core.qr_decoder import decode_qr_single
from backend.core.risk_engine import analyze

logger = logging.getLogger(__name__)

scan_bp = Blueprint("scan", __name__)


@scan_bp.route("/api/scan", methods=["POST"])
def scan():
    """Analyse a QR code image or raw URL for phishing risk.

    **Option A – QR image upload** (multipart/form-data)::

        POST /api/scan
        Content-Type: multipart/form-data
        Field: qr_image=<image file>

    **Option B – raw URL** (JSON body)::

        POST /api/scan
        Content-Type: application/json
        {"url": "https://example.com"}

    **Success response** (200 OK)::

        {
            "scan_id": "...",
            "url": "https://example.com",
            "verdict": "SAFE",
            "risk_score": 12.34,
            "empirical_score": 15.60,
            "audit_log": { ... }
        }

    **Error responses**:

    * 400 – no URL or image provided, or QR decoding failed.
    * 500 – unexpected internal error.

    Returns
    -------
    flask.Response
        JSON payload with the analysis results.
    """
    url = _extract_url_from_request()

    if url is None:
        return jsonify({
            "error": "No URL provided. Submit a QR image (field 'qr_image') "
                     "or a JSON body with a 'url' key.",
        }), 400

    if not url.strip():
        return jsonify({"error": "The supplied URL is empty."}), 400

    try:
        result = analyze(url)
    except Exception:
        logger.exception("Risk analysis failed for URL: %s", url)
        return jsonify({
            "error": "Internal analysis error. Please try again later.",
        }), 500

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_url_from_request() -> str | None:
    """Extract a URL from the incoming Flask request.

    Checks for a QR image upload first; falls back to a JSON ``url``
    field; finally checks form-encoded ``url``.

    Returns
    -------
    str | None
        The extracted URL string, or *None* if nothing was found.
    """
    # 1. QR image upload
    if "qr_image" in request.files:
        image_file = request.files["qr_image"]
        if image_file.filename:
            logger.info("Received QR image upload: %s", image_file.filename)
            try:
                stream = io.BytesIO(image_file.read())
                decoded_url = decode_qr_single(stream)
            except Exception:
                logger.exception("QR decoding failed for uploaded image.")
                return None

            if decoded_url:
                logger.info("Decoded URL from QR image: %s", decoded_url)
                return decoded_url

            logger.warning("No QR code found in the uploaded image.")
            return None

    # 2. JSON body
    json_body = request.get_json(silent=True)
    if json_body and isinstance(json_body.get("url"), str):
        return json_body["url"]

    # 3. Form-encoded body
    form_url = request.form.get("url")
    if form_url:
        return form_url

    return None
