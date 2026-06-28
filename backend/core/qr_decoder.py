"""QR Code Decoder for the QRIntel pipeline.

This module accepts an image file (PNG, JPG, BMP, GIF) containing one or
more QR codes and returns the decoded text payload(s).  Decoding is
performed by `pyzbar <https://pypi.org/project/pyzbar/>`_, a Python
wrapper around the ZBar barcode-reading library.

Typical usage::

    from backend.core.qr_decoder import decode_qr

    urls = decode_qr(Path("scan.png"))
    for url in urls:
        print(url)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import BinaryIO

from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol

logger = logging.getLogger(__name__)

# We only care about QR codes, not other barcode symbologies.
_SYMBOL_TYPES = [ZBarSymbol.QRCODE]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_qr(source: str | Path | BinaryIO) -> list[str]:
    """Decode QR codes from an image and return their text payloads.

    Parameters
    ----------
    source : str | Path | BinaryIO
        File path or file-like object pointing to the image.  Supported
        formats are everything that Pillow can open (PNG, JPEG, BMP,
        GIF, TIFF, WebP, etc.).

    Returns
    -------
    list[str]
        A list of decoded text strings, one per QR code found in the
        image.  Returns an empty list if no QR codes are detected.

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If the image cannot be opened or decoded.
    """
    image = _load_image(source)
    results = pyzbar_decode(image, symbols=_SYMBOL_TYPES)

    payloads: list[str] = []
    for result in results:
        try:
            text = result.data.decode("utf-8")
            payloads.append(text)
            logger.info("Decoded QR payload: %s", text)
        except UnicodeDecodeError:
            logger.warning(
                "QR code contained non-UTF-8 data (%d bytes); skipping.",
                len(result.data),
            )

    if not payloads:
        logger.info("No QR codes detected in the supplied image.")

    return payloads


def decode_qr_single(source: str | Path | BinaryIO) -> str | None:
    """Convenience wrapper that returns only the *first* decoded payload.

    Parameters
    ----------
    source : str | Path | BinaryIO
        Same as :func:`decode_qr`.

    Returns
    -------
    str | None
        The first decoded payload, or *None* if no QR code was found.
    """
    payloads = decode_qr(source)
    return payloads[0] if payloads else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(source: str | Path | BinaryIO) -> Image.Image:
    """Open *source* as a Pillow :class:`~PIL.Image.Image`.

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist on disk.
    ValueError
        If the file cannot be interpreted as an image.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        try:
            img = Image.open(path)
            img.load()  # force full read so pyzbar can access pixel data
            return img
        except Exception as exc:
            raise ValueError(f"Could not open image at {path}: {exc}") from exc

    # File-like object (e.g. Flask's FileStorage.stream)
    try:
        img = Image.open(source)
        img.load()
        return img
    except Exception as exc:
        raise ValueError(f"Could not decode image from stream: {exc}") from exc
