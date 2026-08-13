# scanner.py
# Decodes a barcode from an image.
# Product lookup itself happens in products.py (Open Food Facts).

import cv2
import zxingcpp


def scan_product(image_path):
    """
    Given an image path, decode the barcode in it.

    Returns:
        {"barcode": "..."} on success
        {"error": "..."} on failure
    """

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image file"}

    results = zxingcpp.read_barcodes(image)
    if not results:
        return {"error": "No barcode detected"}

    return {"barcode": results[0].text}
