# Given an image or barcode, find product
import cv2
from pyzbar.pyzbar import decode
import json

def scan_product(image_path):
    """
    Given an image path or frame, extracts the barcode 
    and returns matching product details.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image file"}

    # Decode barcode/QR code from image
    barcodes = decode(image)
    if not barcodes:
        return {"error": "No barcode detected"}

    # Get barcode string value
    barcode_data = barcodes[0].data.decode('utf-8')

    # Load products database
    try:
        with open('data/products.json', 'r') as f:
            products = json.load(f)
    except FileNotFoundError:
        return {"error": "Products file not found"}

    # Search product by barcode
    for product in products:
        if product.get("barcode") == barcode_data:
            return product

    return {"error": "Product not found", "barcode": barcode_data}
