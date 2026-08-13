from dataclasses import asdict
from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import Product
from scanner import scan_product
from products import get_product
from grading import grade_product
from cart import (
    add_product,
    remove_product,
    get_cart,
    analyze_cart,
    cart,
)
from recommendations import recommend


# CONSTANTS

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
}

# Protect shared cart access between requests.
_cart_lock = threading.Lock()


# APP

app = FastAPI(
    title="CERES API",
    description="CERES grocery health analysis backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# PRODUCT

def dict_to_product(data: dict) -> Product:
    """Convert a product dictionary into a Product object."""

    if "error" in data:
        raise ValueError(data["error"])

    return Product(
        barcode=str(data.get("barcode", "")),
        name=data.get("name"),
        brand=data.get("brand"),
        nutriscore=data.get("nutriscore"),
        energy_kj=data.get("energy_kj"),
        fat=data.get("fat"),
        saturated_fat=data.get("saturated_fat"),
        carbohydrates=data.get("carbohydrates"),
        sugars=data.get("sugars"),
        fiber=data.get("fiber"),
        protein=data.get("protein"),
        salt=data.get("salt"),
        sodium=data.get("sodium"),
        ingredients=data.get("ingredients"),
        categories=data.get("categories"),
        countries=data.get("countries"),
    )


def product_response(product: Product, grading: dict) -> dict:
    """Convert Product and grading information into JSON-safe data."""

    return {
        "product": asdict(product),
        "grading": grading,
    }


def cart_response() -> list:
    """Convert the cart into JSON-safe dictionaries."""

    return [
        {
            "product": asdict(item["product"]),
            "grading": item["grading"],
        }
        for item in get_cart()
    ]


def _validate_upload(
    file: UploadFile,
    contents: bytes,
) -> str:
    """
    Validate an uploaded file and return a safe file suffix.
    Raises HTTPException when validation fails.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. Maximum size is "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
            ),
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    return suffix


# BASIC

@app.get("/")
def root():
    return {
        "name": "CERES API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# SCANNING

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Receive an image, scan its barcode,
    find the product, and grade it.
    """

    contents = await file.read()
    suffix = _validate_upload(file, contents)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # 1. Scan barcode
        scanned = scan_product(temp_path)

        # 2. Resolve product data
        if "error" not in scanned:
            product_data = scanned

        else:
            barcode = scanned.get("barcode")

            if not barcode:
                raise HTTPException(
                    status_code=404,
                    detail=scanned.get(
                        "error",
                        "Could not identify barcode",
                    ),
                )

            product_data = get_product(barcode)

            if "error" in product_data:
                raise HTTPException(
                    status_code=404,
                    detail="Product not found for the scanned barcode",
                )

        # 3. Convert dictionary into Product
        product = dict_to_product(product_data)

        # 4. Grade product
        grading = grade_product(product)

        return product_response(
            product,
            grading,
        )

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=str(e),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image",
        )

    finally:
        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


# PRODUCT LOOKUP

@app.get("/product/{barcode}")
def product_lookup(barcode: str):
    """Look up a product directly using its barcode."""

    if not barcode.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid barcode format",
        )

    data = get_product(barcode)

    if "error" in data:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    product_object = dict_to_product(data)
    grading = grade_product(product_object)

    return product_response(
        product_object,
        grading,
    )


# CART

@app.get("/cart")
def cart_endpoint():
    """Return all products currently in the cart."""

    with _cart_lock:
        items = cart_response()
        total = len(cart)

    return {
        "items": items,
        "total_items": total,
    }


@app.post("/cart/add/{barcode}")
def add_to_cart(barcode: str):
    """Find, convert, grade, and add a product to the cart."""

    if not barcode.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid barcode format",
        )

    data = get_product(barcode)

    if "error" in data:
        raise HTTPException(
            status_code=404,
            detail="Product not found",
        )

    product_object = dict_to_product(data)
    grading = grade_product(product_object)

    with _cart_lock:
        add_product(
            product_object,
            grading,
        )

        items = cart_response()

    return {
        "success": True,
        "message": "Product added to cart",
        "item": product_response(
            product_object,
            grading,
        ),
        "cart": items,
    }


@app.delete("/cart/{barcode}")
def remove_from_cart(barcode: str):
    """Remove a product from the cart."""

    with _cart_lock:
        removed = remove_product(barcode)

        if not removed:
            raise HTTPException(
                status_code=404,
                detail="Product not found in cart",
            )

        items = cart_response()

    return {
        "success": True,
        "message": "Product removed from cart",
        "cart": items,
    }


@app.delete("/cart")
def clear_cart():
    """Clear the entire cart."""

    with _cart_lock:
        cart.clear()

    return {
        "success": True,
        "message": "Cart cleared",
        "cart": [],
    }


# CART ANALYSIS

@app.get("/cart/analysis")
def cart_analysis():
    """Analyze the current cart."""

    with _cart_lock:
        analysis = analyze_cart()

    analysis["low_rated_items"] = [
        asdict(product)
        for product in analysis["low_rated_items"]
    ]

    analysis["low_confidence_items"] = [
        asdict(product)
        for product in analysis["low_confidence_items"]
    ]

    return analysis


# RECOMMENDATIONS

@app.get("/recommendations")
def recommendations():
    """Generate recommendations based on the current cart."""

    with _cart_lock:
        return recommend()


# RUN SERVER

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
