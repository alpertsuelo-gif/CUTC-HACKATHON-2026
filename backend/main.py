# main.py
#
# CERES backend API.
#
# Main integration layer that connects:
#   1. scanner.py
#   2. products.py
#   3. models.py
#   4. grading.py
#   5. cart.py
#   6. recommendations.py
#   7. recommendations_ai.py
#
# The frontend communicates with CERES through the API endpoints
# defined in this file.
#
# Main flow:
#   Frontend
#       ↓
#   main.py
#       ↓
#   scanner / products / grading / cart / recommendations
#
# IMPORTANT:
# main.py is responsible for connecting the modules together.
# Individual modules should handle their own specific logic
# rather than duplicating that logic inside this file.


from dataclasses import asdict
from pathlib import Path
import tempfile
import threading

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

from recommendations_ai import recommend_for_cart
from recommendations import recommend


# ============================================================
# CONSTANTS
#
# Configuration values used throughout the API.
#
# MAX_UPLOAD_SIZE_BYTES prevents excessively large image
# uploads from being processed.
#
# ALLOWED_EXTENSIONS limits barcode scanning to supported
# image file types.
# ============================================================

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
}


# Lock used to protect shared cart operations.
#
# The lock is only held while reading or modifying the cart.
# Slow operations such as AI recommendations happen after
# the lock has been released.

_cart_lock = threading.Lock()


# ============================================================
# APP
#
# Creates the FastAPI application and configures CORS.
#
# The frontend communicates with this application through the
# HTTP endpoints defined below.
#
# CORS is enabled so the frontend can communicate with the
# backend even when they are running on different origins.
# ============================================================

app = FastAPI(
    title="CERES API",
    description="CERES grocery health analysis backend",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ERROR HANDLING
#
# Provides a consistent response for unexpected backend errors.
#
# Expected errors should still use HTTPException inside their
# respective endpoints.
#
# Unexpected errors are caught here and returned as a generic
# 500 response instead of exposing internal implementation
# details to the frontend.
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(
        f"[ERROR] {request.method} {request.url}: {exc}"
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": "An unexpected error occurred.",
        },
    )


# ============================================================
# HELPERS
#
# Utility functions used by multiple API endpoints.
#
# These functions handle conversion between the dictionaries
# used by products.py and the Product objects used by the rest
# of the CERES backend.
#
# They also convert Product objects back into dictionaries so
# FastAPI can safely return them as JSON.
# ============================================================

def dict_to_product(data: dict) -> Product:
    """
    Convert a product dictionary into a Product object.
    """

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


def product_response(
    product: Product,
    grading: dict
) -> dict:
    """
    Convert Product and grading information into JSON-safe data.
    """

    return {
        "product": asdict(product),
        "grading": grading,
    }


def cart_response() -> list:
    """
    Convert the cart into JSON-safe dictionaries.
    """

    return [
        {
            "product": asdict(item["product"]),
            "grading": item["grading"],
        }
        for item in get_cart()
    ]


def _validate_upload(
    file: UploadFile,
    contents: bytes
) -> str:
    """
    Validate an uploaded image.

    Checks:
        1. A file was provided.
        2. The file is not empty.
        3. The file does not exceed the size limit.
        4. The file extension is supported.

    Returns:
        The validated file extension.
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
                f"File too large. Maximum size is "
                f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB"
            ),
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    return suffix


# ============================================================
# BASIC
#
# Basic endpoints used to verify that the CERES backend is
# running correctly.
#
# "/" provides basic API information.
#
# "/health" provides a simple health check that can be used
# by the frontend or deployment environment.
# ============================================================

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


# ============================================================
# SCANNING
#
# Receives an image from the frontend and processes it through
# the barcode scanning pipeline.
#
# Pipeline:
#   Image
#       ↓
#   scanner.py
#       ↓
#   Barcode
#       ↓
#   products.py
#       ↓
#   Product dictionary
#       ↓
#   models.py
#       ↓
#   Product object
#       ↓
#   grading.py
#       ↓
#   Product + grade
#
# The uploaded image is temporarily stored on the server
# because scanner.py expects an image file path.
#
# The temporary file is deleted after processing.
# ============================================================

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    Receive an image, scan its barcode,
    find the product, and grade it.
    """

    contents = await file.read()

    suffix = _validate_upload(
        file,
        contents,
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # Scan the barcode from the uploaded image.

        scanned = scan_product(temp_path)

        if "error" in scanned:
            raise HTTPException(
                status_code=404,
                detail=scanned["error"],
            )

        barcode = scanned.get("barcode")

        if not barcode:
            raise HTTPException(
                status_code=404,
                detail="Could not identify barcode",
            )

        # Retrieve product information using the barcode.

        product_data = get_product(barcode)

        if "error" in product_data:
            raise HTTPException(
                status_code=404,
                detail=product_data.get(
                    "error",
                    "Product not found",
                ),
            )

        # Convert dictionary into Product object.

        product = dict_to_product(
            product_data
        )

        # Grade the product.

        grading = grade_product(
            product
        )

        return product_response(
            product,
            grading,
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    except Exception as error:

        print(
            f"[SCAN] Error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the image"
            ),
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


# ============================================================
# PRODUCT LOOKUP
#
# Allows the frontend to request a product directly using
# its barcode without uploading an image.
#
# Pipeline:
#   Barcode
#       ↓
#   products.py
#       ↓
#   Product dictionary
#       ↓
#   models.py
#       ↓
#   Product object
#       ↓
#   grading.py
#       ↓
#   Product + grade
# ============================================================

@app.get("/product/{barcode}")
def product(barcode: str):
    """
    Look up a product directly using its barcode.
    """

    if not barcode.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid barcode format",
        )

    data = get_product(
        barcode
    )

    if "error" in data:
        raise HTTPException(
            status_code=404,
            detail=data.get(
                "error",
                "Product not found",
            ),
        )

    product_object = dict_to_product(
        data
    )

    grading = grade_product(
        product_object
    )

    return product_response(
        product_object,
        grading,
    )


# ============================================================
# CART
#
# Provides endpoints for managing the user's current grocery
# cart.
#
# cart.py owns the actual cart logic.
# main.py only connects that logic to the frontend.
#
# Supported operations:
#   1. View cart
#   2. Add product
#   3. Remove product
#   4. Clear cart
#
# Product flow when adding:
#   Barcode
#       ↓
#   products.py
#       ↓
#   Product
#       ↓
#   grading.py
#       ↓
#   cart.py
# ============================================================

@app.get("/cart")
def cart_endpoint():
    """
    Return all products currently in the cart.
    """

    with _cart_lock:

        items = cart_response()
        total = len(cart)

    return {
        "items": items,
        "total_items": total,
    }


@app.post("/cart/add/{barcode}")
def add_to_cart(barcode: str):
    """
    Find, convert, grade, and add a product
    to the cart.
    """

    if not barcode.isdigit():
        raise HTTPException(
            status_code=400,
            detail="Invalid barcode format",
        )

    data = get_product(
        barcode
    )

    if "error" in data:
        raise HTTPException(
            status_code=404,
            detail=data.get(
                "error",
                "Product not found",
            ),
        )

    product_object = dict_to_product(
        data
    )

    grading = grade_product(
        product_object
    )

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
    """
    Remove a product from the cart.
    """

    with _cart_lock:

        removed = remove_product(
            barcode
        )

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
    """
    Clear the entire cart.
    """

    with _cart_lock:
        cart.clear()

    return {
        "success": True,
        "message": "Cart cleared",
        "cart": [],
    }


# ============================================================
# CART ANALYSIS
#
# Uses the analysis logic from cart.py to evaluate the current
# contents of the cart.
#
# cart.py performs the analysis.
# main.py converts the resulting Product objects into
# JSON-safe dictionaries for the frontend.
#
# This endpoint is separate from recommendations because
# analysis describes the current cart, while recommendations
# suggest possible improvements.
# ============================================================

@app.get("/cart/analysis")
def cart_analysis():
    """
    Analyze the current cart.
    """

    with _cart_lock:

        analysis = analyze_cart()

    analysis["low_rated_items"] = [
        asdict(product)
        for product in analysis["low_rated_items"]
    ]

    analysis["low_confidence_items"] = [
        asdict(product)
        for product in analysis[
            "low_confidence_items"
        ]
    ]

    return analysis


# ============================================================
# RECOMMENDATIONS
#
# Provides healthier product recommendations based on the
# current cart.
#
# Recommendation order:
#   1. recommendation_ai.py
#   2. recommendations.py as fallback
#
# Primary AI pipeline:
#   cart.py
#       ↓
#   Product objects
#       ↓
#   recommendation_ai.py
#       ↓
#   Healthier alternatives
#
# If the AI recommendation system fails, the existing
# recommendations.py system is used instead.
#
# IMPORTANT:
# The cart lock is released before recommendation processing.
# AI/API calls can be slow and should not block other cart
# operations.
# ============================================================

@app.get("/recommendations")
def recommendations():
    """
    Generate healthier product recommendations.

    Primary:
        recommendations_ai.py

    Fallback:
        recommendations.py
    """

    # Copy the current cart contents while protected
    # by the lock.

    with _cart_lock:

        products = [
            item["product"]
            for item in get_cart()
        ]

    # The lock is released here.
    #
    # Recommendation processing can now happen without
    # blocking other cart operations.

    try:

        # ----------------------------------------------------
        # PRIMARY RECOMMENDATION SYSTEM
        # ----------------------------------------------------

        ai_result = recommend_for_cart(
            products
        )

        # Convert the target Product object
        # into a JSON-safe dictionary.

        if ai_result.get("target") is not None:

            ai_result["target"] = asdict(
                ai_result["target"]
            )

        # Convert alternative Product objects
        # into JSON-safe dictionaries.

        ai_result["alternatives"] = [

            {
                **recommendation,
                "product": asdict(
                    recommendation["product"]
                ),
            }

            for recommendation
            in ai_result.get(
                "alternatives",
                [],
            )
        ]

        return {
            "source": "ai",
            "recommendations": ai_result,
        }

    except Exception as error:

        # ----------------------------------------------------
        # FALLBACK RECOMMENDATION SYSTEM
        # ----------------------------------------------------

        print(
            f"[RECOMMENDATION AI] Failed: {error}"
        )

        print(
            "[RECOMMENDATIONS] "
            "Using fallback system..."
        )

        fallback_result = recommend()

        return {
            "source": "fallback",
            "recommendations": fallback_result,
        }


# ============================================================
# RUN SERVER
#
# Starts the FastAPI development server when main.py is run
# directly.
#
# The frontend can communicate with the API through the
# endpoints above while the server is running.
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
