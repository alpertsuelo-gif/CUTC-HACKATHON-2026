# main.py
#
# CERES backend API.
#
# Main structure:
#
#   Frontend
#       ↓
#   FastAPI endpoints
#       ↓
#   ┌───────────────┬───────────────┬──────────────────┐
#   │   scanner.py  │   products.py │     cart.py      │
#   │               │               │                  │
#   │ Barcode scan  │ Product data  │ Cart management  │
#   └───────────────┴───────────────┴──────────────────┘
#                       ↓
#                  grading.py
#                       ↓
#             recommendation system
#
# Recommendation order:
#
#   1. recommendations_ai.py  ← primary
#   2. recommendations.py     ← fallback
#
# main.py is responsible for connecting the frontend to the
# backend modules and converting Python objects into JSON-safe
# responses.


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

# Primary recommendation system.
from recommendations_ai import recommend_for_cart

# Backup recommendation system.
from recommendations import recommend


# ============================================================
# CONSTANTS
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

# Protect shared cart operations from simultaneous mutations.
_cart_lock = threading.Lock()


# ============================================================
# APP
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
# HELPERS
# ============================================================

def dict_to_product(data: dict) -> Product:
    """
    Convert a product dictionary into a Product object.

    products.py returns dictionaries while the rest of the
    backend works with Product objects.
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
    Convert a Product and grading result into JSON-safe data.
    """

    return {
        "product": asdict(product),
        "grading": grading,
    }


def cart_response() -> list:
    """
    Convert cart entries into JSON-safe dictionaries.
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
    Validate an uploaded image and return its safe file suffix.
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
                "File too large. "
                "Maximum size is 10 MB"
            ),
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

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
# ============================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "name": "CERES API",
        "status": "running",
    }


@app.get("/health")
def health():
    """
    Backend health check.
    """

    return {
        "status": "ok",
    }


# ============================================================
# SCANNING
# ============================================================

@app.post("/scan")
async def scan(
    file: UploadFile = File(...)
):
    """
    Receive an image, scan its barcode, retrieve the product,
    and calculate its CERES grade.
    """

    contents = await file.read()

    suffix = _validate_upload(
        file,
        contents,
    )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded image temporarily.
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(
                contents
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Scan barcode.
        # ----------------------------------------------------

        scanned = scan_product(
            temp_path
        )

        # ----------------------------------------------------
        # Resolve product.
        # ----------------------------------------------------

        if "error" not in scanned:

            product_data = scanned

        else:

            barcode = scanned.get(
                "barcode"
            )

            if not barcode:
                raise HTTPException(
                    status_code=404,
                    detail=scanned.get(
                        "error",
                        "Could not identify barcode",
                    ),
                )

            product_data = get_product(
                barcode
            )

            if "error" in product_data:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        "Product not found for "
                        "the scanned barcode"
                    ),
                )

        # ----------------------------------------------------
        # Convert and grade.
        # ----------------------------------------------------

        product = dict_to_product(
            product_data
        )

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

    except Exception:

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while "
                "processing the image"
            ),
        )

    finally:

        if temp_path:
            Path(
                temp_path
            ).unlink(
                missing_ok=True
            )


# ============================================================
# PRODUCT LOOKUP
# ============================================================

@app.get("/product/{barcode}")
def product(
    barcode: str
):
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
            detail="Product not found",
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
def add_to_cart(
    barcode: str
):
    """
    Find, convert, grade, and add a product to the cart.
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
            detail="Product not found",
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
def remove_from_cart(
    barcode: str
):
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
# ============================================================

@app.get("/cart/analysis")
def cart_analysis():
    """
    Analyze the current cart.

    Returns information such as:
        - total items
        - grade counts
        - low-rated products
        - low-confidence products
        - food groups
    """

    with _cart_lock:

        analysis = analyze_cart()

    analysis["low_rated_items"] = [
        asdict(product)
        for product in analysis.get(
            "low_rated_items",
            [],
        )
    ]

    analysis["low_confidence_items"] = [
        asdict(product)
        for product in analysis.get(
            "low_confidence_items",
            [],
        )
    ]

    return analysis


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.get("/recommendations")
def recommendations():
    """
    Generate product recommendations based on the current cart.

    Primary:
        recommendations_ai.py

    Fallback:
        recommendations.py

    The cart lock is released before recommendation generation
    so a slow AI/API request does not block cart operations.
    """

    # --------------------------------------------------------
    # Take a snapshot of the cart.
    # --------------------------------------------------------

    with _cart_lock:

        products = [
            item["product"]
            for item in get_cart()
        ]

    # --------------------------------------------------------
    # PRIMARY: AI RECOMMENDATIONS
    # --------------------------------------------------------

    try:

        ai_result = recommend_for_cart(
            products
        )

        # ----------------------------------------------------
        # Convert target Product object into a dictionary.
        # ----------------------------------------------------

        if ai_result.get("target") is not None:

            ai_result["target"] = asdict(
                ai_result["target"]
            )

        # ----------------------------------------------------
        # Convert alternative Product objects into dictionaries.
        # ----------------------------------------------------

        ai_result["alternatives"] = [

            {
                **alternative,
                "product": asdict(
                    alternative["product"]
                ),
            }

            for alternative
            in ai_result.get(
                "alternatives",
                [],
            )
        ]

        return {
            "source": "ai",
            "recommendations": ai_result,
        }

    # --------------------------------------------------------
    # FALLBACK: RULE-BASED RECOMMENDATIONS
    # --------------------------------------------------------

    except Exception as error:

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
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
