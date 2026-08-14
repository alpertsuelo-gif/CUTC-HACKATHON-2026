# recommendations.py
#
# Cart recommendation layer.
#
# Structure:
#   1. Analyze the current cart.
#   2. Identify products that need better alternatives.
#   3. Search Open Food Facts for candidate products.
#   4. Grade the candidates.
#   5. Return products in the structure expected by the frontend.
#
# IMPORTANT:
# This module returns recommendation PRODUCT objects.
# main.py simply exposes them through /recommendations.


from cart import get_cart, analyze_cart
from grading import grade_product
from models import Product
from products import search_products


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RECOMMENDATIONS = 3


# ============================================================
# HELPERS
# ============================================================

def _dict_to_product(data):
    """
    Convert a product dictionary into a Product object.
    """

    return Product(
        barcode=str(
            data.get(
                "barcode",
                ""
            )
        ),

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
        countries=data.get("countries")
    )


def _get_cart_barcodes():
    """
    Get barcodes already present in the cart.
    """

    return {
        str(
            item["product"].barcode
        )
        for item in get_cart()
    }


def _get_categories():
    """
    Extract useful categories from products currently
    in the cart.
    """

    categories = []

    for item in get_cart():

        product = item.get(
            "product"
        )

        if not product:
            continue

        raw_categories = getattr(
            product,
            "categories",
            None
        )

        if not raw_categories:
            continue

        # Open Food Facts categories can be comma-separated.
        for category in str(
            raw_categories
        ).split(","):

            category = category.strip()

            if category:
                categories.append(
                    category
                )

    return categories


def _candidate_score(grading):
    """
    Convert a product grade into a numerical score.

    Higher score = better recommendation.
    """

    grade = str(
        grading.get(
            "grade",
            ""
        )
    ).upper()

    scores = {
        "A": 5,
        "B": 4,
        "C": 3,
        "D": 2,
        "E": 1
    }

    return scores.get(
        grade,
        0
    )


# ============================================================
# RECOMMENDATION GENERATION
# ============================================================

def recommend():
    """
    Generate product recommendations based on the current cart.

    Returns:

        [
            {
                "product": {...},
                "reason": "..."
            }
        ]
    """

    analysis = analyze_cart()

    # --------------------------------------------------------
    # Empty cart
    # --------------------------------------------------------

    if analysis["total_items"] == 0:
        return []

    cart_barcodes = _get_cart_barcodes()

    candidates = []

    # --------------------------------------------------------
    # Search using cart categories
    # --------------------------------------------------------

    categories = _get_categories()

    for category in categories:

        results = search_products(
            category=category,
            page_size=10
        )

        candidates.extend(
            results
        )

        if len(candidates) >= 30:
            break

    # --------------------------------------------------------
    # Fallback search
    # --------------------------------------------------------

    if not candidates:

        candidates = search_products(
            page_size=20
        )

    # --------------------------------------------------------
    # Remove products already in cart
    # --------------------------------------------------------

    filtered = []

    seen = set()

    for candidate in candidates:

        barcode = str(
            candidate.get(
                "barcode",
                ""
            )
        )

        if not barcode:
            continue

        if barcode in cart_barcodes:
            continue

        if barcode in seen:
            continue

        seen.add(
            barcode
        )

        filtered.append(
            candidate
        )

    # --------------------------------------------------------
    # Grade candidates
    # --------------------------------------------------------

    scored_candidates = []

    for candidate in filtered:

        try:

            product = _dict_to_product(
                candidate
            )

            grading = grade_product(
                product
            )

            score = _candidate_score(
                grading
            )

            if score <= 0:
                continue

            scored_candidates.append(
                (
                    score,
                    product,
                    grading
                )
            )

        except Exception:
            continue

    # --------------------------------------------------------
    # Best products first
    # --------------------------------------------------------

    scored_candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Build frontend response
    # --------------------------------------------------------

    recommendations = []

    for (
        score,
        product,
        grading
    ) in scored_candidates[
        :MAX_RECOMMENDATIONS
    ]:

        grade = grading.get(
            "grade",
            "?"
        )

        recommendations.append(
            {
                "product": {
                    "barcode": product.barcode,
                    "name": product.name,
                    "brand": product.brand,
                    "nutriscore": product.nutriscore,
                },

                "reason":
                    f"Recommended as a higher-rated choice "
                    f"(Grade {grade})."
            }
        )

    return recommendations
