# products.py
#
# Product lookup and search layer.
#
# Lookup order:
#   1. Local products.json
#   2. Open Food Facts API
#   3. Save newly retrieved products locally
#
# Search:
#   - Searches Open Food Facts for possible recommendation candidates.
#
# IMPORTANT:
# get_product() returns a dictionary because main.py converts
# that dictionary into a Product object.

import json
import os
import requests


BASE_URL = "https://world.openfoodfacts.org/api/v2/product"
SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"

DATABASE_PATH = "data/products.json"


# ============================================================
# LOCAL DATABASE
# ============================================================

def load_database():
    """
    Load the local product database.

    Returns:
        dict
    """

    if not os.path.exists(DATABASE_PATH):
        return {}

    try:
        with open(
            DATABASE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if not isinstance(data, dict):
                return {}

            return data

    except (
        json.JSONDecodeError,
        OSError
    ):
        return {}


def save_database(database):
    """
    Save the local product database.
    """

    os.makedirs(
        os.path.dirname(DATABASE_PATH),
        exist_ok=True
    )

    with open(
        DATABASE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            database,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PRODUCT CONVERSION
# ============================================================

def _convert_product(product, barcode):
    """
    Convert an Open Food Facts product into the dictionary
    structure expected by models.py and main.py.
    """

    nutriments = product.get(
        "nutriments",
        {}
    )

    return {
        "barcode": str(barcode),

        "name":
            product.get(
                "product_name"
            ),

        "brand":
            product.get(
                "brands"
            ),

        "nutriscore":
            product.get(
                "nutriscore_grade"
            ),

        "energy_kj":
            nutriments.get(
                "energy-kj_100g"
            ),

        "fat":
            nutriments.get(
                "fat_100g"
            ),

        "saturated_fat":
            nutriments.get(
                "saturated-fat_100g"
            ),

        "carbohydrates":
            nutriments.get(
                "carbohydrates_100g"
            ),

        "sugars":
            nutriments.get(
                "sugars_100g"
            ),

        "fiber":
            nutriments.get(
                "fiber_100g"
            ),

        "protein":
            nutriments.get(
                "proteins_100g"
            ),

        "salt":
            nutriments.get(
                "salt_100g"
            ),

        "sodium":
            nutriments.get(
                "sodium_100g"
            ),

        "ingredients":
            product.get(
                "ingredients_text"
            ),

        "categories":
            product.get(
                "categories"
            ),

        "countries":
            product.get(
                "countries"
            )
    }


# ============================================================
# OPEN FOOD FACTS API
# ============================================================

def get_product_from_api(barcode):
    """
    Retrieve a product from Open Food Facts.

    Returns:
        Product dictionary
        or
        Error dictionary
    """

    barcode = str(barcode).strip()

    if not barcode:
        return {
            "error": "Invalid barcode",
            "barcode": barcode
        }

    url = f"{BASE_URL}/{barcode}"

    headers = {
        "User-Agent": "CERES/1.0"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as error:

        return {
            "error": "Could not connect to Open Food Facts",
            "details": str(error),
            "barcode": barcode
        }

    except ValueError:

        return {
            "error": "Invalid response from Open Food Facts",
            "barcode": barcode
        }

    if data.get("status") != 1:

        return {
            "error": "Product not found",
            "barcode": barcode
        }

    product = data.get(
        "product",
        {}
    )

    return _convert_product(
        product,
        barcode
    )


# ============================================================
# PRODUCT SEARCH
# ============================================================

def search_products(
    category=None,
    page_size=10
):
    """
    Search Open Food Facts for products.

    Used by recommendation.py to find possible products
    that can be recommended to the user.

    Returns:
        list of product dictionaries
    """

    params = {
        "page_size": page_size,
        "fields": (
            "code,"
            "product_name,"
            "brands,"
            "nutriscore_grade,"
            "nutriments,"
            "ingredients_text,"
            "categories,"
            "countries"
        )
    }

    if category:
        params["categories_tags"] = category

    headers = {
        "User-Agent": "CERES/1.0"
    }

    try:

        response = requests.get(
            SEARCH_URL,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except (
        requests.RequestException,
        ValueError
    ):

        return []

    products = []

    for product in data.get(
        "products",
        []
    ):

        barcode = product.get("code")

        if not barcode:
            continue

        converted = _convert_product(
            product,
            barcode
        )

        if not converted.get("name"):
            continue

        products.append(
            converted
        )

    return products


# ============================================================
# MAIN PRODUCT LOOKUP
# ============================================================

def get_product(barcode):
    """
    Find a product by barcode.

    Lookup order:

        1. Local products.json
        2. Open Food Facts API

    API results are saved locally.
    """

    barcode = str(barcode).strip()

    if not barcode:

        return {
            "error": "Invalid barcode"
        }

    database = load_database()

    # --------------------------------------------------------
    # Local lookup
    # --------------------------------------------------------

    if barcode in database:

        print(
            f"[LOCAL] Product found: {barcode}"
        )

        return database[barcode]

    print(
        f"[LOCAL] Product not found: {barcode}"
    )

    print(
        "[API] Querying Open Food Facts..."
    )

    product = get_product_from_api(
        barcode
    )

    # --------------------------------------------------------
    # API error
    # --------------------------------------------------------

    if "error" in product:
        return product

    # --------------------------------------------------------
    # Save API result
    # --------------------------------------------------------

    database[barcode] = product

    save_database(
        database
    )

    print(
        f"[LOCAL] Product saved: {barcode}"
    )

    return product
