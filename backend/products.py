# products.py
#
# Product lookup layer.
#
# Lookup order:
#   1. Local products.json
#   2. Open Food Facts API
#   3. Save newly retrieved product locally
#
# IMPORTANT:
# get_product() returns a dictionary because main.py already
# converts that dictionary into a Product object.

import json
import os
import requests


BASE_URL = "https://world.openfoodfacts.org/api/v3/product"

DATABASE_PATH = "data/products.json"


# ============================================================
# LOCAL DATABASE
# ============================================================

def load_database():
    """
    Load the local product database.

    The database is stored as:

        {
            "barcode": {
                product data
            }
        }

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

            # Make sure the JSON contains a dictionary.
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
        "User-Agent": "GroceryHealth/1.0"
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
            "error":
                "Could not connect to Open Food Facts",

            "details":
                str(error),

            "barcode":
                barcode
        }

    except ValueError:

        return {
            "error":
                "Invalid response from Open Food Facts",

            "barcode":
                barcode
        }

    # --------------------------------------------------------
    # Product not found
    # --------------------------------------------------------

    if data.get("status") != 1:

        return {
            "error":
                "Product not found",

            "barcode":
                barcode
        }

    product = data.get(
        "product",
        {}
    )

    nutriments = product.get(
        "nutriments",
        {}
    )

    # --------------------------------------------------------
    # Convert OFF response into the format expected by
    # models.py and main.py
    # --------------------------------------------------------

    return {
        "barcode": barcode,

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
# MAIN PRODUCT LOOKUP
# ============================================================

def get_product(barcode):
    """
    Find a product by barcode.

    Lookup order:

        1. Local products.json
        2. Open Food Facts API

    If the product is retrieved from the API, it is saved
    locally so subsequent requests don't need the API.

    Returns:
        dict
    """

    barcode = str(barcode).strip()

    if not barcode:

        return {
            "error":
                "Invalid barcode"
        }

    # --------------------------------------------------------
    # 1. Load local database
    # --------------------------------------------------------

    database = load_database()

    # --------------------------------------------------------
    # 2. Check local database
    # --------------------------------------------------------

    if barcode in database:

        print(
            f"[LOCAL] Product found: {barcode}"
        )

        return database[barcode]

    # --------------------------------------------------------
    # 3. Product isn't local
    # --------------------------------------------------------

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
    # 4. API error
    # --------------------------------------------------------

    if "error" in product:

        return product

    # --------------------------------------------------------
    # 5. Save API result locally
    # --------------------------------------------------------

    database[barcode] = product

    save_database(
        database
    )

    print(
        f"[LOCAL] Product saved: {barcode}"
    )

    # --------------------------------------------------------
    # 6. Return dictionary
    # --------------------------------------------------------

    return product