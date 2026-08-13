# Get product information from Open Food Facts using a barcode

import requests


BASE_URL = "https://world.openfoodfacts.org/api/v3/product"


def get_product(barcode):
    """
    Given a barcode from scanner.py, retrieve product information
    from the Open Food Facts API.

    Returns:
        dict: Product information
        dict: Error information if the product cannot be found
    """

    # Make sure barcode is a string
    barcode = str(barcode).strip()

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

    except requests.RequestException as e:
        return {
            "error": "Could not connect to Open Food Facts",
            "details": str(e),
            "barcode": barcode
        }

    data = response.json()

    # Product wasn't found
    if data.get("status") != "success":
        return {
            "error": "Product not found",
            "barcode": barcode
        }

    product = data.get("product", {})
    nutriments = product.get("nutriments", {})

    # Return only the information our program needs
    return {
        "barcode": barcode,

        "name": product.get("product_name"),
        "brand": product.get("brands"),

        "nutriscore": product.get("nutriscore_grade"),

        "energy_kj": nutriments.get("energy-kj_100g"),
        "fat": nutriments.get("fat_100g"),
        "saturated_fat": nutriments.get("saturated-fat_100g"),
        "carbohydrates": nutriments.get("carbohydrates_100g"),
        "sugars": nutriments.get("sugars_100g"),
        "fiber": nutriments.get("fiber_100g"),
        "protein": nutriments.get("proteins_100g"),
        "salt": nutriments.get("salt_100g"),
        "sodium": nutriments.get("sodium_100g"),

        "ingredients": product.get("ingredients_text"),
        "categories": product.get("categories"),
        "countries": product.get("countries")
    }
