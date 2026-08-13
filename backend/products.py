# products.py
# Get product information from Open Food Facts using a barcode

import requests

from models import Product


BASE_URL = "https://world.openfoodfacts.org/api/v2/product"


def get_product(barcode):
    """
    Given a barcode from scanner.py, retrieve product information
    from the Open Food Facts API.

    Returns:
        Product: Product object if found
        dict: Error information if the product cannot be found
    """

    # Make sure barcode is a string
    barcode = str(barcode).strip()

    if not barcode:
        return {
            "error": "Invalid barcode"
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

    except requests.RequestException as e:
        return {
            "error": "Could not connect to Open Food Facts",
            "details": str(e),
            "barcode": barcode
        }

    # Make sure the API returned valid JSON
    try:
        data = response.json()

    except ValueError:
        return {
            "error": "Open Food Facts returned invalid data",
            "barcode": barcode
        }

    # Open Food Facts returns status 0 when product isn't found
    if data.get("status") != 1:
        return {
            "error": "Product not found",
            "barcode": barcode
        }

    product = data.get("product", {})
    nutriments = product.get("nutriments", {})

    # Create our Product object
    return Product(
        barcode=barcode,

        name=product.get("product_name"),
        brand=product.get("brands"),

        nutriscore=product.get("nutriscore_grade"),

        energy_kj=nutriments.get("energy-kj_100g"),
        fat=nutriments.get("fat_100g"),
        saturated_fat=nutriments.get("saturated-fat_100g"),
        carbohydrates=nutriments.get("carbohydrates_100g"),
        sugars=nutriments.get("sugars_100g"),
        fiber=nutriments.get("fiber_100g"),
        protein=nutriments.get("proteins_100g"),
        salt=nutriments.get("salt_100g"),
        sodium=nutriments.get("sodium_100g"),

        ingredients=product.get("ingredients_text"),
        categories=product.get("categories"),
        countries=product.get("countries")
    )

if __name__ == "__main__": #TESTING
    barcode = input("Enter barcode: ")

    result = get_product(barcode)

    if isinstance(result, Product):
        print("\nProduct found:")
        print(f"Name: {result.name}")
        print(f"Brand: {result.brand}")
        print(f"Barcode: {result.barcode}")
        print(f"Sugars: {result.sugars} g/100g")
        print(f"Saturated fat: {result.saturated_fat} g/100g")
        print(f"Fiber: {result.fiber} g/100g")
        print(f"Protein: {result.protein} g/100g")
        print(f"Categories: {result.categories}")
        print(f"Nutri-Score: {result.nutriscore}")

    else:
        print(result["error"])