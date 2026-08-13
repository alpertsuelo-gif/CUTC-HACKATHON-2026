from models import Product


cart = []


def add_product(product: Product, grading: dict):
    """Add a product and its grading result to the cart."""

    if not isinstance(product, Product):
        raise TypeError("product must be a Product object")

    if not isinstance(grading, dict):
        raise TypeError("grading must be a dictionary")

    cart.append({
        "product": product,
        "grading": grading.copy()
    })


def remove_product(barcode: str):
    """Remove a product from the cart using its barcode."""

    barcode = str(barcode).strip()

    for item in cart:
        product = item["product"]

        if str(product.barcode).strip() == barcode:
            cart.remove(item)
            return True

    return False


def get_cart():
    """Return the current cart."""

    return cart.copy()


def analyze_cart():
    """Analyze the products currently in the cart."""

    analysis = {
        "total_items": len(cart),
        "grades": {
            "A": 0,
            "B": 0,
            "C": 0,
            "D": 0,
            "E": 0
        },
        "food_groups": {},
        "low_rated_items": [],
        "low_confidence_items": []
    }

    for item in cart:
        product = item["product"]
        grading = item["grading"]

        # Count grades
        grade = grading.get("grade")

        if grade in analysis["grades"]:
            analysis["grades"][grade] += 1

        # Count food groups
        food_group = grading.get("food_group")

        if food_group:
            analysis["food_groups"][food_group] = (
                analysis["food_groups"].get(food_group, 0) + 1
            )

        # Track low-rated products
        if grade in ("D", "E"):
            analysis["low_rated_items"].append(product)

        # Track uncertain grading
        if grading.get("confidence") == "low":
            analysis["low_confidence_items"].append(product)

    return analysis
