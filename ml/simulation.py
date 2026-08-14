# ml/simulation.py
#
# Simulation engine for training the CERES recommendation AI.
#
# The model is NOT being trained to determine whether a product
# is healthy. grading.py already does that.
#
# Instead, this module answers:
#
#   "If I change this cart, how does the existing CERES
#    health score change?"
#
# It supports:
#   - loading Open Food Facts products
#   - converting CSV rows into Product objects
#   - scoring products with backend.grading
#   - calculating cart health
#   - extracting cart-state features
#   - simulating add/remove/replace actions
#
# IMPORTANT:
# This module never modifies the real backend cart.


import csv
import gzip
import random
from pathlib import Path

from grading import grade_product
from models import Product


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    Path(__file__).parent
    / "en.openfoodfacts.org.products.csv.gz"
)

MAX_DATASET_ROWS = 100_000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# NUTRITION FEATURES
# ============================================================

NUTRIENTS = [
    "energy_kj",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "protein",
    "salt",
    "sodium",
]


# ============================================================
# DATASET LOADING
# ============================================================

def load_products(max_rows=MAX_DATASET_ROWS):
    """
    Load products from the Open Food Facts compressed CSV.

    Products do NOT need every nutritional field.

    Products with absolutely no nutritional information are
    discarded because they cannot provide useful information
    to the ML system.

    Args:
        max_rows: Maximum number of CSV rows to inspect.

    Returns:
        list[Product]
    """

    products = []

    with gzip.open(
        DATASET_PATH,
        mode="rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        for row_number, row in enumerate(reader):

            if row_number >= max_rows:
                break

            barcode = _clean(
                row.get("code")
            )

            name = _clean(
                row.get("product_name")
            )

            if not barcode or not name:
                continue

            product = _row_to_product(row)

            if product is None:
                continue

            if not has_nutrition_data(product):
                continue

            products.append(product)

    return products


# ============================================================
# CSV → PRODUCT
# ============================================================

def _row_to_product(row):
    """
    Convert an Open Food Facts CSV row into a CERES Product.
    """

    try:
        return Product(
            barcode=str(
                row.get("code", "")
            ),

            name=_clean(
                row.get("product_name")
            ),

            brand=_clean(
                row.get("brands")
            ),

            nutriscore=_clean(
                row.get("nutriscore_grade")
            ),

            energy_kj=_number(
                row.get("energy-kj_100g")
            ),

            fat=_number(
                row.get("fat_100g")
            ),

            saturated_fat=_number(
                row.get("saturated-fat_100g")
            ),

            carbohydrates=_number(
                row.get("carbohydrates_100g")
            ),

            sugars=_number(
                row.get("sugars_100g")
            ),

            fiber=_number(
                row.get("fiber_100g")
            ),

            protein=_number(
                row.get("proteins_100g")
            ),

            salt=_number(
                row.get("salt_100g")
            ),

            sodium=_number(
                row.get("sodium_100g")
            ),

            ingredients=_clean(
                row.get("ingredients_text")
            ),

            categories=_clean(
                row.get("categories")
            ),

            countries=_clean(
                row.get("countries")
            ),
        )

    except Exception:
        return None


# ============================================================
# DATA CLEANING
# ============================================================

def _clean(value):
    """
    Convert empty CSV values to None.
    """

    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def _number(value):
    """
    Safely convert a CSV value to float.
    """

    if value is None:
        return None

    try:
        value = str(value).strip()

        if value == "":
            return None

        return float(value)

    except (ValueError, TypeError):
        return None


# ============================================================
# DATA QUALITY
# ============================================================

def has_nutrition_data(product):
    """
    Return True if the product has at least some nutritional
    information.

    We deliberately do NOT require every nutrient.
    """

    for nutrient in NUTRIENTS:

        value = getattr(
            product,
            nutrient,
            None,
        )

        if isinstance(value, (int, float)):
            return True

    return False


# ============================================================
# PRODUCT SCORING
# ============================================================

def score_product(product):
    """
    Score a product using the existing CERES grading system.
    """

    return grade_product(product)


# ============================================================
# CART SCORE
# ============================================================

def calculate_cart_score(cart_items):
    """
    Calculate the average CERES health score of a cart.

    cart_items format:

        {
            "product": Product,
            "grading": dict
        }
    """

    if not cart_items:
        return 0.0

    scores = []

    for item in cart_items:

        grading = item.get(
            "grading",
            {},
        )

        score = grading.get("score")

        if isinstance(score, (int, float)):
            scores.append(float(score))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ============================================================
# CART FEATURE EXTRACTION
# ============================================================

def extract_cart_features(cart_items):
    """
    Convert a cart into numerical features for the ML model.

    Missing nutritional values are represented using:

        value = 0
        missing = 1

    This allows the model to distinguish:

        actual zero

    from:

        unknown value
    """

    features = {}

    # --------------------------------------------------------
    # Current health
    # --------------------------------------------------------

    features["cart_health_score"] = calculate_cart_score(
        cart_items
    )

    features["cart_item_count"] = len(cart_items)

    # --------------------------------------------------------
    # Food groups
    # --------------------------------------------------------

    groups = {
        "fruits_and_vegetables": 0,
        "whole_grain_foods": 0,
        "meat_and_alternatives": 0,
        "dairy_and_alternatives": 0,
    }

    for item in cart_items:

        grading = item.get(
            "grading",
            {},
        )

        group = grading.get(
            "food_group"
        )

        normalized = normalize_food_group(
            group
        )

        if normalized in groups:
            groups[normalized] += 1

    for group, count in groups.items():

        features[
            f"cart_{group}"
        ] = count

    # --------------------------------------------------------
    # Nutritional features
    # --------------------------------------------------------

    for nutrient in NUTRIENTS:

        total = 0.0
        available = 0

        for item in cart_items:

            product = item["product"]

            value = getattr(
                product,
                nutrient,
                None,
            )

            if isinstance(value, (int, float)):

                total += float(value)

                available += 1

        features[
            f"cart_{nutrient}"
        ] = total

        features[
            f"cart_{nutrient}_missing"
        ] = (
            1
            if available < len(cart_items)
            else 0
        )

    return features


# ============================================================
# FOOD GROUP NORMALIZATION
# ============================================================

def normalize_food_group(food_group):
    """
    Convert grading.py food-group labels into the four
    Canada Food Guide-inspired categories used by CERES.
    """

    if not food_group:
        return None

    value = str(
        food_group
    ).lower().strip()

    if value in (
        "fruit",
        "fruits",
        "vegetable",
        "vegetables",
        "fruits_and_vegetables",
        "fruit_and_vegetable",
    ):
        return "fruits_and_vegetables"

    if value in (
        "whole_grain",
        "whole_grains",
        "whole grain",
        "whole_grain_food",
        "whole_grain_foods",
    ):
        return "whole_grain_foods"

    if value in (
        "meat",
        "meats",
        "poultry",
        "chicken",
        "beef",
        "pork",
        "turkey",
        "fish",
        "seafood",
        "egg",
        "eggs",
        "legume",
        "legumes",
        "beans",
        "lentils",
        "peas",
        "nuts",
        "seeds",
        "meat_and_alternatives",
    ):
        return "meat_and_alternatives"

    if value in (
        "dairy",
        "dairy_products",
        "milk",
        "cheese",
        "yogurt",
        "yoghurt",
        "dairy_alternative",
        "dairy_alternatives",
        "plant_milk",
        "soy_milk",
        "milk_alternative",
        "milk_alternatives",
        "dairy_and_alternatives",
    ):
        return "dairy_and_alternatives"

    return None


# ============================================================
# PRODUCT FEATURES
# ============================================================

def extract_product_features(product, grading):
    """
    Extract numerical features describing a candidate product.
    """

    features = {}

    grading_score = grading.get(
        "score"
    )

    features["candidate_score"] = (
        float(grading_score)
        if isinstance(
            grading_score,
            (int, float)
        )
        else 0.0
    )

    # --------------------------------------------------------
    # Nutrients
    # --------------------------------------------------------

    for nutrient in NUTRIENTS:

        value = getattr(
            product,
            nutrient,
            None,
        )

        if isinstance(value, (int, float)):

            features[
                f"candidate_{nutrient}"
            ] = float(value)

            features[
                f"candidate_{nutrient}_missing"
            ] = 0

        else:

            features[
                f"candidate_{nutrient}"
            ] = 0.0

            features[
                f"candidate_{nutrient}_missing"
            ] = 1

    # --------------------------------------------------------
    # Food group
    # --------------------------------------------------------

    group = normalize_food_group(
        grading.get("food_group")
    )

    for food_group in (
        "fruits_and_vegetables",
        "whole_grain_foods",
        "meat_and_alternatives",
        "dairy_and_alternatives",
    ):

        features[
            f"candidate_{food_group}"
        ] = (
            1
            if group == food_group
            else 0
        )

    return features


# ============================================================
# SIMULATE REMOVE
# ============================================================

def simulate_remove(cart_items, barcode):
    """
    Simulate removing one product from a cart.

    Returns the complete ML training representation.
    """

    before = calculate_cart_score(
        cart_items
    )

    removed_item = None

    simulated_cart = []

    for item in cart_items:

        if (
            str(item["product"].barcode)
            == str(barcode)
            and removed_item is None
        ):
            removed_item = item
            continue

        simulated_cart.append(item)

    after = calculate_cart_score(
        simulated_cart
    )

    if removed_item is None:
        return None

    candidate_product = removed_item["product"]
    candidate_grading = removed_item["grading"]

    features = extract_cart_features(
        cart_items
    )

    features.update(
        extract_product_features(
            candidate_product,
            candidate_grading,
        )
    )

    return {
        **features,

        "action": "remove",

        "candidate_barcode": str(
            candidate_product.barcode
        ),

        "candidate_name": candidate_product.name,

        "before_score": round(
            before,
            4
        ),

        "after_score": round(
            after,
            4
        ),

        "score_change": round(
            after - before,
            4
        ),
    }


# ============================================================
# SIMULATE ADD
# ============================================================

def simulate_add(cart_items, product):
    """
    Simulate adding a product to the cart.
    """

    before = calculate_cart_score(
        cart_items
    )

    grading = score_product(
        product
    )

    simulated_cart = list(
        cart_items
    )

    simulated_cart.append({
        "product": product,
        "grading": grading,
    })

    after = calculate_cart_score(
        simulated_cart
    )

    features = extract_cart_features(
        cart_items
    )

    features.update(
        extract_product_features(
            product,
            grading,
        )
    )

    return {
        **features,

        "action": "add",

        "candidate_barcode": str(
            product.barcode
        ),

        "candidate_name": product.name,

        "before_score": round(
            before,
            4
        ),

        "after_score": round(
            after,
            4
        ),

        "score_change": round(
            after - before,
            4
        ),
    }


# ============================================================
# SIMULATE REPLACE
# ============================================================

def simulate_replace(
    cart_items,
    old_barcode,
    new_product,
):
    """
    Simulate replacing one existing product.
    """

    before = calculate_cart_score(
        cart_items
    )

    new_grading = score_product(
        new_product
    )

    simulated_cart = []

    found = False

    for item in cart_items:

        if (
            str(item["product"].barcode)
            == str(old_barcode)
            and not found
        ):

            simulated_cart.append({
                "product": new_product,
                "grading": new_grading,
            })

            found = True

        else:
            simulated_cart.append(item)

    if not found:
        return None

    after = calculate_cart_score(
        simulated_cart
    )

    features = extract_cart_features(
        cart_items
    )

    features.update(
        extract_product_features(
            new_product,
            new_grading,
        )
    )

    return {
        **features,

        "action": "replace",

        "candidate_barcode": str(
            new_product.barcode
        ),

        "candidate_name": new_product.name,

        "replaced_barcode": str(
            old_barcode
        ),

        "before_score": round(
            before,
            4
        ),

        "after_score": round(
            after,
            4
        ),

        "score_change": round(
            after - before,
            4
        ),
    }


# ============================================================
# GENERATE RANDOM CART
# ============================================================

def generate_random_cart(
    dataset_products,
    min_size=2,
    max_size=10,
):
    """
    Create a simulated cart from the dataset.

    Products are graded using grading.py.
    """

    if not dataset_products:
        return []

    size = random.randint(
        min_size,
        min(
            max_size,
            len(dataset_products),
        ),
    )

    selected = random.sample(
        dataset_products,
        size,
    )

    cart_items = []

    for product in selected:

        grading = score_product(
            product
        )

        cart_items.append({
            "product": product,
            "grading": grading,
        })

    return cart_items


# ============================================================
# GENERATE TRAINING EXAMPLE
# ============================================================

def generate_example(
    cart_items,
    dataset_products,
):
    """
    Generate one random training example.

    Possible actions:

        add
        remove
        replace
    """

    if not cart_items:
        return None

    action = random.choice([
        "add",
        "remove",
        "replace",
    ])

    if action == "remove":

        item = random.choice(
            cart_items
        )

        return simulate_remove(
            cart_items,
            item["product"].barcode,
        )

    if action == "add":

        product = random.choice(
            dataset_products
        )

        return simulate_add(
            cart_items,
            product,
        )

    # Replace
    old_item = random.choice(
        cart_items
    )

    new_product = random.choice(
        dataset_products
    )

    return simulate_replace(
        cart_items,
        old_item["product"].barcode,
        new_product,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading Open Food Facts..."
    )

    products = load_products(
        max_rows=10_000
    )

    print(
        f"Loaded {len(products)} usable products."
    )

    if products:

        cart_items = generate_random_cart(
            products,
            min_size=3,
            max_size=6,
        )

        print(
            f"Generated cart with {len(cart_items)} items."
        )

        example = generate_example(
            cart_items,
            products,
        )

        print()
        print("Example:")
        print(example)