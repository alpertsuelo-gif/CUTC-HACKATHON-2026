from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(
    __file__
).resolve().parent

PROJECT_DIR = (
    BACKEND_DIR.parent
)

ML_DIR = (
    PROJECT_DIR /
    "ml"
)


if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )

if str(ML_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ML_DIR),
    )


# ============================================================
# EXISTING BACKEND
# ============================================================

from cart import get_cart
from grading import grade_product
from models import Product


# ============================================================
# ML
# ============================================================

from model import RecommendationModel

from features import (
    training_feature_names,
    encode_action,
    encode_food_group,
    encode_nutriscore,
)


MODEL_PATH = (
    ML_DIR /
    "recommendation_model.pkl"
)


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():

    global _model

    if _model is None:

        _model = (
            RecommendationModel()
        )

        _model.load(
            MODEL_PATH
        )

    return _model


# ============================================================
# SAFE NUMBERS
# ============================================================

def number(value):

    try:

        value = float(value)

        if pd.isna(value):
            return 0.0

        return value

    except (
        TypeError,
        ValueError,
    ):

        return 0.0


# ============================================================
# CART
# ============================================================

def normalize_cart(cart):

    result = []

    for item in cart:

        if isinstance(
            item,
            Product,
        ):

            product = item

            grading = (
                grade_product(
                    product
                )
            )

        elif isinstance(
            item,
            dict,
        ):

            product = item.get(
                "product"
            )

            grading = item.get(
                "grading"
            )

            if (
                grading is None
                and product is not None
            ):

                grading = (
                    grade_product(
                        product
                    )
                )

        else:

            continue

        if product is None:
            continue

        result.append(
            {
                "product": product,
                "grading": grading,
            }
        )

    return result


# ============================================================
# CART SCORE
# ============================================================

def cart_health_score(cart_items):
    """
    Calculate a user-friendly 0-100 cart health score.

    Product scores are converted into a softer 0-100 scale
    before averaging, so healthy products are not penalized
    excessively.
    """

    if not cart_items:
        return 0.0

    health_scores = []

    for item in cart_items:

        grading = item["grading"]

        score = number(
            grading.get(
                "score",
                0,
            )
        )

        # ----------------------------------------------------
        # Convert CERES score to 0-100.
        #
        # A (7+)      -> 90-100
        # B (3-6.99)  -> 75-89
        # C (-1-2.99) -> 60-74
        # D (-5--1.01) -> 35-59
        # E (<-5)     -> 0-34
        # ----------------------------------------------------

        if score >= 7:
            health = 90 + min(
                (score - 7) * 2,
                10,
            )

        elif score >= 3:
            health = 75 + (
                (score - 3) / 4
            ) * 15

        elif score >= -1:
            health = 60 + (
                (score + 1) / 4
            ) * 15

        elif score >= -5:
            health = 35 + (
                (score + 5) / 4
            ) * 25

        else:
            health = max(
                0,
                35 + (score + 5) * 5,
            )

        health_scores.append(
            health
        )

    return sum(health_scores) / len(
        health_scores
    )

# ============================================================
# PRODUCT FEATURES
# ============================================================

def product_features(
    product,
):

    grading = (
        grade_product(
            product
        )
    )

    food_group = grading.get(
        "food_group",
        "unknown",
    )

    nutriscore = (
        product.nutriscore
        or "unknown"
    )

    return {
        "product_score":
            number(
                grading.get(
                    "score",
                    0,
                )
            ),

        "energy_kj":
            number(
                product.energy_kj
            ),

        "fat":
            number(
                product.fat
            ),

        "saturated_fat":
            number(
                product.saturated_fat
            ),

        "carbohydrates":
            number(
                product.carbohydrates
            ),

        "sugars":
            number(
                product.sugars
            ),

        "fiber":
            number(
                product.fiber
            ),

        "protein":
            number(
                product.protein
            ),

        "salt":
            number(
                product.salt
            ),

        "sodium":
            number(
                product.sodium
            ),

        "food_group":
            food_group,

        "nutriscore":
            str(
                nutriscore
            ).lower(),
    }


# ============================================================
# FEATURE VECTOR
# ============================================================

def create_features(
    cart_items,
    product,
    action,
):

    cart_score = (
        cart_health_score(
            cart_items
        )
    )

    features = (
        product_features(
            product
        )
    )

    values = [
        len(cart_items),
        cart_score,

        features["product_score"],

        features["energy_kj"],
        features["fat"],
        features["saturated_fat"],
        features["carbohydrates"],
        features["sugars"],
        features["fiber"],
        features["protein"],
        features["salt"],
        features["sodium"],
    ]

    values.extend(
        encode_action(
            action
        )
    )

    values.extend(
        encode_food_group(
            features["food_group"]
        )
    )

    values.extend(
        encode_nutriscore(
            features["nutriscore"]
        )
    )

    return pd.DataFrame(
        [values],
        columns=training_feature_names(),
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(
    cart_items,
    product,
    action,
):

    model = load_model()

    X = create_features(
        cart_items,
        product,
        action,
    )

    prediction = model.predict(
        X
    )

    return float(
        prediction[0]
    )


# ============================================================
# REMOVAL
# ============================================================

# ============================================================
# REMOVAL
# ============================================================

def recommend_removal(
    cart_items,
):

    candidates = []

    for item in cart_items:

        product = item["product"]
        grading = item["grading"]

        # ----------------------------------------------------
        # Only recommend removing products graded D or E.
        #
        # A, B, and C products are never considered for removal.
        # ----------------------------------------------------

        grade = str(
            grading.get(
                "grade",
                "C",
            )
        ).upper()

        if grade not in {"D", "E"}:
            continue

        prediction = predict(
            cart_items,
            product,
            "remove",
        )

        candidates.append(
            {
                "product":
                    product,

                "grading":
                    grading,

                "predicted_change":
                    prediction,
            }
        )

    # No unhealthy products in the cart.
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x:
            x["predicted_change"],
    )


# ============================================================
# ADDITION
# ============================================================

def recommend_addition(
    cart_items,
    candidate_products,
):

    candidates = []

    for product in (
        candidate_products
    ):

        prediction = predict(
            cart_items,
            product,
            "add",
        )

        grading = (
            grade_product(
                product
            )
        )

        candidates.append(
            {
                "product":
                    product,

                "food_group":
                    grading.get(
                        "food_group",
                        "unknown",
                    ),

                "predicted_change":
                    prediction,
            }
        )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda x:
            x["predicted_change"],
    )


# ============================================================
# COMPLETE RECOMMENDATION
# ============================================================

def recommend(
    cart=None,
    candidate_products=None,
):

    if cart is None:

        cart = get_cart()

    cart_items = (
        normalize_cart(
            cart
        )
    )

    if not cart_items:

        return {
            "message":
                "Cart is empty.",

            "add": None,
            "remove": None,
        }

    current_score = (
        cart_health_score(
            cart_items
        )
    )

    # --------------------------------------------------------
    # REMOVE
    # --------------------------------------------------------

    removal = (
        recommend_removal(
            cart_items
        )
    )

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    addition = None

    if candidate_products:

        addition = (
            recommend_addition(
                cart_items,
                candidate_products,
            )
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    result = {
        "cart_health_score":
            round(
                abs(current_score),
                3,
            ),

        "add": None,

        "remove": None,
    }

    if addition:

        product = (
            addition["product"]
        )

        result["add"] = {
            "food_segment":
                addition[
                    "food_group"
                ],

            "example_product":
                product.name,

            "predicted_score_change":
                round(
                    addition[
                        "predicted_change"
                    ],
                    3,
                ),

            "reason":
                (
                    "This food segment is predicted "
                    "to produce the largest improvement "
                    "to the cart health score."
                ),
        }

    if removal:

        product = (
            removal["product"]
        )

        result["remove"] = {
            "product":
                product.name,

            "brand":
                product.brand,

            "food_segment":
                removal[
                    "grading"
                ].get(
                    "food_group",
                    "unknown",
                ),

            "predicted_score_change":
                round(
                    removal[
                        "predicted_change"
                    ],
                    3,
                )*16,

            "reason":
                (
                    "Removing this product is predicted "
                    "to produce the largest improvement "
                    "to the cart health score."
                ),
        }

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading cart..."
    )

    cart = get_cart()

    result = recommend(
        cart=cart,
        candidate_products=[],
    )

    print()
    print(
        "AI RECOMMENDATION"
    )
    print(
        "================="
    )

    print(
        result
    )