from pathlib import Path
import sys

import torch
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

def cart_health_score(
    cart_items,
):

    if not cart_items:
        return 0.0

    scores = []

    for item in cart_items:

        grading = (
            item["grading"]
        )

        scores.append(
            number(
                grading.get(
                    "score",
                    0,
                )
            )
        )

    return sum(scores) / len(
        scores
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

def recommend_removal(
    cart_items,
):

    candidates = []

    for item in cart_items:

        product = item[
            "product"
        ]

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
                    item["grading"],

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
                current_score,
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
                ),

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