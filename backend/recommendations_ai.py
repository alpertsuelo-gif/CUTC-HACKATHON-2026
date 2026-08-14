"""
Content-based recommendation engine for GroceryHealth.

This is the baseline recommendation system.

Given a target product, the system:

    1. Calculates its health score using grading.py
    2. Identifies its food group
    3. Finds products in the same food group
    4. Removes products that are not healthier
    5. Compares nutritional composition
    6. Ranks healthier alternatives

This is the baseline BEFORE the PyTorch recommendation model.

The goal is to eventually compare:

    Rule-based baseline
            VS
       PyTorch model
"""


import math

from grading import (
    grade_product,
    identify_food_group
)


# ============================================================
# CONFIGURATION
# ============================================================

# Nutritional features used to determine how similar two
# products are.
#
# These correspond directly to fields in models.Product.

FEATURES = [
    "energy_kj",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "protein",
    "salt",
]


# ------------------------------------------------------------
# Health-score range
# ------------------------------------------------------------
#
# Based on the current grading.py:
#
# Maximum possible positive contributions:
#
#     fiber       +3
#     protein     +3
#     food group  +2
#
# Maximum = +8
#
# Maximum negative contributions:
#
#     energy          -2
#     sugars          -3
#     saturated fat   -3
#     salt            -3
#     food category   -2
#
# Minimum = -13
#
# Therefore the theoretical current range is approximately:
#
#     -13 to +8
#
# We use these values to normalize health improvement.
#
MIN_HEALTH_SCORE = -13.0
MAX_HEALTH_SCORE = 8.0


# ------------------------------------------------------------
# Recommendation weights
# ------------------------------------------------------------
#
# Health improvement is more important than nutritional
# similarity.
#
# This means the system will generally prefer a much healthier
# alternative even if it is somewhat less nutritionally similar.

HEALTH_WEIGHT = 0.70
SIMILARITY_WEIGHT = 0.30


# ============================================================
# PRODUCT INFORMATION
# ============================================================

def get_features(product):
    """
    Extract nutritional features from a Product.

    Returns:
        list[float]
            Nutritional feature vector.

        None
            If any required nutritional value is missing.
    """

    values = []

    for feature in FEATURES:

        value = getattr(
            product,
            feature,
            None
        )

        # Missing value
        if value is None:
            return None

        try:
            value = float(value)

        except (TypeError, ValueError):
            return None

        # Reject NaN
        if math.isnan(value):
            return None

        values.append(value)

    return values


# ============================================================
# HEALTH SCORE
# ============================================================

def get_health_score(product):
    """
    Calculate the health score using grading.py.

    We do NOT store the score inside Product.

    Instead:

        Product
            ↓
        grade_product()
            ↓
        score

    Returns:
        float
        None if the product cannot be graded.
    """

    try:

        result = grade_product(
            product
        )

    except Exception:
        return None

    if not isinstance(result, dict):
        return None

    score = result.get(
        "score"
    )

    if score is None:
        return None

    try:
        score = float(score)

    except (TypeError, ValueError):
        return None

    if math.isnan(score):
        return None

    return score


# ============================================================
# PRODUCT GRADE INFORMATION
# ============================================================

def get_grade_information(product):
    """
    Get the complete grading information for a product.

    This avoids duplicating the grading logic.

    Returns:
        dictionary from grade_product()
        None if grading fails.
    """

    try:

        result = grade_product(
            product
        )

    except Exception:
        return None

    if not isinstance(result, dict):
        return None

    return result


# ============================================================
# FOOD GROUP
# ============================================================

def get_food_group(product):
    """
    Determine the food group using the SAME classification
    system as grading.py.

    Possible results include:

        fruit
        vegetable
        legume
        whole_grain
        cereal
        bread
        dairy
        meat
        fish
        nuts
        beverage
        snack
        confectionery
        dessert
        sauce
        unknown
    """

    categories = getattr(
        product,
        "categories",
        None
    )

    if not categories:
        return "unknown"

    try:

        return identify_food_group(
            categories
        )

    except Exception:
        return "unknown"


# ============================================================
# FOOD GROUP COMPATIBILITY
# ============================================================

def same_food_group(
    product_a,
    product_b
):
    """
    Determine whether two products belong to the same
    food group.

    Products with unknown food groups are NOT considered
    compatible.

    This prevents the system from recommending arbitrary
    products when Open Food Facts category information is
    missing.
    """

    group_a = get_food_group(
        product_a
    )

    group_b = get_food_group(
        product_b
    )

    if group_a == "unknown":
        return False

    if group_b == "unknown":
        return False

    return group_a == group_b


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def calculate_feature_ranges(products):
    """
    Calculate the minimum and maximum value of each
    nutritional feature.

    These ranges are calculated from the candidate database.

    Returns:
        (
            minimums,
            maximums
        )
    """

    feature_vectors = []

    for product in products:

        features = get_features(
            product
        )

        if features is not None:

            feature_vectors.append(
                features
            )

    if not feature_vectors:

        raise ValueError(
            "No products contain sufficient "
            "nutritional data."
        )

    minimums = []
    maximums = []

    for index in range(
        len(FEATURES)
    ):

        values = [
            row[index]
            for row in feature_vectors
        ]

        minimums.append(
            min(values)
        )

        maximums.append(
            max(values)
        )

    return (
        minimums,
        maximums
    )


def normalize_features(
    features,
    minimums,
    maximums
):
    """
    Normalize nutritional features to [0, 1].

    Formula:

        x' = (x - min) / (max - min)
    """

    normalized = []

    for (
        value,
        minimum,
        maximum
    ) in zip(
        features,
        minimums,
        maximums
    ):

        # If every product has exactly the same value
        # for a feature, that feature provides no useful
        # information for similarity.
        if maximum == minimum:

            normalized.append(
                0.0
            )

        else:

            normalized.append(
                (value - minimum)
                /
                (maximum - minimum)
            )

    return normalized


# ============================================================
# NUTRITIONAL DISTANCE
# ============================================================

def nutritional_distance(
    product_a,
    product_b,
    minimums,
    maximums
):
    """
    Calculate Euclidean distance between two normalized
    nutritional profiles.

    Smaller distance = greater nutritional similarity.
    """

    features_a = get_features(
        product_a
    )

    features_b = get_features(
        product_b
    )

    if features_a is None:
        return None

    if features_b is None:
        return None

    normalized_a = normalize_features(
        features_a,
        minimums,
        maximums
    )

    normalized_b = normalize_features(
        features_b,
        minimums,
        maximums
    )

    distance = math.sqrt(
        sum(
            (
                a - b
            ) ** 2

            for a, b in zip(
                normalized_a,
                normalized_b
            )
        )
    )

    return distance


def similarity_from_distance(
    distance
):
    """
    Convert nutritional distance into a similarity score.

    Returns approximately:

        1.0 = extremely similar
        0.0 = very different
    """

    if distance is None:
        return 0.0

    return (
        1.0
        /
        (1.0 + distance)
    )


# ============================================================
# HEALTH IMPROVEMENT
# ============================================================

def calculate_health_improvement(
    original_score,
    candidate_score
):
    """
    Calculate the raw health improvement.

    Example:

        Original = -5
        Candidate = +5

        Improvement = +10
    """

    return (
        candidate_score
        -
        original_score
    )


def normalize_health_improvement(
    improvement
):
    """
    Convert health improvement into approximately [0, 1].

    Uses the theoretical score range of the current grading.py.

    A larger positive improvement receives a larger value.
    """

    if improvement <= 0:
        return 0.0

    score_range = (
        MAX_HEALTH_SCORE
        -
        MIN_HEALTH_SCORE
    )

    if score_range <= 0:
        return 0.0

    normalized = (
        improvement
        /
        score_range
    )

    return min(
        max(
            normalized,
            0.0
        ),
        1.0
    )


# ============================================================
# FINAL RECOMMENDATION SCORE
# ============================================================

def calculate_recommendation_score(
    normalized_health_improvement,
    similarity
):
    """
    Calculate the final ranking score.

    Formula:

        R =
            health_weight * health_improvement
            +
            similarity_weight * similarity
    """

    return (
        HEALTH_WEIGHT
        *
        normalized_health_improvement
        +
        SIMILARITY_WEIGHT
        *
        similarity
    )


# ============================================================
# FIND ALTERNATIVES
# ============================================================

def recommend_alternatives(
    target,
    products,
    top_k=3
):
    """
    Find healthier alternatives for a target product.

    A candidate must:

        1. Be different from the target.
        2. Have a known food group.
        3. Be in the SAME food group.
        4. Have a higher health score.
        5. Have complete nutritional data.

    Candidates are ranked according to:

        70% health improvement
        30% nutritional similarity

    Returns:
        List of recommendation dictionaries.
    """

    if target is None:
        return []

    # --------------------------------------------------------
    # Grade target
    # --------------------------------------------------------

    target_grade = get_grade_information(
        target
    )

    if target_grade is None:
        return []

    target_score = target_grade.get(
        "score"
    )

    if target_score is None:
        return []

    # --------------------------------------------------------
    # Identify target food group
    # --------------------------------------------------------

    target_group = target_grade.get(
        "food_group",
        "unknown"
    )

    if target_group == "unknown":
        return []

    # --------------------------------------------------------
    # Calculate normalization ranges
    # --------------------------------------------------------

    try:

        minimums, maximums = (
            calculate_feature_ranges(
                products
            )
        )

    except ValueError:

        return []

    candidates = []

    # ========================================================
    # Evaluate every candidate
    # ========================================================

    for product in products:

        # ----------------------------------------------------
        # Don't recommend the same object
        # ----------------------------------------------------

        if product is target:
            continue

        # ----------------------------------------------------
        # Candidate must have nutritional data
        # ----------------------------------------------------

        candidate_features = (
            get_features(product)
        )

        if candidate_features is None:
            continue

        # ----------------------------------------------------
        # Grade candidate
        # ----------------------------------------------------

        candidate_grade = (
            get_grade_information(
                product
            )
        )

        if candidate_grade is None:
            continue

        candidate_score = (
            candidate_grade.get(
                "score"
            )
        )

        if candidate_score is None:
            continue

        # ----------------------------------------------------
        # Candidate must be in same food group
        # ----------------------------------------------------

        candidate_group = (
            candidate_grade.get(
                "food_group",
                "unknown"
            )
        )

        if candidate_group == "unknown":
            continue

        if candidate_group != target_group:
            continue

        # ----------------------------------------------------
        # Candidate MUST be healthier
        # ----------------------------------------------------

        if candidate_score <= target_score:
            continue

        # ----------------------------------------------------
        # Calculate nutritional distance
        # ----------------------------------------------------

        distance = nutritional_distance(
            target,
            product,
            minimums,
            maximums
        )

        if distance is None:
            continue

        # ----------------------------------------------------
        # Calculate nutritional similarity
        # ----------------------------------------------------

        similarity = (
            similarity_from_distance(
                distance
            )
        )

        # ----------------------------------------------------
        # Calculate health improvement
        # ----------------------------------------------------

        improvement = (
            calculate_health_improvement(
                target_score,
                candidate_score
            )
        )

        normalized_improvement = (
            normalize_health_improvement(
                improvement
            )
        )

        # ----------------------------------------------------
        # Calculate recommendation score
        # ----------------------------------------------------

        recommendation_score = (
            calculate_recommendation_score(
                normalized_improvement,
                similarity
            )
        )

        # ----------------------------------------------------
        # Store recommendation
        # ----------------------------------------------------

        candidates.append({

            "product": product,

            "food_group":
                candidate_group,

            "original_score":
                target_score,

            "candidate_score":
                candidate_score,

            "health_improvement":
                improvement,

            "normalized_health_improvement":
                normalized_improvement,

            "nutritional_distance":
                distance,

            "similarity":
                similarity,

            "recommendation_score":
                recommendation_score
        })

    # ========================================================
    # Rank candidates
    # ========================================================

    candidates.sort(
        key=lambda item:
            item[
                "recommendation_score"
            ],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# FIND UNHEALTHIEST PRODUCT
# ============================================================

def find_unhealthiest_product(
    products
):
    """
    Find the product with the lowest health score.

    The health score is calculated by grading.py.

    We deliberately do NOT use ML for this.

    The purpose of the AI/recommendation system is to find
    the best alternative, not to replace an already-defined
    deterministic scoring system.
    """

    scored_products = []

    for product in products:

        grade = get_grade_information(
            product
        )

        if grade is None:
            continue

        score = grade.get(
            "score"
        )

        if score is None:
            continue

        scored_products.append(
            (
                product,
                score
            )
        )

    if not scored_products:
        return None

    product, score = min(
        scored_products,
        key=lambda item:
            item[1]
    )

    return product


# ============================================================
# COMPLETE CART RECOMMENDATION
# ============================================================

def recommend_for_cart(
    products,
    top_k=3
):
    """
    Find the unhealthiest product in a cart and generate
    healthier alternatives.

    Returns:

        {
            "target": Product,
            "target_score": float,
            "food_group": str,
            "alternatives": [...]
        }
    """

    if not products:

        return {
            "target": None,
            "target_score": None,
            "food_group": None,
            "alternatives": []
        }

    # --------------------------------------------------------
    # Find worst product
    # --------------------------------------------------------

    target = find_unhealthiest_product(
        products
    )

    if target is None:

        return {
            "target": None,
            "target_score": None,
            "food_group": None,
            "alternatives": []
        }

    # --------------------------------------------------------
    # Grade target
    # --------------------------------------------------------

    target_grade = (
        get_grade_information(
            target
        )
    )

    target_score = (
        target_grade.get(
            "score"
        )
    )

    food_group = (
        target_grade.get(
            "food_group"
        )
    )

    # --------------------------------------------------------
    # Find alternatives
    # --------------------------------------------------------

    alternatives = (
        recommend_alternatives(
            target,
            products,
            top_k
        )
    )

    return {
        "target": target,
        "target_score": target_score,
        "food_group": food_group,
        "alternatives": alternatives
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_recommendations(
    result
):
    """
    Print recommendation results in a human-readable format.
    """

    target = result.get(
        "target"
    )

    if target is None:

        print(
            "No suitable target product found."
        )

        return

    alternatives = result.get(
        "alternatives",
        []
    )

    target_name = getattr(
        target,
        "name",
        "Unknown product"
    )

    target_score = result.get(
        "target_score"
    )

    food_group = result.get(
        "food_group"
    )

    print()
    print("=" * 60)
    print(
        "HEALTHIER ALTERNATIVE RECOMMENDATIONS"
    )
    print("=" * 60)

    print()
    print(
        f"Lowest-rated product: "
        f"{target_name}"
    )

    print(
        f"Food group: "
        f"{food_group}"
    )

    print(
        f"Health score: "
        f"{target_score:.2f}"
    )

    if not alternatives:

        print()
        print(
            "No suitable healthier alternatives "
            "were found."
        )

        return

    print()
    print(
        "Recommended alternatives:"
    )

    print("-" * 60)

    for index, recommendation in enumerate(
        alternatives,
        start=1
    ):

        product = (
            recommendation[
                "product"
            ]
        )

        name = getattr(
            product,
            "name",
            "Unknown product"
        )

        candidate_score = (
            recommendation[
                "candidate_score"
            ]
        )

        improvement = (
            recommendation[
                "health_improvement"
            ]
        )

        similarity = (
            recommendation[
                "similarity"
            ]
        )

        recommendation_score = (
            recommendation[
                "recommendation_score"
            ]
        )

        print()
        print(
            f"{index}. {name}"
        )

        print(
            f"   Food group: "
            f"{recommendation['food_group']}"
        )

        print(
            f"   Health score: "
            f"{candidate_score:.2f}"
        )

        print(
            f"   Health improvement: "
            f"+{improvement:.2f}"
        )

        print(
            f"   Nutritional similarity: "
            f"{similarity * 100:.1f}%"
        )

        print(
            f"   Recommendation score: "
            f"{recommendation_score:.3f}"
        )

    print()
    print("=" * 60)


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "recommendations_ai.py loaded successfully."
    )

    print()
    print(
        "Content-based recommendation engine ready."
    )

    print()
    print(
        "This module requires:"
    )

    print(
        "  - models.Product"
    )

    print(
        "  - grading.grade_product()"
    )

    print(
        "  - grading.identify_food_group()"
    )