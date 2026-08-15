from models import Product


# ============================================================
# FOOD GROUP CLASSIFICATION
# ============================================================

FOOD_GROUPS = {
    "fruit": [
        "fruits",
        "fruit",
        "fresh fruits",
        "berries",
    ],

    "vegetable": [
        "vegetables",
        "vegetable",
        "fresh vegetables",
        "leaf vegetables",
    ],

    "legume": [
        "legumes",
        "pulses",
        "beans",
        "lentils",
        "chickpeas",
        "peas",
    ],

    "whole_grain": [
        "whole grain",
        "whole grains",
        "wholemeal",
        "whole wheat",
        "oat",
        "oats",
        "bran",
    ],

    "cereal": [
        "breakfast cereals",
        "cereals",
        "cereal",
        "granola",
    ],

    "bread": [
        "breads",
        "bread",
        "wholemeal bread",
    ],

    "dairy": [
        "dairy",
        "milk",
        "yogurts",
        "yogurt",
        "cheeses",
        "cheese",
    ],

    "meat": [
        "meats",
        "meat",
        "beef",
        "chicken",
        "pork",
        "turkey",
    ],

    "fish": [
        "fish",
        "seafood",
        "shellfish",
    ],

    "nuts": [
        "nuts",
        "nut",
        "almonds",
        "walnuts",
        "peanuts",
        "seeds",
    ],

    "beverage": [
        "beverages",
        "beverage",
        "drinks",
        "soft drinks",
        "juices",
        "fruit juices",
        "sodas",
    ],

    "snack": [
        "snacks",
        "snack",
        "chips",
        "crisps",
        "crackers",
    ],

    "confectionery": [
        "confectioneries",
        "confectionery",
        "chocolates",
        "chocolate",
        "candies",
        "candy",
        "sweets",
    ],

    "dessert": [
        "desserts",
        "dessert",
        "ice creams",
        "ice cream",
        "cakes",
        "cookies",
        "biscuits",
        "pastries",
    ],

    "sauce": [
        "sauces",
        "sauce",
        "dressings",
        "mayonnaise",
    ],
}


def identify_food_group(categories):
    """
    Determine the most likely food group from Open Food Facts
    categories.
    """

    if not categories:
        return "unknown"

    categories = categories.lower()

    # More specific categories first.
    for group, keywords in FOOD_GROUPS.items():
        for keyword in keywords:
            if keyword in categories:
                return group

    return "unknown"


# ============================================================
# SCORING HELPERS
# ============================================================

def add_score(score, breakdown, factor, points):
    """
    Add points to the score and record the reason.
    """

    score += points

    if factor not in breakdown:
        breakdown[factor] = 0

    breakdown[factor] += points

    return score


# ============================================================
# ENERGY
# ============================================================

def score_energy(product, score, breakdown):
    """
    Energy should be treated as a relatively minor factor.

    High energy density is not automatically unhealthy:
        - oats
        - nuts
        - seeds
        - whole grains

    can naturally be energy dense while still being nutritious.
    """

    if product.energy_kj is None:
        return score

    energy = product.energy_kj
    group = identify_food_group(product.categories)

    # Whole grains and legumes
    # should not be heavily penalized for energy.
    if group in {
        "whole_grain",
        "legume",
        "cereal",
        "bread",
    }:

        if energy > 2500:
            score = add_score(
                score,
                breakdown,
                "energy",
                -1
            )
        else:
            breakdown["energy"] = 0

    # Nuts and seeds are naturally energy dense.
    elif group == "nuts":

        if energy > 3000:
            score = add_score(
                score,
                breakdown,
                "energy",
                -1
            )
        else:
            breakdown["energy"] = 0

    # Beverages
    elif group == "beverage":

        if energy > 300:
            score = add_score(
                score,
                breakdown,
                "energy",
                -2
            )
        elif energy > 150:
            score = add_score(
                score,
                breakdown,
                "energy",
                -1
            )
        else:
            breakdown["energy"] = 0

    else:

        if energy > 3000:
            score = add_score(
                score,
                breakdown,
                "energy",
                -2
            )
        elif energy > 2000:
            score = add_score(
                score,
                breakdown,
                "energy",
                -1
            )
        else:
            breakdown["energy"] = 0

    return score


# ============================================================
# SUGAR
# ============================================================

def score_sugar(product, score, breakdown):
    """
    Sugar is treated differently depending on food group.

    Naturally occurring sugar in fruit and dairy is much less
    concerning than high sugar in soft drinks or confectionery.
    """

    if product.sugars is None:
        return score

    sugar = product.sugars
    group = identify_food_group(product.categories)

    # Fruit
    if group == "fruit":

        if sugar > 30:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        else:
            breakdown["sugars"] = 0

    # Dairy contains naturally occurring lactose.
    elif group == "dairy":

        if sugar > 20:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        elif sugar > 12:
            score = add_score(
                score,
                breakdown,
                "sugars",
                0
            )
        else:
            breakdown["sugars"] = 0

    # Whole grains such as oats can contain small amounts
    # of naturally occurring carbohydrate/sugar.
    elif group in {
        "whole_grain",
        "cereal",
        "bread",
        "legume",
    }:

        if sugar > 20:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -2
            )
        elif sugar > 10:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        else:
            breakdown["sugars"] = 0

    # Beverages are treated more strictly.
    elif group == "beverage":

        if sugar > 10:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -3
            )
        elif sugar > 5:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -2
            )
        elif sugar > 2.5:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        else:
            breakdown["sugars"] = 0

    # Confectionery and desserts.
    elif group in {
        "confectionery",
        "dessert",
    }:

        if sugar > 30:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -3
            )
        elif sugar > 15:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -2
            )
        elif sugar > 8:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        else:
            breakdown["sugars"] = 0

    else:

        if sugar > 22.5:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -2
            )
        elif sugar > 10:
            score = add_score(
                score,
                breakdown,
                "sugars",
                -1
            )
        else:
            breakdown["sugars"] = 0

    return score


# ============================================================
# SATURATED FAT
# ============================================================

def score_saturated_fat(product, score, breakdown):
    """
    Saturated fat is important, but should not dominate the score.
    """

    if product.saturated_fat is None:
        return score

    saturated_fat = product.saturated_fat
    group = identify_food_group(product.categories)

    # Nuts/seeds can contain substantial fat while still being
    # nutritionally beneficial.
    if group == "nuts":

        if saturated_fat > 15:
            score = add_score(
                score,
                breakdown,
                "saturated_fat",
                -1
            )
        else:
            breakdown["saturated_fat"] = 0

    # Dairy
    elif group == "dairy":

        if saturated_fat > 10:
            score = add_score(
                score,
                breakdown,
                "saturated_fat",
                -2
            )
        elif saturated_fat > 5:
            score = add_score(
                score,
                breakdown,
                "saturated_fat",
                -1
            )
        else:
            breakdown["saturated_fat"] = 0

    else:

        if saturated_fat > 10:
            score = add_score(
                score,
                breakdown,
                "saturated_fat",
                -2
            )
        elif saturated_fat > 5:
            score = add_score(
                score,
                breakdown,
                "saturated_fat",
                -1
            )
        else:
            breakdown["saturated_fat"] = 0

    return score


# ============================================================
# SALT
# ============================================================

def score_salt(product, score, breakdown):
    """
    Salt remains an important negative factor, but thresholds
    are slightly relaxed.
    """

    if product.salt is None:
        return score

    salt = product.salt

    if salt > 2:
        score = add_score(
            score,
            breakdown,
            "salt",
            -2
        )

    elif salt > 1:
        score = add_score(
            score,
            breakdown,
            "salt",
            -1
        )

    else:
        breakdown["salt"] = 0

    return score


# ============================================================
# FIBER
# ============================================================

def score_fiber(product, score, breakdown):
    """
    Strongly reward fiber, especially in whole grains,
    cereals, bread, and legumes.
    """

    if product.fiber is None:
        return score

    fiber = product.fiber
    group = identify_food_group(product.categories)

    if group in {
        "whole_grain",
        "cereal",
        "bread",
        "legume",
    }:

        if fiber >= 8:
            score = add_score(
                score,
                breakdown,
                "fiber",
                4
            )

        elif fiber >= 5:
            score = add_score(
                score,
                breakdown,
                "fiber",
                3
            )

        elif fiber >= 3:
            score = add_score(
                score,
                breakdown,
                "fiber",
                2
            )

        else:
            breakdown["fiber"] = 0

    else:

        if fiber >= 6:
            score = add_score(
                score,
                breakdown,
                "fiber",
                2
            )

        elif fiber >= 3:
            score = add_score(
                score,
                breakdown,
                "fiber",
                1
            )

        else:
            breakdown["fiber"] = 0

    return score


# ============================================================
# PROTEIN
# ============================================================

def score_protein(product, score, breakdown):
    """
    Reward protein without making it a requirement for
    foods such as oats, fruits, and vegetables.
    """

    if product.protein is None:
        return score

    protein = product.protein
    group = identify_food_group(product.categories)

    if group in {
        "legume",
        "meat",
        "fish",
        "dairy",
        "nuts",
    }:

        if protein >= 20:
            score = add_score(
                score,
                breakdown,
                "protein",
                3
            )

        elif protein >= 10:
            score = add_score(
                score,
                breakdown,
                "protein",
                2
            )

        elif protein >= 5:
            score = add_score(
                score,
                breakdown,
                "protein",
                1
            )

        else:
            breakdown["protein"] = 0

    else:

        if protein >= 10:
            score = add_score(
                score,
                breakdown,
                "protein",
                2
            )

        elif protein >= 5:
            score = add_score(
                score,
                breakdown,
                "protein",
                1
            )

        else:
            breakdown["protein"] = 0

    return score


# ============================================================
# FOOD GROUP
# ============================================================

def score_food_group(product, score, breakdown):
    """
    Strongly reward naturally nutritious food groups.

    This is the main change that allows foods such as plain oats,
    lentils, fruits, and vegetables to achieve A grades.
    """

    group = identify_food_group(product.categories)

    if group == "vegetable":

        score = add_score(
            score,
            breakdown,
            "food_group",
            5
        )

    elif group == "fruit":

        score = add_score(
            score,
            breakdown,
            "food_group",
            5
        )

    elif group == "legume":

        score = add_score(
            score,
            breakdown,
            "food_group",
            5
        )

    elif group == "whole_grain":

        score = add_score(
            score,
            breakdown,
            "food_group",
            5
        )

    elif group == "fish":

        score = add_score(
            score,
            breakdown,
            "food_group",
            4
        )

    elif group == "nuts":

        score = add_score(
            score,
            breakdown,
            "food_group",
            4
        )

    elif group == "dairy":

        score = add_score(
            score,
            breakdown,
            "food_group",
            2
        )

    elif group == "bread":

        score = add_score(
            score,
            breakdown,
            "food_group",
            2
        )

    else:

        breakdown["food_group"] = 0

    return score


# ============================================================
# PROCESSING / CATEGORY PENALTIES
# ============================================================

def score_processing(product, score, breakdown):
    """
    Apply modest penalties to discretionary food categories.

    These penalties are intentionally small so that nutritional
    information remains more important than category labels.
    """

    group = identify_food_group(product.categories)

    if group == "confectionery":

        score = add_score(
            score,
            breakdown,
            "food_category",
            -3
        )

    elif group == "dessert":

        score = add_score(
            score,
            breakdown,
            "food_category",
            -2
        )

    elif group == "snack":

        score = add_score(
            score,
            breakdown,
            "food_category",
            -1
        )

    else:

        breakdown["food_category"] = 0

    return score


# ============================================================
# WHOLE FOOD BONUS
# ============================================================

def score_whole_food(product, score, breakdown):
    """
    Give a small additional bonus to foods that are inherently
    nutritious staples.

    This is especially useful for:
        - oats
        - whole grains
        - legumes
        - fruits
        - vegetables
    """

    group = identify_food_group(product.categories)

    if group in {
        "fruit",
        "vegetable",
        "legume",
        "whole_grain",
    }:

        score = add_score(
            score,
            breakdown,
            "whole_food",
            2
        )

    else:

        breakdown["whole_food"] = 0

    return score


# ============================================================
# MISSING DATA
# ============================================================

def calculate_data_completeness(product):
    """
    Calculate how much nutritional information is available.
    """

    fields = [
        product.energy_kj,
        product.sugars,
        product.saturated_fat,
        product.salt,
        product.fiber,
        product.protein,
    ]

    available = sum(
        value is not None
        for value in fields
    )

    return available / len(fields)


# ============================================================
# FINAL GRADE
# ============================================================

def convert_score_to_grade(score):
    """
    Convert the internal score into an A-E grade.

    The thresholds are intentionally more forgiving than the
    previous version.
    """

    if score >= 7:
        return "A"

    elif score >= 3:
        return "B"

    elif score >= -1:
        return "C"

    elif score >= -5:
        return "D"

    else:
        return "E"


# ============================================================
# MAIN GRADING FUNCTION
# ============================================================

def grade_product(product: Product) -> dict:
    """
    Calculate the overall CERES health grade.
    """

    if not isinstance(product, Product):
        raise TypeError(
            "grade_product() requires a Product object"
        )

    score = 0
    breakdown = {}

    # --------------------------------------------------------
    # Identify food group
    # --------------------------------------------------------

    food_group = identify_food_group(
        product.categories
    )

    # --------------------------------------------------------
    # Negative factors
    # --------------------------------------------------------

    score = score_energy(
        product,
        score,
        breakdown
    )

    score = score_sugar(
        product,
        score,
        breakdown
    )

    score = score_saturated_fat(
        product,
        score,
        breakdown
    )

    score = score_salt(
        product,
        score,
        breakdown
    )

    # --------------------------------------------------------
    # Positive nutritional factors
    # --------------------------------------------------------

    score = score_fiber(
        product,
        score,
        breakdown
    )

    score = score_protein(
        product,
        score,
        breakdown
    )

    # --------------------------------------------------------
    # Food quality
    # --------------------------------------------------------

    score = score_food_group(
        product,
        score,
        breakdown
    )

    score = score_whole_food(
        product,
        score,
        breakdown
    )

    # --------------------------------------------------------
    # Processing/category
    # --------------------------------------------------------

    score = score_processing(
        product,
        score,
        breakdown
    )

    # --------------------------------------------------------
    # Final grade
    # --------------------------------------------------------

    grade = convert_score_to_grade(score)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    completeness = calculate_data_completeness(
        product
    )

    if completeness >= 0.83:
        confidence = "high"

    elif completeness >= 0.50:
        confidence = "medium"

    else:
        confidence = "low"

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "barcode": product.barcode,
        "name": product.name,
        "brand": product.brand,

        "food_group": food_group,

        "score": score,
        "grade": grade,

        "confidence": confidence,

        "data_completeness": round(
            completeness * 100,
            1
        ),

        "breakdown": breakdown,

        "nutriscore_reference": product.nutriscore,
    }