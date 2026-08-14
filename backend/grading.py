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
    Determine the most likely food group from the Open Food Facts
    category string.

    Open Food Facts may return categories as a comma-separated
    string containing multiple categories.

    Returns:
        str: identified food group or "unknown"
    """

    if not categories:
        return "unknown"

    categories = categories.lower()

    # Check more specific groups first.
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
    Add points to the total score and record the reason.
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
    Score energy density.

    Thresholds are slightly adjusted depending on food group.
    """

    if product.energy_kj is None:
        return score

    energy = product.energy_kj
    group = identify_food_group(product.categories)

    # Beverages generally have lower energy density.
    if group == "beverage":
        if energy > 300:
            score = add_score(
                score, breakdown, "energy", -2
            )
        elif energy > 150:
            score = add_score(
                score, breakdown, "energy", -1
            )
        else:
            breakdown["energy"] = 0

    # Nuts naturally have high energy density, so avoid
    # excessively penalizing them.
    elif group == "nuts":
        if energy > 3000:
            score = add_score(
                score, breakdown, "energy", -2
            )
        elif energy > 2500:
            score = add_score(
                score, breakdown, "energy", -1
            )
        else:
            breakdown["energy"] = 0

    else:
        if energy > 2500:
            score = add_score(
                score, breakdown, "energy", -2
            )
        elif energy > 1500:
            score = add_score(
                score, breakdown, "energy", -1
            )
        else:
            breakdown["energy"] = 0

    return score


# ============================================================
# SUGAR
# ============================================================

def score_sugar(product, score, breakdown):
    """
    Score sugar content.

    Added/free sugar is not always directly available from
    Open Food Facts, so this uses total sugar as a proxy.
    """

    if product.sugars is None:
        return score

    sugar = product.sugars
    group = identify_food_group(product.categories)

    # Naturally sweet foods such as fruit are treated differently.
    if group == "fruit":
        if sugar > 30:
            score = add_score(
                score, breakdown, "sugars", -1
            )
        else:
            breakdown["sugars"] = 0

    elif group == "dairy":
        # Dairy contains naturally occurring lactose.
        if sugar > 15:
            score = add_score(
                score, breakdown, "sugars", -2
            )
        elif sugar > 7:
            score = add_score(
                score, breakdown, "sugars", -1
            )
        else:
            breakdown["sugars"] = 0

    elif group == "beverage":
        if sugar > 10:
            score = add_score(
                score, breakdown, "sugars", -3
            )
        elif sugar > 5:
            score = add_score(
                score, breakdown, "sugars", -2
            )
        elif sugar > 2.5:
            score = add_score(
                score, breakdown, "sugars", -1
            )
        else:
            breakdown["sugars"] = 0

    else:
        if sugar > 22.5:
            score = add_score(
                score, breakdown, "sugars", -3
            )
        elif sugar > 10:
            score = add_score(
                score, breakdown, "sugars", -2
            )
        elif sugar > 5:
            score = add_score(
                score, breakdown, "sugars", -1
            )
        else:
            breakdown["sugars"] = 0

    return score


# ============================================================
# SATURATED FAT
# ============================================================

def score_saturated_fat(product, score, breakdown):
    """
    Score saturated fat.
    """

    if product.saturated_fat is None:
        return score

    saturated_fat = product.saturated_fat
    group = identify_food_group(product.categories)

    # Cheese and some dairy products naturally contain
    # more saturated fat.
    if group == "dairy":
        if saturated_fat > 10:
            score = add_score(
                score, breakdown, "saturated_fat", -2
            )
        elif saturated_fat > 5:
            score = add_score(
                score, breakdown, "saturated_fat", -1
            )
        else:
            breakdown["saturated_fat"] = 0

    else:
        if saturated_fat > 10:
            score = add_score(
                score, breakdown, "saturated_fat", -3
            )
        elif saturated_fat > 5:
            score = add_score(
                score, breakdown, "saturated_fat", -2
            )
        elif saturated_fat > 1.5:
            score = add_score(
                score, breakdown, "saturated_fat", -1
            )
        else:
            breakdown["saturated_fat"] = 0

    return score


# ============================================================
# SALT
# ============================================================

def score_salt(product, score, breakdown):
    """
    Score salt content.
    """

    if product.salt is None:
        return score

    salt = product.salt

    if salt > 1.5:
        score = add_score(
            score, breakdown, "salt", -3
        )
    elif salt > 0.75:
        score = add_score(
            score, breakdown, "salt", -2
        )
    elif salt > 0.3:
        score = add_score(
            score, breakdown, "salt", -1
        )
    else:
        breakdown["salt"] = 0

    return score


# ============================================================
# FIBER
# ============================================================

def score_fiber(product, score, breakdown):
    """
    Reward fiber.

    Fiber is particularly valuable in cereal, bread,
    whole-grain and legume products.
    """

    if product.fiber is None:
        return score

    fiber = product.fiber
    group = identify_food_group(product.categories)

    if group in {"whole_grain", "cereal", "bread", "legume"}:

        if fiber >= 8:
            score = add_score(
                score, breakdown, "fiber", 3
            )
        elif fiber >= 5:
            score = add_score(
                score, breakdown, "fiber", 2
            )
        elif fiber >= 3:
            score = add_score(
                score, breakdown, "fiber", 1
            )
        else:
            breakdown["fiber"] = 0

    else:

        if fiber >= 6:
            score = add_score(
                score, breakdown, "fiber", 2
            )
        elif fiber >= 3:
            score = add_score(
                score, breakdown, "fiber", 1
            )
        else:
            breakdown["fiber"] = 0

    return score


# ============================================================
# PROTEIN
# ============================================================

def score_protein(product, score, breakdown):
    """
    Reward protein, with a larger benefit for protein-rich
    food groups.
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
        "nuts"
    }:

        if protein >= 20:
            score = add_score(
                score, breakdown, "protein", 3
            )
        elif protein >= 10:
            score = add_score(
                score, breakdown, "protein", 2
            )
        elif protein >= 5:
            score = add_score(
                score, breakdown, "protein", 1
            )
        else:
            breakdown["protein"] = 0

    else:

        if protein >= 10:
            score = add_score(
                score, breakdown, "protein", 2
            )
        elif protein >= 5:
            score = add_score(
                score, breakdown, "protein", 1
            )
        else:
            breakdown["protein"] = 0

    return score


# ============================================================
# POSITIVE FOOD-GROUP FACTORS
# ============================================================

def score_food_group(product, score, breakdown):
    """
    Apply a small positive bonus for nutritionally favorable
    food groups.

    This prevents the system from relying entirely on nutrients.
    """

    group = identify_food_group(product.categories)

    if group in {"fruit", "vegetable", "legume"}:
        score = add_score(
            score, breakdown, "food_group", 2
        )

    elif group in {"whole_grain", "fish", "nuts"}:
        score = add_score(
            score, breakdown, "food_group", 1
        )

    else:
        breakdown["food_group"] = 0

    return score


# ============================================================
# PROCESSING / CATEGORY PENALTIES
# ============================================================

def score_processing(product, score, breakdown):
    """
    Apply modest penalties to categories that frequently
    correspond to highly processed or discretionary foods.

    This is deliberately kept small so that category labels
    do not overwhelm the nutritional information.
    """

    group = identify_food_group(product.categories)

    if group == "confectionery":
        score = add_score(
            score, breakdown, "food_category", -2
        )

    elif group == "dessert":
        score = add_score(
            score, breakdown, "food_category", -1
        )

    elif group == "snack":
        score = add_score(
            score, breakdown, "food_category", -1
        )

    else:
        breakdown["food_category"] = 0

    return score


# ============================================================
# MISSING DATA
# ============================================================

def calculate_data_completeness(product):
    """
    Calculate how much of the nutritional information is available.

    This is useful because an apparently healthy score based on
    incomplete information should not be presented with the same
    confidence as a score based on complete information.
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
        value is not None for value in fields
    )

    return available / len(fields)


# ============================================================
# FINAL GRADE
# ============================================================

def convert_score_to_grade(score):
    """
    Convert numerical score into an A-E grade.

    This is our project's grading scale, not the official
    Nutri-Score algorithm.
    """

    if score >= 8:
        return "A"

    elif score >= 4:
        return "B"

    elif score >= 0:
        return "C"

    elif score >= -4:
        return "D"

    else:
        return "E"


# ============================================================
# MAIN GRADING FUNCTION
# ============================================================

def grade_product(product: Product) -> dict:
    """
    Calculate the overall health grade for a Product.

    The function combines:
        - energy
        - sugar
        - saturated fat
        - salt
        - fiber
        - protein
        - food group
        - category/processing considerations

    Returns a dictionary containing the grade, score,
    food group, confidence, and scoring breakdown.
    """

    if not isinstance(product, Product):
        raise TypeError(
            "grade_product() requires a Product object"
        )

    score = 0
    breakdown = {}

    # Identify food group
    food_group = identify_food_group(
        product.categories
    )

    # Negative factors
    score = score_energy(
        product, score, breakdown
    )

    score = score_sugar(
        product, score, breakdown
    )

    score = score_saturated_fat(
        product, score, breakdown
    )

    score = score_salt(
        product, score, breakdown
    )

    # Positive factors
    score = score_fiber(
        product, score, breakdown
    )

    score = score_protein(
        product, score, breakdown
    )

    # Food group
    score = score_food_group(
        product, score, breakdown
    )

    # Processing/category consideration
    score = score_processing(
        product, score, breakdown
    )

    # Final grade
    grade = convert_score_to_grade(score)

    # Determine confidence from available data
    completeness = calculate_data_completeness(product)

    if completeness >= 0.83:
        confidence = "high"

    elif completeness >= 0.50:
        confidence = "medium"

    else:
        confidence = "low"

    return {
        "barcode": product.barcode,
        "name": product.name,
        "brand": product.brand,

        "food_group": food_group,

        "score": score,
        "grade": grade,

        "confidence": confidence,
        "data_completeness": round(
            completeness * 100, 1
        ),

        "breakdown": breakdown,

        "nutriscore_reference": product.nutriscore
    }