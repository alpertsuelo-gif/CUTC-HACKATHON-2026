'''Returns scoring given:
nutrition, ingredients, food group, segment, etc
'''

# This is a determinisitc scoring algorithm
# Takes into account positive and negative factors for a score out of 100


"""
GroceryHealth grading engine.

This is a simplified implementation inspired by the updated
2023 Nutri-Score algorithm for general solid foods.

IMPORTANT:
This is NOT an official Nutri-Score calculator.
GroceryHealth uses Nutri-Score principles as the basis for
its own transparent A-E health grading system.
"""


# ---------------------------------------------------------
# Nutri-Score point tables
# ---------------------------------------------------------

def points_from_thresholds(value, thresholds):
    """
    Return the number of points based on a series of thresholds.

    Example:
        thresholds = [335, 670, 1005]
        value = 800
        -> 2
    """

    points = 0

    for threshold in thresholds:
        if value > threshold:
            points += 1
        else:
            break

    return points


# ---------------------------------------------------------
# NEGATIVE COMPONENTS
# ---------------------------------------------------------

def energy_points(kj):
    """
    Energy points for general solid foods.

    Nutri-Score 2023:
    0-10 points
    Thresholds increase by 335 kJ.
    """

    thresholds = [
        335,
        670,
        1005,
        1340,
        1675,
        2010,
        2345,
        2680,
        3015,
        3350
    ]

    return points_from_thresholds(kj, thresholds)


def saturated_fat_points(saturated_fat):
    """
    Saturated fat points.

    0-10 points.
    1 additional point for each gram above 1 g/100 g.
    """

    thresholds = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ]

    return points_from_thresholds(saturated_fat, thresholds)


def sugar_points(sugar):
    """
    Sugar points for general solid foods.

    Updated Nutri-Score scale:
    0-15 points.
    """

    thresholds = [
        3.4,
        6.8,
        10,
        14,
        17,
        20,
        24,
        27,
        31,
        34,
        37,
        41,
        44,
        48,
        51
    ]

    return points_from_thresholds(sugar, thresholds)


def salt_points(salt):
    """
    Salt points.

    Updated Nutri-Score scale:
    0-20 points.
    Each additional 0.2 g salt/100 g adds a point.
    """

    thresholds = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
        3.4,
        3.6,
        3.8,
        4.0
    ]

    return points_from_thresholds(salt, thresholds)


# ---------------------------------------------------------
# POSITIVE COMPONENTS
# ---------------------------------------------------------

def fiber_points(fiber):
    """
    Fibre points.

    Updated Nutri-Score uses a 0-5 scale.

    The updated algorithm uses a stricter fibre scale than
    the original algorithm.
    """

    thresholds = [
        3.0,
        3.7,
        4.7,
        5.9,
        7.4
    ]

    return points_from_thresholds(fiber, thresholds)


def protein_points(protein):
    """
    Protein points.

    Updated Nutri-Score uses a 0-7 scale for general foods.

    Thresholds:
    2.4, 4.8, 7.2, 9.6, 12.0, 14.5, 17.0 g/100 g
    """

    thresholds = [
        2.4,
        4.8,
        7.2,
        9.6,
        12.0,
        14.5,
        17.0
    ]

    return points_from_thresholds(protein, thresholds)


def fruit_vegetable_legume_points(percent):
    """
    Points for the percentage of fruit, vegetables and legumes.

    This simplified version uses the updated 0-5 scale:

        <= 40%  -> 0
        > 40%   -> 1
        > 60%   -> 2
        > 80%   -> 5

    Your dataset can later be expanded to distinguish the
    full Nutri-Score ingredient categories.
    """

    if percent > 80:
        return 5
    elif percent > 60:
        return 2
    elif percent > 40:
        return 1
    else:
        return 0


# ---------------------------------------------------------
# INGREDIENT ANALYSIS
# ---------------------------------------------------------

def analyze_ingredients(ingredients):
    """
    Analyze the ingredient list.

    This does NOT change the Nutri-Score calculation directly.

    Instead, it provides explanations/warnings for the user.

    This is useful for GroceryHealth because the app is designed
    to combine nutrition-label information with ingredients.
    """

    ingredients_text = " ".join(ingredients).lower()

    warnings = []
    positives = []

    # Added sugar indicators
    sugar_terms = [
        "sugar",
        "glucose syrup",
        "fructose syrup",
        "corn syrup",
        "malt syrup"
    ]

    if any(term in ingredients_text for term in sugar_terms):
        warnings.append("Contains added sugars")

    # Whole grain
    if "whole grain" in ingredients_text or "whole wheat" in ingredients_text:
        positives.append("Contains whole grains")

    # Protein additions
    protein_terms = [
        "whey protein",
        "soy protein",
        "pea protein",
        "protein isolate",
        "protein concentrate"
    ]

    if any(term in ingredients_text for term in protein_terms):
        positives.append("Contains added protein ingredients")

    return positives, warnings


# ---------------------------------------------------------
# GRADE CONVERSION
# ---------------------------------------------------------

def score_to_grade(score, food_group="general"):
    """
    Convert a Nutri-Score-style nutritional score into A-E.

    Lower scores are better.

    General-food thresholds used here:

        <= 0   -> A
        1-2    -> B
        3-10   -> C
        11-18  -> D
        >= 19  -> E

    These follow the updated general-food Nutri-Score
    classification structure.
    """

    if score <= 0:
        return "A"

    elif score <= 2:
        return "B"

    elif score <= 10:
        return "C"

    elif score <= 18:
        return "D"

    else:
        return "E"


# ---------------------------------------------------------
# MAIN GRADING FUNCTION
# ---------------------------------------------------------

def grade_product(product):
    """
    Grade a grocery product.

    Expected product structure:

        {
            "name": "...",
            "food_group": "...",
            "segment": "...",
            "nutrition": {
                "calories": ...,
                "protein": ...,
                "fat": ...,
                "saturated_fat": ...,
                "carbohydrates": ...,
                "fiber": ...,
                "sugar": ...,
                "salt": ...,
                "calcium": ...
            },
            "ingredients": [...]
        }

    Returns:

        {
            "score": ...,
            "grade": "A-E",
            "positive_factors": [...],
            "negative_factors": [...],
            "breakdown": {...}
        }
    """

    nutrition = product["nutrition"]

    # -----------------------------------------------------
    # Convert calories to kJ
    # -----------------------------------------------------

    calories = nutrition["calories"]

    # 1 kcal = approximately 4.184 kJ
    energy_kj = calories * 4.184

    # -----------------------------------------------------
    # Get nutrition values
    # -----------------------------------------------------

    protein = nutrition.get("protein", 0)
    fiber = nutrition.get("fiber", 0)
    sugar = nutrition.get("sugar", 0)
    saturated_fat = nutrition.get("saturated_fat", 0)

    # Your dataset should eventually use salt directly.
    salt = nutrition.get("salt", 0)

    # -----------------------------------------------------
    # Calculate negative points
    # -----------------------------------------------------

    energy = energy_points(energy_kj)
    saturated = saturated_fat_points(saturated_fat)
    sugars = sugar_points(sugar)
    salt_score = salt_points(salt)

    negative_points = (
        energy
        + saturated
        + sugars
        + salt_score
    )

    # -----------------------------------------------------
    # Calculate positive points
    # -----------------------------------------------------

    fiber_score = fiber_points(fiber)
    protein_score = protein_points(protein)

    # For now, use food-group information to estimate
    # whether the product contains significant F/V/legumes.
    #
    # Your future dataset should contain this explicitly.
    food_group = product.get("food_group", "").lower()

    if food_group in ["fruit", "vegetables"]:
        fvl_percent = 100

    elif food_group == "protein":
        segment = product.get("segment", "").lower()

        if segment == "legumes":
            fvl_percent = 100
        else:
            fvl_percent = 0

    else:
        fvl_percent = 0

    fvl_score = fruit_vegetable_legume_points(fvl_percent)

    # -----------------------------------------------------
    # Calculate final Nutri-Score-style score
    # -----------------------------------------------------

    positive_points = (
        fiber_score
        + protein_score
        + fvl_score
    )

    # Updated Nutri-Score has a special rule:
    #
    # When negative points are high, protein points may not
    # be counted for general foods.
    #
    # This prevents a highly sugary/salty product from getting
    # too much benefit simply because it contains protein.
    if negative_points >= 11 and fvl_score < 5:
        positive_points -= protein_score
        protein_score_used = 0
    else:
        protein_score_used = protein_score

    final_score = negative_points - (
        fiber_score
        + protein_score_used
        + fvl_score
    )

    # -----------------------------------------------------
    # Convert score to A-E
    # -----------------------------------------------------

    grade = score_to_grade(
        final_score,
        food_group
    )

    # -----------------------------------------------------
    # Explain the result
    # -----------------------------------------------------

    positive_factors = []
    negative_factors = []

    if protein_score >= 3:
        positive_factors.append(
            "Good source of protein"
        )

    if fiber_score >= 2:
        positive_factors.append(
            "Good source of fibre"
        )

    if fvl_score >= 2:
        positive_factors.append(
            "High fruit, vegetable or legume content"
        )

    if energy >= 5:
        negative_factors.append(
            "High energy density"
        )

    if saturated >= 5:
        negative_factors.append(
            "High saturated fat"
        )

    if sugars >= 5:
        negative_factors.append(
            "High sugar"
        )

    if salt_score >= 5:
        negative_factors.append(
            "High salt"
        )

    # -----------------------------------------------------
    # Ingredient analysis
    # -----------------------------------------------------

    ingredient_positives, ingredient_warnings = (
        analyze_ingredients(
            product.get("ingredients", [])
        )
    )

    positive_factors.extend(ingredient_positives)
    negative_factors.extend(ingredient_warnings)

    # -----------------------------------------------------
    # Return complete result
    # -----------------------------------------------------

    return {
        "score": final_score,
        "grade": grade,

        "positive_factors": positive_factors,

        "negative_factors": negative_factors,

        "breakdown": {
            "negative_points": {
                "energy": energy,
                "saturated_fat": saturated,
                "sugar": sugars,
                "salt": salt_score
            },

            "positive_points": {
                "fiber": fiber_score,
                "protein": protein_score_used,
                "fruit_vegetable_legume": fvl_score
            }
        }
    }

