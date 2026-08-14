# recommendations.py
#
# Cart recommendation layer.
#
# This module analyzes the current cart through cart.py and
# generates three types of feedback:
#
#   1. Warnings
#      Problems detected in the cart, such as low-rated products
#      or products with limited grading confidence.
#
#   2. Suggestions
#      Practical improvements based on the cart's food groups.
#
#   3. Positive feedback
#      Recognition of higher-rated choices already present.
#
# Recommendation flow:
#
#   cart.py
#       ↓
#   analyze_cart()
#       ↓
#   recommendations.py
#       ↓
#   warnings / suggestions / positive feedback
#
# IMPORTANT:
# recommend() takes no arguments because it retrieves and
# analyzes the current cart through analyze_cart().
#

from cart import analyze_cart


# ============================================================
# MAIN RECOMMENDATION
#
# Coordinates the recommendation system.
#
# The cart is analyzed once, then the resulting analysis is
# passed to each recommendation category.
# ============================================================

def recommend():
    """
    Analyze the current cart and generate recommendations.
    """

    analysis = analyze_cart()

    # Handle an empty cart before generating recommendations.

    if analysis["total_items"] == 0:
        return {
            "warnings": [],
            "suggestions": [
                "Add products to your cart first."
            ],
            "positive_feedback": []
        }

    warnings = []
    suggestions = []
    positive_feedback = []

    warnings.extend(
        _grade_recommendations(analysis)
    )

    warnings.extend(
        _confidence_recommendations(analysis)
    )

    suggestions.extend(
        _food_group_recommendations(analysis)
    )

    positive_feedback.extend(
        _positive_feedback(analysis)
    )

    return {
        "warnings": warnings,
        "suggestions": suggestions,
        "positive_feedback": positive_feedback
    }


# ============================================================
# GRADE RECOMMENDATIONS
#
# Checks the grades produced by cart.py.
#
# Products graded D or E are treated as lower-rated choices.
# ============================================================

def _grade_recommendations(analysis):
    """
    Generate warnings based on the cart's grades.
    """

    grades = analysis["grades"]

    low_rated = (
        grades["D"] +
        grades["E"]
    )

    if low_rated >= 2:
        return [
            "Your cart contains several lower-rated products. "
            "Consider choosing higher-rated alternatives."
        ]

    if low_rated == 1:
        return [
            "Your cart contains a lower-rated product. "
            "Consider a higher-rated alternative."
        ]

    return []


# ============================================================
# CONFIDENCE RECOMMENDATIONS
#
# Warns the user when one or more products have insufficient
# data for a highly reliable grade.
# ============================================================

def _confidence_recommendations(analysis):
    """
    Warn when some product grades have low confidence.
    """

    count = len(
        analysis["low_confidence_items"]
    )

    if count == 0:
        return []

    return [
        f"{count} product(s) have limited data, "
        "so their grades may be less reliable."
    ]


# ============================================================
# FOOD GROUP RECOMMENDATIONS
#
# Checks whether the cart contains fruit or vegetables.
#
# Balance recommendations are intentionally disabled for carts
# containing fewer than three products to avoid making overly
# aggressive suggestions from very little information.
# ============================================================

def _food_group_recommendations(analysis):
    """
    Suggest food groups when the cart appears unbalanced.
    """

    food_groups = analysis["food_groups"]

    # Don't make balance recommendations for tiny carts.

    if analysis["total_items"] < 3:
        return []

    suggestions = []

    has_fruit = (
        food_groups.get("fruit", 0) > 0
    )

    has_vegetable = (
        food_groups.get("vegetable", 0) > 0
    )

    if not has_fruit and not has_vegetable:
        suggestions.append(
            "Consider adding fruit or vegetables "
            "to your cart."
        )

    return suggestions


# ============================================================
# POSITIVE FEEDBACK
#
# Recognizes products with higher grades.
#
# A and B are considered higher-rated choices.
# ============================================================

def _positive_feedback(analysis):
    """
    Provide positive feedback for higher-rated choices.
    """

    grades = analysis["grades"]

    good_choices = (
        grades["A"] +
        grades["B"]
    )

    if good_choices == 0:
        return []

    if good_choices == 1:
        return [
            "Your cart contains a higher-rated choice."
        ]

    return [
        f"Your cart contains "
        f"{good_choices} higher-rated choices."
    ]
