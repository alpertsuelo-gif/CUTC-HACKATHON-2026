def create_product(
    id,
    barcode,
    name,
    brand,
    food_group,
    segment,
    nutrition,
    ingredients
):
    return {
        "id": id,
        "barcode": barcode,
        "name": name,
        "brand": brand,
        "food_group": food_group,
        "segment": segment,
        "nutrition": nutrition,
        "ingredients": ingredients
    }


def create_grade(
    product_id,
    score,
    grade,
    positive_factors,
    negative_factors
):
    return {
        "product_id": product_id,
        "score": score,
        "grade": grade,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors
    }