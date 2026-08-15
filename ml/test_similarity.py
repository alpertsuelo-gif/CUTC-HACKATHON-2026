from pathlib import Path
import sys


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(
    __file__
).resolve().parent

PROJECT_DIR = (
    ML_DIR.parent
)

BACKEND_DIR = (
    PROJECT_DIR /
    "backend"
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
# IMPORTS
# ============================================================

from models import Product

from ml.product_similarity import (
    DEFAULT_DATASET,
    NUTRITION_FEATURES,
    load_product_database,
    product_to_vector,
    euclidean_distance,
    distance_to_similarity,
    find_healthier_alternatives,
)


# ============================================================
# TEST DATA
# ============================================================

def create_test_product():

    return Product(
        barcode="TEST-CHOCOLATE",
        name="Test Chocolate",
        brand="Test Brand",

        # IMPORTANT:
        # grading.py expects categories to be a string.
        categories="en:chocolates, en:confectioneries",

        energy_kj=2200,
        fat=30,
        saturated_fat=18,
        carbohydrates=55,
        sugars=45,
        fiber=4,
        protein=5,
        salt=0.2,
        sodium=0.08,

        nutriscore="e",
    )


# ============================================================
# TEST 1
# ============================================================

def test_dataset_exists():

    print("=" * 70)
    print("1. Open Food Facts dataset")
    print("=" * 70)

    assert DEFAULT_DATASET.exists(), (
        f"Dataset does not exist:\n"
        f"{DEFAULT_DATASET}"
    )

    print(
        f"Dataset found:\n"
        f"{DEFAULT_DATASET}"
    )

    print("PASS\n")


# ============================================================
# TEST 2
# ============================================================

def test_dataset_loading():

    print("=" * 70)
    print("2. Dataset loading")
    print("=" * 70)

    df = load_product_database()

    assert not df.empty, (
        "Open Food Facts dataset loaded "
        "but contains no products."
    )

    print(
        f"Products loaded: {len(df):,}"
    )

    print(
        f"Columns available: {len(df.columns)}"
    )

    print("PASS\n")


# ============================================================
# TEST 3
# ============================================================

def test_nutrition_features():

    print("=" * 70)
    print("3. Nutritional features")
    print("=" * 70)

    df = load_product_database()

    missing = [
        feature
        for feature in NUTRITION_FEATURES
        if feature not in df.columns
    ]

    assert not missing, (
        "Missing nutritional features: "
        + str(missing)
    )

    print("Required nutritional features:")

    for feature in NUTRITION_FEATURES:
        print(
            f"  {feature}"
        )

    print("PASS\n")


# ============================================================
# TEST 4
# ============================================================

def test_product_vector():

    print("=" * 70)
    print("4. Product vector")
    print("=" * 70)

    product = create_test_product()

    vector = product_to_vector(
        product
    )

    assert len(vector) == len(
        NUTRITION_FEATURES
    )

    assert vector.shape == (
        len(NUTRITION_FEATURES),
    )

    print(
        "Vector:"
    )

    print(vector)

    print("PASS\n")


# ============================================================
# TEST 5
# ============================================================

def test_euclidean_distance():

    print("=" * 70)
    print("5. Euclidean distance")
    print("=" * 70)

    vector_a = product_to_vector(
        create_test_product()
    )

    vector_b = vector_a.copy()

    distance = euclidean_distance(
        vector_a,
        vector_b,
    )

    assert distance == 0.0, (
        "Identical vectors should have "
        "zero Euclidean distance."
    )

    print(
        f"Distance between identical products: "
        f"{distance}"
    )

    print("PASS\n")


# ============================================================
# TEST 6
# ============================================================

def test_similarity():

    print("=" * 70)
    print("6. Similarity calculation")
    print("=" * 70)

    product = create_test_product()

    vector = product_to_vector(
        product
    )

    distance = euclidean_distance(
        vector,
        vector,
    )

    similarity = (
        distance_to_similarity(
            distance
        )
    )

    assert similarity == 1.0, (
        "Identical products should have "
        "similarity of 1.0."
    )

    print(
        f"Similarity of identical products: "
        f"{similarity}"
    )

    print("PASS\n")


# ============================================================
# TEST 7
# ============================================================

def test_healthier_alternatives():

    print("=" * 70)
    print("7. Healthier alternatives")
    print("=" * 70)

    product = create_test_product()

    alternatives = (
        find_healthier_alternatives(
            product,
            limit=5,
        )
    )

    assert isinstance(
        alternatives,
        list,
    )

    print(
        f"Alternatives found: "
        f"{len(alternatives)}"
    )

    for index, alternative in enumerate(
        alternatives,
        start=1,
    ):

        print()
        print(
            f"{index}. "
            f"{alternative['name']}"
        )

        print(
            f"   Brand: "
            f"{alternative['brand']}"
        )

        print(
            f"   Health grade: "
            f"{alternative['health_grade']}"
        )

        print(
            f"   Similarity: "
            f"{alternative['similarity']}"
        )

        print(
            f"   Distance: "
            f"{alternative['distance']}"
        )

    print("\nPASS\n")


# ============================================================
# TEST 8
# ============================================================

def test_alternatives_are_healthier():

    print("=" * 70)
    print("8. Alternative health grades")
    print("=" * 70)

    product = create_test_product()

    original_grade = "E"

    alternatives = (
        find_healthier_alternatives(
            product,
            limit=5,
        )
    )

    grade_rank = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
    }

    for alternative in alternatives:

        grade = (
            alternative[
                "health_grade"
            ]
        )

        assert grade in grade_rank

        assert (
            grade_rank[grade]
            <
            grade_rank[original_grade]
        ), (
            f"Alternative {alternative['name']} "
            f"does not have a better health grade."
        )

    print(
        "All alternatives have a better "
        "health grade than the test product."
    )

    print("PASS\n")


# ============================================================
# TEST 9
# ============================================================

def test_alternative_structure():

    print("=" * 70)
    print("9. Alternative structure")
    print("=" * 70)

    product = create_test_product()

    alternatives = (
        find_healthier_alternatives(
            product,
            limit=3,
        )
    )

    required_fields = {
        "name",
        "brand",
        "barcode",
        "health_grade",
        "similarity",
        "distance",
        "categories",
    }

    for alternative in alternatives:

        missing = (
            required_fields
            - set(alternative.keys())
        )

        assert not missing, (
            f"Alternative is missing fields: "
            f"{missing}"
        )

    print(
        "All alternatives contain the "
        "required fields."
    )

    print("PASS\n")


# ============================================================
# RUN TESTS
# ============================================================

def main():

    tests = [
        test_dataset_exists,
        test_dataset_loading,
        test_nutrition_features,
        test_product_vector,
        test_euclidean_distance,
        test_similarity,
        test_healthier_alternatives,
        test_alternatives_are_healthier,
        test_alternative_structure,
    ]

    passed = 0
    failed = 0

    print()
    print(
        "PRODUCT SIMILARITY TEST SUITE"
    )
    print(
        "============================="
    )
    print()

    for test in tests:

        try:

            test()

            passed += 1

        except Exception as error:

            failed += 1

            print()
            print(
                "FAIL"
            )

            print(
                f"Error: {error}"
            )

            print()

    print("=" * 70)
    print("SIMILARITY TEST SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Total:  {len(tests)}"
    )

    print()

    if failed == 0:

        print(
            "ALL SIMILARITY TESTS PASSED"
        )

    else:

        print(
            "SOME SIMILARITY TESTS FAILED"
        )


if __name__ == "__main__":
    main()