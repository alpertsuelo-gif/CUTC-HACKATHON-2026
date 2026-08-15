from pathlib import Path
import sys
import traceback

import pandas as pd


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

from grading import grade_product

from cart import (
    get_cart,
    analyze_cart,
)

from recommendations_ai import (
    recommend,
    predict,
    normalize_cart,
    cart_health_score,
    score_change_to_grade,
)

from product_similarity import (
    find_healthier_alternatives,
)

from features import (
    create_training_features,
    training_feature_names,
)

from model import RecommendationModel


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    ML_DIR /
    "training_data.csv"
)

MODEL_PATH = (
    ML_DIR /
    "recommendation_model.pkl"
)

OPENFOODFACTS_PATH = (
    ML_DIR /
    "en.openfoodfacts.org.products.csv.gz"
)


# ============================================================
# TEST HELPER
# ============================================================

def run_test(
    number,
    name,
    function,
):

    print("=" * 70)
    print(
        f"{number}. {name}"
    )
    print("=" * 70)

    try:

        function()

        print()
        print("PASS")
        print()

        return True

    except Exception as error:

        print()
        print("FAIL")
        print()

        print(
            f"Error: {error}"
        )

        print()

        traceback.print_exc()

        print()

        return False


# ============================================================
# TEST 1
# ============================================================

def test_imports():

    assert Product is not None
    assert grade_product is not None
    assert get_cart is not None
    assert analyze_cart is not None

    assert recommend is not None
    assert predict is not None

    assert RecommendationModel is not None

    print(
        "Backend and ML modules imported successfully."
    )


# ============================================================
# TEST 2
# ============================================================

def test_product_creation():

    product = Product(
        barcode="TEST001",
        name="Test Apple",
        brand="Test Brand",

        categories="en:fruits",

        energy_kj=218,
        fat=0.2,
        saturated_fat=0.0,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0.0,
        sodium=0.0,

        nutriscore="a",
    )

    assert product.name == (
        "Test Apple"
    )

    print(
        f"Product created successfully: "
        f"{product.name}"
    )


# ============================================================
# TEST 3
# ============================================================

def test_grading():

    product = Product(
        barcode="TEST002",
        name="Test Apple",

        brand="Test Brand",

        categories="en:fruits",

        energy_kj=218,
        fat=0.2,
        saturated_fat=0.0,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0.0,
        sodium=0.0,

        nutriscore="a",
    )

    grading = (
        grade_product(
            product
        )
    )

    assert isinstance(
        grading,
        dict,
    )

    assert "score" in grading

    print(
        "Product grading completed successfully."
    )

    print(
        grading
    )


# ============================================================
# TEST 4
# ============================================================

def test_cart():

    cart = get_cart()

    assert isinstance(
        cart,
        list,
    )

    print(
        "Current cart retrieved successfully."
    )

    print(
        f"Cart type: {type(cart).__name__}"
    )

    print(
        f"Cart size: {len(cart)}"
    )

    analysis = analyze_cart()

    assert isinstance(
        analysis,
        dict,
    )

    print()
    print(
        "Cart analysis completed successfully."
    )

    print(
        analysis
    )


# ============================================================
# TEST 5
# ============================================================

def test_training_dataset():

    assert DATASET_PATH.exists(), (
        f"Training dataset does not exist: "
        f"{DATASET_PATH}"
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    assert len(df) > 0

    assert "action" in df.columns

    assert "score_change" in df.columns

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print()
    print("Actions:")

    print(
        df["action"].value_counts()
    )

    print()
    print(
        "Score change statistics:"
    )

    print(
        df["score_change"].describe()
    )


# ============================================================
# TEST 6
# ============================================================

def test_feature_generation():

    df = pd.read_csv(
        DATASET_PATH
    )

    X = create_training_features(
        df
    )

    assert isinstance(
        X,
        pd.DataFrame,
    )

    assert len(X) == len(df)

    feature_names = (
        training_feature_names()
    )

    assert list(X.columns) == (
        feature_names
    )

    print(
        f"Feature matrix shape: "
        f"{X.shape}"
    )

    print()
    print("Features:")

    for feature in feature_names:

        print(
            f"  {feature}"
        )


# ============================================================
# TEST 7
# ============================================================

def test_model_training():

    df = pd.read_csv(
        DATASET_PATH
    )

    X = create_training_features(
        df
    )

    y = df[
        "score_change"
    ]

    model = (
        RecommendationModel()
    )

    X_train = X.iloc[
        : int(len(X) * 0.8)
    ]

    X_test = X.iloc[
        int(len(X) * 0.8):
    ]

    y_train = y.iloc[
        : int(len(y) * 0.8)
    ]

    y_test = y.iloc[
        int(len(y) * 0.8):
    ]

    model.train(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    assert len(predictions) == (
        len(y_test)
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print(
        f"Predictions: "
        f"{len(predictions)}"
    )


# ============================================================
# TEST 8
# ============================================================

def test_model_predictions():

    df = pd.read_csv(
        DATASET_PATH
    )

    X = create_training_features(
        df
    )

    y = df[
        "score_change"
    ]

    model = (
        RecommendationModel()
    )

    split = int(
        len(X) * 0.8
    )

    model.train(
        X.iloc[:split],
        y.iloc[:split],
    )

    predictions = model.predict(
        X.iloc[split:]
    )

    assert len(predictions) > 0

    print(
        "Sample predictions:"
    )

    for index, prediction in enumerate(
        predictions[:10],
        start=1,
    ):

        print(
            f"  Sample {index}: "
            f"{prediction:.4f}"
        )


# ============================================================
# TEST 9
# ============================================================

def test_model_save_load():

    df = pd.read_csv(
        DATASET_PATH
    )

    X = create_training_features(
        df
    )

    y = df[
        "score_change"
    ]

    split = int(
        len(X) * 0.8
    )

    model = (
        RecommendationModel()
    )

    model.train(
        X.iloc[:split],
        y.iloc[:split],
    )

    temporary_model = (
        ML_DIR /
        "_test_model.pkl"
    )

    model.save(
        temporary_model
    )

    assert temporary_model.exists()

    loaded = (
        RecommendationModel()
    )

    loaded.load(
        temporary_model
    )

    original_predictions = (
        model.predict(
            X.iloc[split:split + 10]
        )
    )

    loaded_predictions = (
        loaded.predict(
            X.iloc[split:split + 10]
        )
    )

    assert (
        original_predictions
        == loaded_predictions
    ).all()

    temporary_model.unlink()

    print(
        "Model successfully saved."
    )

    print(
        "Model successfully loaded."
    )

    print(
        "Predictions remain identical."
    )


# ============================================================
# TEST 10
# ============================================================

def test_feature_importance():

    df = pd.read_csv(
        DATASET_PATH
    )

    X = create_training_features(
        df
    )

    y = df[
        "score_change"
    ]

    model = (
        RecommendationModel()
    )

    model.train(
        X,
        y,
    )

    importance = (
        model.feature_importance()
    )

    assert importance is not None

    print(
        "Top 10 features:"
    )

    if isinstance(
        importance,
        dict,
    ):

        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for name, value in (
            sorted_features[:10]
        ):

            print(
                f"  {name:<30} "
                f"{value:.4f}"
            )


# ============================================================
# TEST 11
# ============================================================

def test_saved_model():

    assert MODEL_PATH.exists(), (
        f"Trained model does not exist:\n"
        f"{MODEL_PATH}\n\n"
        "Run:\n"
        "python -m ml.train"
    )

    model = (
        RecommendationModel()
    )

    model.load(
        MODEL_PATH
    )

    print(
        "Loaded trained model:"
    )

    print(
        f"  {MODEL_PATH}"
    )


# ============================================================
# TEST 12
# ============================================================

def test_ai_prediction():

    assert MODEL_PATH.exists(), (
        "Trained model does not exist."
    )

    product = Product(
        barcode="TEST003",
        name="Test Apple",

        brand="Test Brand",

        categories="en:fruits",

        energy_kj=218,
        fat=0.2,
        saturated_fat=0.0,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0.0,
        sodium=0.0,

        nutriscore="a",
    )

    cart = [
        product
    ]

    cart_items = (
        normalize_cart(
            cart
        )
    )

    prediction = predict(
        cart_items,
        product,
        "add",
    )

    assert isinstance(
        prediction,
        float,
    )

    recommendation_grade = (
        score_change_to_grade(
            prediction
        )
    )

    assert recommendation_grade in {
        "A",
        "B",
        "C",
        "D",
        "E",
    }

    print(
        f"Prediction: "
        f"{prediction:.4f}"
    )

    print(
        f"Recommendation grade: "
        f"{recommendation_grade}"
    )


# ============================================================
# TEST 13
# ============================================================

def test_similarity():

    assert OPENFOODFACTS_PATH.exists(), (
        "Open Food Facts dataset does not exist."
    )

    product = Product(
        barcode="TEST004",
        name="Test Chocolate",

        brand="Test Brand",

        categories=(
            "en:chocolates, "
            "en:confectioneries"
        ),

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

    grade_rank = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
    }

    for alternative in alternatives:

        assert "name" in alternative
        assert "health_grade" in alternative
        assert "similarity" in alternative
        assert "distance" in alternative

        grade = (
            alternative[
                "health_grade"
            ]
        )

        assert grade in grade_rank

        assert (
            grade_rank[grade]
            <
            grade_rank["E"]
        )

    print(
        f"Healthier alternatives found: "
        f"{len(alternatives)}"
    )

    for alternative in alternatives:

        print(
            f"  {alternative['name']} "
            f"({alternative['health_grade']}) "
            f"similarity="
            f"{alternative['similarity']:.4f}"
        )


# ============================================================
# TEST 14
# ============================================================

def test_end_to_end():

    assert MODEL_PATH.exists(), (
        "Trained model does not exist."
    )

    product_1 = Product(
        barcode="TEST005",
        name="Test Apple",

        brand="Test Brand",

        categories="en:fruits",

        energy_kj=218,
        fat=0.2,
        saturated_fat=0.0,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0.0,
        sodium=0.0,

        nutriscore="a",
    )

    product_2 = Product(
        barcode="TEST006",
        name="Test Chocolate",

        brand="Test Brand",

        categories=(
            "en:chocolates, "
            "en:confectioneries"
        ),

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

    candidate = Product(
        barcode="TEST007",
        name="Test Broccoli",

        brand="Test Brand",

        categories="en:vegetables",

        energy_kj=146,
        fat=0.4,
        saturated_fat=0.1,
        carbohydrates=7,
        sugars=1.7,
        fiber=2.6,
        protein=2.8,
        salt=0.0,
        sodium=0.0,

        nutriscore="a",
    )

    cart = [
        product_1,
        product_2,
    ]

    result = recommend(
        cart=cart,
        candidate_products=[
            candidate
        ],
    )

    assert isinstance(
        result,
        dict,
    )

    assert (
        "cart_health_score"
        in result
    )

    assert "add" in result
    assert "remove" in result

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    if result["add"] is not None:

        addition = result["add"]

        assert (
            "food_segment"
            in addition
        )

        assert (
            "example_product"
            in addition
        )

        assert (
            "grade"
            in addition
        )

        assert addition[
            "grade"
        ] in {
            "A",
            "B",
            "C",
            "D",
            "E",
        }

    # --------------------------------------------------------
    # REMOVE
    # --------------------------------------------------------

    if result["remove"] is not None:

        removal = result["remove"]

        assert (
            "product"
            in removal
        )

        assert (
            "grade"
            in removal
        )

        assert removal[
            "grade"
        ] in {
            "A",
            "B",
            "C",
            "D",
            "E",
        }

        assert (
            "alternatives"
            in removal
        )

        assert isinstance(
            removal[
                "alternatives"
            ],
            list,
        )

        print()
        print(
            "Removal recommendation:"
        )

        print(
            f"  Product: "
            f"{removal['product']}"
        )

        print(
            f"  Grade: "
            f"{removal['grade']}"
        )

        print(
            f"  Alternatives: "
            f"{len(removal['alternatives'])}"
        )

    print()
    print(
        "End-to-end recommendation completed."
    )

    print()
    print(
        result
    )


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def main():

    tests = [

        (
            1,
            "Backend and ML imports",
            test_imports,
        ),

        (
            2,
            "Product creation",
            test_product_creation,
        ),

        (
            3,
            "Product grading",
            test_grading,
        ),

        (
            4,
            "Cart functionality",
            test_cart,
        ),

        (
            5,
            "Training dataset",
            test_training_dataset,
        ),

        (
            6,
            "Feature generation",
            test_feature_generation,
        ),

        (
            7,
            "Random Forest training",
            test_model_training,
        ),

        (
            8,
            "Random Forest predictions",
            test_model_predictions,
        ),

        (
            9,
            "Random Forest save/load",
            test_model_save_load,
        ),

        (
            10,
            "Random Forest feature importance",
            test_feature_importance,
        ),

        (
            11,
            "Existing trained model",
            test_saved_model,
        ),

        (
            12,
            "AI recommendation predictions",
            test_ai_prediction,
        ),

        (
            13,
            "Product similarity and alternatives",
            test_similarity,
        ),

        (
            14,
            "End-to-end AI recommendation",
            test_end_to_end,
        ),
    ]

    passed = 0
    failed = 0

    print()
    print(
        "======================================================================"
    )
    print(
        "CUTC HACKATHON AI TEST SUITE"
    )
    print(
        "======================================================================"
    )
    print()

    for number, name, function in tests:

        if run_test(
            number,
            name,
            function,
        ):

            passed += 1

        else:

            failed += 1

    print(
        "======================================================================"
    )
    print(
        "FINAL TEST SUMMARY"
    )
    print(
        "======================================================================"
    )

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
            "ALL TESTS PASSED"
        )

        print(
            "The recommendation engine is ready "
            "for application integration."
        )

    else:

        print(
            "SOME TESTS FAILED"
        )

        print(
            "Fix the failed stage(s) before "
            "integrating the recommendation engine."
        )


if __name__ == "__main__":
    main()