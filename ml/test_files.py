"""
CUTC Hackathon 2026
Grocery Health AI - Integration Tests

Tests:

1. Backend and ML imports
2. Product creation
3. Product grading
4. Cart functionality
5. Training dataset
6. Feature generation
7. Random Forest training
8. Random Forest prediction
9. Random Forest save/load
10. Feature importance
11. Existing trained model
12. AI recommendation prediction
13. End-to-end recommendation

Run from project root:

    python -m ml.test_files
"""

from pathlib import Path
import sys
import traceback
import tempfile

import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

ML_DIR = Path(__file__).resolve().parent
ROOT_DIR = ML_DIR.parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ML_DIR))


# ============================================================
# TEST COUNTERS
# ============================================================

passed = 0
failed = 0


def test(name, function):
    """
    Run a test and report the result.
    """

    global passed
    global failed

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    try:
        function()

        print()
        print("PASS")

        passed += 1

    except Exception as error:

        print()
        print("FAIL")
        print()
        print(f"Error: {error}")
        print()

        traceback.print_exc()

        failed += 1


# ============================================================
# TEST 1
# ============================================================

def test_imports():

    from models import Product
    from grading import grade_product
    from cart import get_cart, analyze_cart

    from features import (
        dataframe_to_features,
        training_feature_names,
    )

    from model import RecommendationModel

    assert Product is not None
    assert grade_product is not None
    assert get_cart is not None
    assert analyze_cart is not None
    assert dataframe_to_features is not None
    assert training_feature_names is not None
    assert RecommendationModel is not None

    print(
        "Backend and ML modules imported successfully."
    )


test(
    "1. Backend and ML imports",
    test_imports,
)


# ============================================================
# TEST 2
# ============================================================

def test_product():

    from models import Product

    product = Product(
        barcode="TEST001",
        name="Test Apple",
        brand="Test Brand",
        categories="en:fruits, en:apples",
        energy_kj=220,
        fat=0.3,
        saturated_fat=0.1,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0.0,
        sodium=0.0,
        nutriscore="a",
    )

    assert isinstance(product, Product)

    assert product.name == "Test Apple"

    assert product.barcode == "TEST001"

    assert isinstance(product.categories, str)

    print(
        f"Product created successfully: "
        f"{product.name}"
    )


test(
    "2. Product creation",
    test_product,
)


# ============================================================
# TEST 3
# ============================================================

def test_grading():

    from models import Product
    from grading import grade_product

    product = Product(
        barcode="TEST002",
        name="Test Broccoli",
        brand="Test Brand",
        categories="en:vegetables, en:broccoli",
        energy_kj=150,
        fat=0.4,
        saturated_fat=0.1,
        carbohydrates=7,
        sugars=1.5,
        fiber=3.3,
        protein=2.8,
        salt=0.05,
        sodium=0.02,
        nutriscore="a",
    )

    assert isinstance(product, Product)

    assert isinstance(
        product.categories,
        str,
    )

    grading = grade_product(product)

    assert grading is not None

    assert isinstance(
        grading,
        dict,
    )

    assert "score" in grading

    print(
        "Grading result:"
    )

    print(grading)


test(
    "3. Product grading",
    test_grading,
)


# ============================================================
# TEST 4
# ============================================================

def test_cart():

    from cart import get_cart, analyze_cart

    cart = get_cart()

    assert cart is not None

    print(
        "Current cart retrieved successfully."
    )

    print(
        f"Cart type: {type(cart).__name__}"
    )

    try:
        print(
            f"Cart size: {len(cart)}"
        )
    except TypeError:
        print(
            "Cart does not support len()."
        )

    result = analyze_cart()

    assert result is not None

    assert isinstance(
        result,
        dict,
    )

    print()
    print(
        "Cart analysis completed successfully."
    )

    print(result)


test(
    "4. Cart functionality",
    test_cart,
)


# ============================================================
# TEST 5
# ============================================================

def test_training_data():

    path = ML_DIR / "training_data.csv"

    assert path.exists(), (
        f"Training data not found: {path}"
    )

    df = pd.read_csv(path)

    assert len(df) > 0, (
        "Training dataset is empty."
    )

    assert "action" in df.columns

    assert "score_change" in df.columns

    actions = set(
        df["action"]
        .dropna()
        .astype(str)
    )

    required_actions = {
        "add",
        "remove",
        "replace",
    }

    missing = (
        required_actions - actions
    )

    assert not missing, (
        f"Missing actions: {missing}"
    )

    print(
        f"Dataset: {path}"
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


test(
    "5. Training dataset",
    test_training_data,
)


# ============================================================
# TEST 6
# ============================================================

def test_features():

    from features import (
        dataframe_to_features,
        training_feature_names,
    )

    path = ML_DIR / "training_data.csv"

    df = pd.read_csv(path)

    X = dataframe_to_features(df)

    expected_columns = (
        training_feature_names()
    )

    assert len(X) == len(df)

    assert list(X.columns) == (
        expected_columns
    )

    assert X.shape[1] > 0

    assert not X.isnull().any().any(), (
        "Feature matrix contains NaN."
    )

    assert np.isfinite(
        X.to_numpy()
    ).all(), (
        "Feature matrix contains infinity."
    )

    print(
        f"Feature matrix shape: {X.shape}"
    )

    print()
    print("Features:")

    for feature in X.columns:
        print(
            f"  {feature}"
        )


test(
    "6. Feature generation",
    test_features,
)


# ============================================================
# TEST 7
# ============================================================

def test_model_training():

    from sklearn.model_selection import (
        train_test_split,
    )

    from features import (
        dataframe_to_features,
    )

    from model import (
        RecommendationModel,
    )

    df = pd.read_csv(
        ML_DIR / "training_data.csv"
    )

    X = dataframe_to_features(df)

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    model = RecommendationModel(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    assert len(predictions) == len(X_test)

    assert np.isfinite(predictions).all()

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )


test(
    "7. Random Forest training",
    test_model_training,
)


# ============================================================
# TEST 8
# ============================================================

def test_model_prediction():

    from sklearn.model_selection import (
        train_test_split,
    )

    from features import (
        dataframe_to_features,
    )

    from model import (
        RecommendationModel,
    )

    df = pd.read_csv(
        ML_DIR / "training_data.csv"
    )

    X = dataframe_to_features(df)

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    model = RecommendationModel(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    sample = X_test.iloc[:10]

    predictions = model.predict(sample)

    assert len(predictions) == 10

    assert np.isfinite(predictions).all()

    print(
        "Sample predictions:"
    )

    print()

    for index, prediction in enumerate(
        predictions
    ):

        print(
            f"  Sample {index + 1}: "
            f"{prediction:.4f}"
        )


test(
    "8. Random Forest predictions",
    test_model_prediction,
)


# ============================================================
# TEST 9
# ============================================================

def test_model_save_load():

    from sklearn.model_selection import (
        train_test_split,
    )

    from features import (
        dataframe_to_features,
    )

    from model import (
        RecommendationModel,
    )

    df = pd.read_csv(
        ML_DIR / "training_data.csv"
    )

    X = dataframe_to_features(df)

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    model = RecommendationModel(
        n_estimators=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    original_predictions = model.predict(
        X_test.iloc[:10]
    )

    with tempfile.TemporaryDirectory() as directory:

        path = (
            Path(directory)
            / "test_model.pkl"
        )

        model.save(path)

        assert path.exists()

        loaded_model = (
            RecommendationModel()
        )

        loaded_model.load(path)

        loaded_predictions = (
            loaded_model.predict(
                X_test.iloc[:10]
            )
        )

    assert np.allclose(
        original_predictions,
        loaded_predictions,
    )

    print(
        "Model successfully saved."
    )

    print(
        "Model successfully loaded."
    )

    print(
        "Predictions remain identical."
    )


test(
    "9. Random Forest save/load",
    test_model_save_load,
)


# ============================================================
# TEST 10
# ============================================================

def test_feature_importance():

    from features import (
        dataframe_to_features,
    )

    from model import (
        RecommendationModel,
    )

    df = pd.read_csv(
        ML_DIR / "training_data.csv"
    )

    X = dataframe_to_features(df)

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    model = RecommendationModel(
        n_estimators=20,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X,
        y,
    )

    importances = (
        model.feature_importances()
    )

    assert len(importances) == X.shape[1]

    assert np.isfinite(importances).all()

    assert abs(
        importances.sum() - 1.0
    ) < 0.001

    ranked = sorted(
        zip(
            X.columns,
            importances,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    print(
        "Top 10 features:"
    )

    print()

    for name, importance in ranked[:10]:

        print(
            f"  {name:30s}"
            f"{importance:.4f}"
        )


test(
    "10. Random Forest feature importance",
    test_feature_importance,
)


# ============================================================
# TEST 11
# ============================================================

def test_saved_model():

    from model import (
        RecommendationModel,
    )

    model_path = (
        ML_DIR /
        "recommendation_model.pkl"
    )

    assert model_path.exists(), (
        "recommendation_model.pkl does not exist."
    )

    model = RecommendationModel()

    model.load(model_path)

    assert model.model is not None

    print(
        "Loaded trained model:"
    )

    print(
        f"  {model_path}"
    )


test(
    "11. Existing trained model",
    test_saved_model,
)


# ============================================================
# HELPER: CREATE TEST PRODUCTS
# ============================================================

def create_test_products():

    from models import Product

    apple = Product(
        barcode="TEST201",
        name="Apple",
        brand="Test",
        categories="en:fruits, en:apples",
        energy_kj=220,
        fat=0.3,
        saturated_fat=0.1,
        carbohydrates=12,
        sugars=10,
        fiber=2.4,
        protein=0.3,
        salt=0,
        sodium=0,
        nutriscore="a",
    )

    chocolate = Product(
        barcode="TEST202",
        name="Chocolate",
        brand="Test",
        categories="en:confectioneries, en:chocolate",
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

    broccoli = Product(
        barcode="TEST203",
        name="Broccoli",
        brand="Test",
        categories="en:vegetables, en:broccoli",
        energy_kj=150,
        fat=0.4,
        saturated_fat=0.1,
        carbohydrates=7,
        sugars=1.5,
        fiber=3.3,
        protein=2.8,
        salt=0.05,
        sodium=0.02,
        nutriscore="a",
    )

    bread = Product(
        barcode="TEST204",
        name="White Bread",
        brand="Test",
        categories="en:breads, en:bread",
        energy_kj=1050,
        fat=2,
        saturated_fat=0.5,
        carbohydrates=50,
        sugars=4,
        fiber=2,
        protein=9,
        salt=1.0,
        sodium=0.4,
        nutriscore="c",
    )

    lentils = Product(
        barcode="TEST205",
        name="Lentils",
        brand="Test",
        categories="en:legumes, en:lentils",
        energy_kj=480,
        fat=0.4,
        saturated_fat=0.1,
        carbohydrates=20,
        sugars=1,
        fiber=8,
        protein=9,
        salt=0.02,
        sodium=0.01,
        nutriscore="a",
    )

    return (
        apple,
        chocolate,
        broccoli,
        bread,
        lentils,
    )


# ============================================================
# TEST 12
# ============================================================

def test_ai_prediction():

    model_path = (
        ML_DIR /
        "recommendation_model.pkl"
    )

    assert model_path.exists(), (
        "Trained model does not exist."
    )

    from recommendations_ai import (
        normalize_cart,
        predict,
    )

    (
        apple,
        chocolate,
        broccoli,
        bread,
        lentils,
    ) = create_test_products()

    cart = [
        apple,
        chocolate,
    ]

    # Make sure categories have the format
    # expected by backend/grading.py.

    for product in cart:

        assert isinstance(
            product.categories,
            str,
        )

    cart_items = normalize_cart(cart)

    assert cart_items is not None

    assert len(cart_items) == 2

    # --------------------------------------------------------
    # Removal
    # --------------------------------------------------------

    removal_prediction = predict(
        cart_items,
        chocolate,
        "remove",
    )

    assert isinstance(
        removal_prediction,
        float,
    )

    assert np.isfinite(
        removal_prediction
    )

    print(
        "Removal prediction:"
    )

    print(
        f"  Chocolate -> "
        f"{removal_prediction:.4f}"
    )

    # --------------------------------------------------------
    # Addition
    # --------------------------------------------------------

    addition_prediction = predict(
        cart_items,
        broccoli,
        "add",
    )

    assert isinstance(
        addition_prediction,
        float,
    )

    assert np.isfinite(
        addition_prediction
    )

    print(
        "Addition prediction:"
    )

    print(
        f"  Broccoli -> "
        f"{addition_prediction:.4f}"
    )


test(
    "12. AI recommendation predictions",
    test_ai_prediction,
)


# ============================================================
# TEST 13
# ============================================================

def test_end_to_end():

    model_path = (
        ML_DIR /
        "recommendation_model.pkl"
    )

    assert model_path.exists(), (
        "Trained model does not exist."
    )

    from recommendations_ai import (
        recommend,
    )

    (
        apple,
        chocolate,
        broccoli,
        bread,
        lentils,
    ) = create_test_products()

    # --------------------------------------------------------
    # Cart
    # --------------------------------------------------------

    cart = [
        apple,
        chocolate,
        bread,
    ]

    # --------------------------------------------------------
    # Candidate products
    # --------------------------------------------------------

    candidates = [
        broccoli,
        lentils,
    ]

    # --------------------------------------------------------
    # Validate product format
    # --------------------------------------------------------

    for product in cart + candidates:

        assert isinstance(
            product.categories,
            str,
        )

    # --------------------------------------------------------
    # Run recommendation engine
    # --------------------------------------------------------

    result = recommend(
        cart=cart,
        candidate_products=candidates,
    )

    assert result is not None

    assert isinstance(
        result,
        dict,
    )

    assert "cart_health_score" in result

    assert "add" in result

    assert "remove" in result

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print(
        "FINAL AI RECOMMENDATION"
    )

    print(
        "------------------------"
    )

    print(result)

    print()

    # --------------------------------------------------------
    # Addition
    # --------------------------------------------------------

    if result["add"] is not None:

        print(
            "Recommended addition:"
        )

        print(
            f"  Segment: "
            f"{result['add']['food_segment']}"
        )

        print(
            f"  Example: "
            f"{result['add']['example_product']}"
        )

        print(
            f"  Predicted change: "
            f"{result['add']['predicted_score_change']}"
        )

    else:

        print(
            "No addition recommendation."
        )

    print()

    # --------------------------------------------------------
    # Removal
    # --------------------------------------------------------

    if result["remove"] is not None:

        print(
            "Recommended removal:"
        )

        print(
            f"  Product: "
            f"{result['remove']['product']}"
        )

        print(
            f"  Segment: "
            f"{result['remove']['food_segment']}"
        )

        print(
            f"  Predicted change: "
            f"{result['remove']['predicted_score_change']}"
        )

    else:

        print(
            "No removal recommendation."
        )


test(
    "13. End-to-end AI recommendation",
    test_end_to_end,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)

print()

print(
    f"Passed: {passed}"
)

print(
    f"Failed: {failed}"
)

print(
    f"Total:  {passed + failed}"
)

print()

if failed == 0:

    print(
        "ALL TESTS PASSED"
    )

    print()
    print(
        "Backend, model, and recommendation "
        "pipeline are functioning."
    )

else:

    print(
        "SOME TESTS FAILED"
    )

    print()
    print(
        "Fix the failed stage(s) before continuing."
    )

    sys.exit(1)