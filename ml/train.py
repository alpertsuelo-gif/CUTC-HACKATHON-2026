from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.features import (
    dataframe_to_features,
)

from ml.model import RecommendationModel


BASE_DIR = Path(
    __file__
).resolve().parent

DATA_PATH = (
    BASE_DIR /
    "training_data.csv"
)

MODEL_PATH = (
    BASE_DIR /
    "recommendation_model.pkl"
)


def train():

    print("Loading training data...")

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    if "score_change" not in df.columns:
        raise ValueError(
            "training_data.csv must contain "
            "'score_change'."
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = dataframe_to_features(
        df
    )

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = RecommendationModel(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print()
    print(
        "MODEL PERFORMANCE"
    )
    print(
        "-----------------"
    )

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print()
    print(
        "TOP FEATURES"
    )
    print(
        "------------"
    )

    importances = (
        model.feature_importances()
    )

    feature_importance = sorted(
        zip(
            X.columns,
            importances,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, importance in (
        feature_importance[:15]
    ):

        print(
            f"{name:30s} "
            f"{importance:.4f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(
        MODEL_PATH
    )

    print()
    print(
        f"Model saved to:"
    )

    print(
        MODEL_PATH
    )


if __name__ == "__main__":
    train()