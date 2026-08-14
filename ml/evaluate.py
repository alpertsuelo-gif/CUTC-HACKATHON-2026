from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from features import dataframe_to_features
from model import RecommendationModel


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


def evaluate():

    print("Loading data...")

    df = pd.read_csv(
        DATA_PATH
    )

    X = dataframe_to_features(
        df
    )

    y = pd.to_numeric(
        df["score_change"],
        errors="coerce",
    ).fillna(0)

    _, X_test, _, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    model = RecommendationModel()

    model.load(
        MODEL_PATH
    )

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
        "Random Forest Evaluation"
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


if __name__ == "__main__":
    evaluate()