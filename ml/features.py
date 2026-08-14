import pandas as pd


FOOD_GROUPS = [
    "fruit",
    "vegetable",
    "legume",
    "whole_grain",
    "cereal",
    "bread",
    "dairy",
    "meat",
    "fish",
    "nuts",
    "beverage",
    "snack",
    "confectionery",
    "dessert",
    "sauce",
    "unknown",
]


NUTRISCORES = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "unknown",
]


ACTIONS = [
    "add",
    "remove",
    "replace",
]


NUMERIC_FEATURES = [
    "cart_size",
    "cart_health_score",
    "product_score",
    "energy_kj",
    "fat",
    "saturated_fat",
    "carbohydrates",
    "sugars",
    "fiber",
    "protein",
    "salt",
    "sodium",
]


def encode_action(action):
    """
    One-hot encode an action.
    """

    return [
        1.0 if action == candidate else 0.0
        for candidate in ACTIONS
    ]


def encode_food_group(food_group):
    """
    One-hot encode a food group.
    """

    food_group = str(
        food_group or "unknown"
    ).lower()

    if food_group not in FOOD_GROUPS:
        food_group = "unknown"

    return [
        1.0 if food_group == group else 0.0
        for group in FOOD_GROUPS
    ]


def encode_nutriscore(nutriscore):
    """
    One-hot encode Nutri-Score.
    """

    nutriscore = str(
        nutriscore or "unknown"
    ).lower()

    if nutriscore not in NUTRISCORES:
        nutriscore = "unknown"

    return [
        1.0 if nutriscore == grade else 0.0
        for grade in NUTRISCORES
    ]


def training_feature_names():
    """
    Return feature names in exactly the order
    used by the model.
    """

    names = list(NUMERIC_FEATURES)

    names += [
        f"action_{action}"
        for action in ACTIONS
    ]

    names += [
        f"food_group_{group}"
        for group in FOOD_GROUPS
    ]

    names += [
        f"nutriscore_{grade}"
        for grade in NUTRISCORES
    ]

    return names


def dataframe_to_features(df):
    """
    Convert training dataframe into model features.
    """

    rows = []

    for _, row in df.iterrows():

        features = []

        # Numerical features
        for column in NUMERIC_FEATURES:

            value = pd.to_numeric(
                row.get(column, 0),
                errors="coerce",
            )

            if pd.isna(value):
                value = 0.0

            features.append(float(value))

        # Action
        features.extend(
            encode_action(
                row.get("action")
            )
        )

        # Food group
        features.extend(
            encode_food_group(
                row.get("food_group")
            )
        )

        # Nutri-score
        features.extend(
            encode_nutriscore(
                row.get("nutriscore")
            )
        )

        rows.append(features)

    return pd.DataFrame(
        rows,
        columns=training_feature_names(),
    )