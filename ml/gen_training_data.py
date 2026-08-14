# ml/gen_training_data.py
#
# Generates the supervised learning dataset for the
# CERES recommendation AI.
#
# Each row represents:
#
#     current cart
#          +
#     candidate action
#          ↓
#     resulting health-score change
#
# Target:
#
#     score_change
#
# The model will eventually learn to predict this value.


import csv
import random
from pathlib import Path

from ml.simulation import (
    load_products,
    generate_random_cart,
    generate_example,
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_PATH = (
    Path(__file__).parent
    / "training_data.csv"
)

MAX_DATASET_ROWS = 100_000

NUM_EXAMPLES = 20_000

MIN_CART_SIZE = 2

MAX_CART_SIZE = 10

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# GENERATE DATA
# ============================================================

def generate_training_data(
    num_examples=NUM_EXAMPLES,
    max_dataset_rows=MAX_DATASET_ROWS,
):
    """
    Generate the complete supervised ML dataset.
    """

    print(
        "Loading Open Food Facts dataset..."
    )

    products = load_products(
        max_rows=max_dataset_rows
    )

    print(
        f"Loaded {len(products)} usable products."
    )

    if not products:
        raise RuntimeError(
            "No usable products were loaded."
        )

    examples = []

    attempts = 0

    max_attempts = (
        num_examples * 3
    )

    while (
        len(examples) < num_examples
        and attempts < max_attempts
    ):

        attempts += 1

        # ----------------------------------------------------
        # Create random cart
        # ----------------------------------------------------

        cart_items = generate_random_cart(
            products,
            min_size=MIN_CART_SIZE,
            max_size=MAX_CART_SIZE,
        )

        if not cart_items:
            continue

        # ----------------------------------------------------
        # Simulate action
        # ----------------------------------------------------

        example = generate_example(
            cart_items,
            products,
        )

        if example is None:
            continue

        examples.append(
            example
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if len(examples) % 1000 == 0:

            print(
                f"Generated "
                f"{len(examples)}/"
                f"{num_examples}"
            )

    if not examples:

        raise RuntimeError(
            "No training examples were generated."
        )

    return examples


# ============================================================
# SAVE CSV
# ============================================================

def save_training_data(
    examples,
    output_path=OUTPUT_PATH,
):
    """
    Save training examples as CSV.
    """

    if not examples:
        raise ValueError(
            "No examples to save."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Union of all fields.
    fieldnames = []

    seen = set()

    for example in examples:

        for key in example:

            if key not in seen:

                seen.add(key)

                fieldnames.append(
                    key
                )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            examples
        )

    print()
    print(
        f"Saved {len(examples)} examples to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    examples = generate_training_data()

    save_training_data(
        examples
    )


if __name__ == "__main__":
    main()