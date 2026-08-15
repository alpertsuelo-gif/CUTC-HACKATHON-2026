from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parent

PROJECT_DIR = ML_DIR.parent

DEFAULT_DATASET = (
    ML_DIR /
    "en.openfoodfacts.org.products.csv.gz"
)


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 10_000

MAX_CANDIDATES = 500

MIN_SIMILARITY = 0.50


# ============================================================
# NUTRITIONAL FEATURES
# ============================================================

NUTRITION_FEATURES = [
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


# ============================================================
# DATASET COLUMNS
# ============================================================

DATASET_COLUMNS = [
    "code",
    "product_name",
    "brands",
    "categories",

    "energy-kj_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",

    "nutriscore_grade",
]


# ============================================================
# DATA TYPES
# ============================================================

TEXT_COLUMNS = {
    "code": "string",
    "product_name": "string",
    "brands": "string",
    "categories": "string",
    "nutriscore_grade": "string",
}


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_number(value):

    try:

        value = float(value)

        if not math.isfinite(value):
            return 0.0

        return value

    except (
        TypeError,
        ValueError,
    ):

        return 0.0


# ============================================================
# HEALTH GRADE
# ============================================================

def health_grade(product):

    if isinstance(
        product,
        dict,
    ):

        grade = product.get(
            "nutriscore_grade"
        )

        if grade is None:

            grade = product.get(
                "health_grade"
            )

    else:

        grade = getattr(
            product,
            "nutriscore",
            None,
        )

    if grade is None:
        return "E"

    grade = str(
        grade
    ).strip().lower()

    if grade not in {
        "a",
        "b",
        "c",
        "d",
        "e",
    }:

        return "E"

    return grade.upper()


# ============================================================
# HEALTH RANK
# ============================================================

def health_rank(
    grade,
):

    ranks = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
    }

    return ranks.get(
        str(grade).upper(),
        4,
    )


# ============================================================
# PRODUCT → VECTOR
# ============================================================

def product_to_vector(
    product,
):

    if isinstance(
        product,
        dict,
    ):

        values = [

            safe_number(
                product.get(
                    "energy_kj",
                    product.get(
                        "energy-kj_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "fat",
                    product.get(
                        "fat_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "saturated_fat",
                    product.get(
                        "saturated-fat_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "carbohydrates",
                    product.get(
                        "carbohydrates_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "sugars",
                    product.get(
                        "sugars_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "fiber",
                    product.get(
                        "fiber_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "protein",
                    product.get(
                        "proteins_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "salt",
                    product.get(
                        "salt_100g",
                        0,
                    ),
                )
            ),

            safe_number(
                product.get(
                    "sodium",
                    product.get(
                        "sodium_100g",
                        0,
                    ),
                )
            ),
        ]

        return np.array(
            values,
            dtype=float,
        )

    return np.array(
        [
            safe_number(
                getattr(
                    product,
                    "energy_kj",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "fat",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "saturated_fat",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "carbohydrates",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "sugars",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "fiber",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "protein",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "salt",
                    0,
                )
            ),

            safe_number(
                getattr(
                    product,
                    "sodium",
                    0,
                )
            ),
        ],
        dtype=float,
    )


# ============================================================
# NORMALIZE VECTOR
# ============================================================

def normalize_vector(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    scales = np.array(
        [
            4000.0,
            100.0,
            50.0,
            100.0,
            100.0,
            30.0,
            50.0,
            10.0,
            4.0,
        ],
        dtype=float,
    )

    normalized = (
        vector / scales
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(
    vector_a,
    vector_b,
):

    vector_a = normalize_vector(
        vector_a
    )

    vector_b = normalize_vector(
        vector_b
    )

    return float(
        np.linalg.norm(
            vector_a - vector_b
        )
    )


# ============================================================
# DISTANCE → SIMILARITY
# ============================================================

def distance_to_similarity(
    distance,
):

    distance = max(
        0.0,
        safe_number(
            distance
        ),
    )

    return float(
        1.0 /
        (
            1.0 +
            distance
        )
    )


# ============================================================
# DATASET READER
# ============================================================

def load_product_database(
    dataset_path=DEFAULT_DATASET,
    max_rows=10_000,
):

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Open Food Facts dataset does not exist: "
            f"{dataset_path}"
        )

    return pd.read_csv(
        dataset_path,
        compression="gzip",
        sep="\t",
        usecols=lambda column:
            column in DATASET_COLUMNS,
        dtype=TEXT_COLUMNS,
        nrows=max_rows,
        low_memory=True,
    )


# ============================================================
# DATASET ROW → DICTIONARY
# ============================================================

def row_to_product_dict(
    row,
):

    return {
        "barcode":
            str(
                row.get(
                    "code",
                    "",
                )
            ),

        "name":
            str(
                row.get(
                    "product_name",
                    "",
                )
            ),

        "brand":
            str(
                row.get(
                    "brands",
                    "",
                )
            ),

        "categories":
            str(
                row.get(
                    "categories",
                    "",
                )
            ),

        "energy_kj":
            safe_number(
                row.get(
                    "energy-kj_100g",
                    0,
                )
            ),

        "fat":
            safe_number(
                row.get(
                    "fat_100g",
                    0,
                )
            ),

        "saturated_fat":
            safe_number(
                row.get(
                    "saturated-fat_100g",
                    0,
                )
            ),

        "carbohydrates":
            safe_number(
                row.get(
                    "carbohydrates_100g",
                    0,
                )
            ),

        "sugars":
            safe_number(
                row.get(
                    "sugars_100g",
                    0,
                )
            ),

        "fiber":
            safe_number(
                row.get(
                    "fiber_100g",
                    0,
                )
            ),

        "protein":
            safe_number(
                row.get(
                    "proteins_100g",
                    0,
                )
            ),

        "salt":
            safe_number(
                row.get(
                    "salt_100g",
                    0,
                )
            ),

        "sodium":
            safe_number(
                row.get(
                    "sodium_100g",
                    0,
                )
            ),

        "health_grade":
            health_grade(
                {
                    "nutriscore_grade":
                        row.get(
                            "nutriscore_grade"
                        )
                }
            ),
    }


# ============================================================
# CATEGORY SIMILARITY
# ============================================================

def category_similarity(
    original,
    candidate,
):

    original_categories = str(
        getattr(
            original,
            "categories",
            "",
        )
        or ""
    ).lower()

    candidate_categories = str(
        candidate.get(
            "categories",
            "",
        )
        or ""
    ).lower()

    if not original_categories:
        return 0.0

    if not candidate_categories:
        return 0.0

    original_words = set(
        word.strip()
        for word in
        original_categories.split(",")
        if word.strip()
    )

    candidate_words = set(
        word.strip()
        for word in
        candidate_categories.split(",")
        if word.strip()
    )

    if not original_words:
        return 0.0

    union = (
        original_words
        |
        candidate_words
    )

    if not union:
        return 0.0

    intersection = (
        original_words
        &
        candidate_words
    )

    return (
        len(intersection)
        /
        len(union)
    )


# ============================================================
# HEALTHIER ALTERNATIVES
# ============================================================

def find_healthier_alternatives(
    product,
    limit=5,
    dataset_path=DEFAULT_DATASET,
    chunk_size=CHUNK_SIZE,
):

    dataset_path = Path(
        dataset_path
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Open Food Facts dataset does not exist: "
            f"{dataset_path}"
        )

    limit = max(
        1,
        int(limit),
    )

    original_grade = (
        health_grade(
            product
        )
    )

    original_rank = (
        health_rank(
            original_grade
        )
    )

    original_vector = (
        product_to_vector(
            product
        )
    )

    original_normalized = (
        normalize_vector(
            original_vector
        )
    )

    original_barcode = str(
        getattr(
            product,
            "barcode",
            "",
        )
        or ""
    )

    candidates = []

    # ========================================================
    # CHUNKED DATASET PROCESSING
    # ========================================================

    reader = pd.read_csv(
        dataset_path,
        compression="gzip",
        sep="\t",
        usecols=lambda column:
            column in DATASET_COLUMNS,
        dtype=TEXT_COLUMNS,
        chunksize=chunk_size,
        low_memory=True,
    )

    for chunk in reader:

        # ----------------------------------------------------
        # Product names
        # ----------------------------------------------------

        if (
            "product_name"
            not in chunk.columns
        ):
            continue

        chunk = chunk[
            chunk[
                "product_name"
            ].notna()
        ]

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # Nutritional columns
        # ----------------------------------------------------

        numeric_columns = [
            "energy-kj_100g",
            "fat_100g",
            "saturated-fat_100g",
            "carbohydrates_100g",
            "sugars_100g",
            "fiber_100g",
            "proteins_100g",
            "salt_100g",
            "sodium_100g",
        ]

        for column in numeric_columns:

            if column in chunk.columns:

                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Require enough nutritional information
        # ----------------------------------------------------

        nutrition_available = (
            chunk[
                numeric_columns
            ].notna().sum(axis=1)
        )

        chunk = chunk[
            nutrition_available >= 5
        ]

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # Health grades
        # ----------------------------------------------------

        if (
            "nutriscore_grade"
            not in chunk.columns
        ):
            continue

        grades = (
            chunk[
                "nutriscore_grade"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        grade_ranks = (
            grades.map(
                {
                    "A": 0,
                    "B": 1,
                    "C": 2,
                    "D": 3,
                    "E": 4,
                }
            )
        )

        # Keep only products with a valid Nutri-Score.
        valid_grade = (
            grade_ranks.notna()
        )

        chunk = chunk[
            valid_grade
        ]

        grade_ranks = (
            grade_ranks.loc[
                chunk.index
            ]
        )

        # ----------------------------------------------------
        # Only healthier products
        # ----------------------------------------------------

        chunk = chunk[
            grade_ranks
            < original_rank
        ]

        if chunk.empty:
            continue

        # ====================================================
        # CALCULATE SIMILARITY
        # ====================================================

        for _, row in chunk.iterrows():

            candidate = (
                row_to_product_dict(
                    row
                )
            )

            # ------------------------------------------------
            # Don't recommend same product
            # ------------------------------------------------

            candidate_barcode = str(
                candidate.get(
                    "barcode",
                    "",
                )
                or ""
            )

            if (
                original_barcode
                and
                candidate_barcode
                and
                candidate_barcode
                ==
                original_barcode
            ):
                continue

            # ------------------------------------------------
            # Nutritional vector
            # ------------------------------------------------

            candidate_vector = (
                product_to_vector(
                    candidate
                )
            )

            candidate_normalized = (
                normalize_vector(
                    candidate_vector
                )
            )

            # ------------------------------------------------
            # Euclidean distance
            # ------------------------------------------------

            distance = float(
                np.linalg.norm(
                    original_normalized
                    -
                    candidate_normalized
                )
            )

            similarity = (
                distance_to_similarity(
                    distance
                )
            )

            if (
                similarity
                <
                MIN_SIMILARITY
            ):
                continue

            # ------------------------------------------------
            # Category similarity
            # ------------------------------------------------

            category_score = (
                category_similarity(
                    product,
                    candidate,
                )
            )

            # ------------------------------------------------
            # Combined score
            # ------------------------------------------------

            ranking_score = (
                similarity * 0.80
                +
                category_score * 0.20
            )

            candidate[
                "distance"
            ] = round(
                distance,
                6,
            )

            candidate[
                "similarity"
            ] = round(
                similarity,
                6,
            )

            candidate[
                "category_similarity"
            ] = round(
                category_score,
                6,
            )

            candidate[
                "ranking_score"
            ] = round(
                ranking_score,
                6,
            )

            candidates.append(
                candidate
            )

        # ----------------------------------------------------
        # Keep memory bounded
        # ----------------------------------------------------

        if len(candidates) > (
            MAX_CANDIDATES
        ):

            candidates.sort(
                key=lambda item:
                    item[
                        "ranking_score"
                    ],
                reverse=True,
            )

            candidates = candidates[
                :MAX_CANDIDATES
            ]

    # ========================================================
    # FINAL SORT
    # ========================================================

    candidates.sort(
        key=lambda item:
            item[
                "ranking_score"
            ],
        reverse=True,
    )

    return candidates[
        :limit
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Testing product similarity..."
    )

    print()

    print(
        "Dataset:"
    )

    print(
        DEFAULT_DATASET
    )

    print()

    print(
        "Loading a small sample..."
    )

    df = load_product_database(
        max_rows=1000
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    print()

    print(
        "Columns loaded:"
    )

    for column in df.columns:

        print(
            f"  {column}"
        )

    print()

    print(
        "Product similarity module loaded successfully."
    )