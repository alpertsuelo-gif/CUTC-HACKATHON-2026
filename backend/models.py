from dataclasses import dataclass
from typing import Optional


@dataclass
class Product:
    """
    Represents a product retrieved from Open Food Facts.

    Nutritional values are given per 100g unless otherwise specified.
    """

    barcode: str

    name: Optional[str] = None
    brand: Optional[str] = None

    # Open Food Facts Nutri-Score
    nutriscore: Optional[str] = None

    # Nutritional information per 100g
    energy_kj: Optional[float] = None
    fat: Optional[float] = None
    saturated_fat: Optional[float] = None
    carbohydrates: Optional[float] = None
    sugars: Optional[float] = None
    fiber: Optional[float] = None
    protein: Optional[float] = None
    salt: Optional[float] = None
    sodium: Optional[float] = None

    # Additional information
    ingredients: Optional[str] = None
    categories: Optional[str] = None
    countries: Optional[str] = None

    def __post_init__(self):
        """
        Round nutritional values to 2 decimal places.

        This keeps product data clean and prevents long floating-point
        values from appearing throughout the application.
        """

        nutrition_fields = [
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

        for field in nutrition_fields:
            value = getattr(self, field)

            if value is not None:
                try:
                    setattr(self, field, round(float(value), 2))
                except (TypeError, ValueError):
                    setattr(self, field, None)