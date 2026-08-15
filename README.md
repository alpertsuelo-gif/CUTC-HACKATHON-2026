# Ceres

## Motivation:

Recently, there has been a societal trend towards improving one's health. Everyone is trying to live and eat healthier, yet many people don't have adequate nutritional value to make educated decisions on what they consume. Nutritional facts are complex and reading it properly takes background knowledge that most people lack. 

Ceres is meant to simplify healthy shopping, providing simple and comprehensive nutiritional information along with recommendations on how to improve your shopping habits. As one uses Ceres throughout their shopping trip, they will be informed and encouraged enough to create healthy shopping habits. 

## Features:

Ceres is meant to be used throughout the shopping process thus, has several tools to help shoppers throughout their experience.

### Scanner:

Our scanner automatically scans barcodes, with multiple input options for increased accessibility. The scanner calls upon the Open Food Facts database API to recognize millions of food items. Once an item is scanned into the system, it is stored locally onto a json file. This ensures that repeated item lookups are fast and efficient, while uncessary overhead by calling the API is removed. 

The scanner is designed to work directly within the shopping experience, allowing users to quickly evaluate products as they encounter them in a store. Once a barcode is recognized, Ceres retrieves the product's nutritional information, ingredients, categories, and other relevant data. The product is then processed through Ceres' health-scoring system before being presented to the user, allowing them to immediately understand the quality of their choice without needing to interpret a lengthy nutrition label themselves.

### Health Score:

In order to maintain accurate health scoring, we use a deterministic algorithm to assess the health of a product to ensure consistent grading. We take into account all the nutritional value provided, along with product segment and food group to create a hollistic grading scheme that can be distilled into a single grade score. 

Products are graded from A to E, with A representing the healthiest choices and E representing products that should be consumed less frequently. The algorithm considers factors such as energy, sugar, saturated fat, salt, fiber, and protein. Positive nutritional characteristics are rewarded, while excessive amounts of less desirable nutrients are penalized. The algorithm also accounts for the type of food being evaluated, preventing naturally nutrient-dense foods from being unfairly penalized.

### Cart:

The cart allows users to keep track of the products they have scanned throughout their shopping trip. Each product is stored along with its health grade and nutritional information, allowing users to evaluate their entire purchase rather than judging products individually.

Ceres also analyzes the overall balance of the cart. It calculates an aggregate health score and identifies nutritional or food-group gaps, helping users understand whether their shopping choices form a balanced diet. This shifts the focus from simply finding individual "healthy" products towards making healthier choices across the entire shopping trip.

### Recommendations:

Ceres uses a machine-learning recommendation system to provide personalized suggestions based on the contents of the user's cart. Rather than simply recommending products with the highest health grades, the system considers how a potential change would affect the overall health of the cart.

The recommendation system can suggest products to add when they are predicted to improve the cart's health and nutritional balance. It can also identify products that may be worth removing when they are having a particularly negative impact on the cart. To prevent the system from making unreasonable recommendations, products graded A, B, or C are never recommended for removal; only products graded D or E are considered.