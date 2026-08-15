# Ceres

Built for the CUTC 2026 Summer Hackathon by:

Ammar — Backend / ML
Asaki — Backend / API 
Yashu — Frontend

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

Ceres uses a custom machine-learning recommendation system to provide personalized suggestions based on the contents of the user's cart. Rather than simply recommending products with the highest health grades, the system considers how a potential change would affect the overall health of the cart.

The recommendation system can suggest products to add when they are predicted to improve the cart's health and nutritional balance. It can also identify products that may be worth removing when they are having a particularly negative impact on the cart. To prevent the system from making unreasonable recommendations, products graded A, B, or C are never recommended for removal; only products graded D or E are considered.

## How It Works

Ceoduct Information
   ↓
Health Scoring
   ↓
Cart Analysis
   ↓
Machine Learning Recommendations
   ↓
Improved Shopping Decisionsres follows a simple pipeline:

Barcode
   ↓
Open Food Facts
   ↓
Pr

## Architecture

Ceres is organized into three layers: the frontend, backend, and machine-learning system. This separation keeps the application modular and ensures that deterministic health grading and AI recommendations have distinct responsibilities.

### Frontend

The frontend is built with HTML, CSS, and JavaScript and provides the main shopping interface for scanning products, viewing health grades, managing the cart, and receiving recommendations. It uses a modern glassy aesthetic with translucent cards and clean visual elements to create a polished and approachable experience.

Despite the visual focus, accessibility remains a priority. Clear labels, simple navigation, large interactive elements, and consistent presentation of important information allow users to quickly understand their choices without relying solely on color or complex nutritional terminology.

The profile system allows users to customize their experience based on their nutritional preferences and goals, making recommendations more relevant to each individual while maintaining a consistent underlying health-scoring system. The frontend focuses on presentation and interaction, while nutritional calculations, cart analysis, personalization, and machine-learning logic are handled by the backend.

### Backend

The backend is built with FastAPI and connects the frontend to the product database, grading system, cart, and recommendation model.

When a product is scanned, Ceres retrieves its information from the Open Food Facts API and caches it locally. This reduces repeated API requests and makes future lookups faster.

Products are then evaluated using a deterministic health-scoring algorithm based on factors including energy, sugar, saturated fat, salt, fiber, protein, food group, and product category. The backend also manages the cart and analyzes its overall nutritional balance.

We decided to use deterministic grading for individual products while using machine learning for contextual, cart-level recommendations. This keeps health grades consistent and explainable while allowing recommendations to account for the contents of the entire cart.

### Machine Learning

Ceres uses a Random Forest model to predict which actions are most likely to improve the user's cart. The model considers features such as cart health score, cart size, candidate-product nutrition, food group, Nutri-Score, and the proposed action.

Random Forest was chosen because the recommendation problem uses structured, tabular data. It can capture non-linear relationships and interactions between nutritional and cart-level features without requiring a large or complex neural network. It is also relatively fast to train and run, making it suitable for generating recommendations during a shopping trip.

The model predicts the expected effect of actions such as adding or removing products. Ceres then selects the action with the greatest predicted improvement.

Machine learning does not determine whether an individual product is healthy. Instead, the deterministic grading system assigns the A-E grade 

To keep recommendations sensible, products graded A, B, or C are not eligible for removal recommendations; only D and E products are considered for removal.
