from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor


class RecommendationModel:
    """
    Random Forest regression model used to predict
    the change in cart health score caused by an action.
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        """
        Train the Random Forest.
        """

        self.model.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict score changes.
        """

        return self.model.predict(X)

    def feature_importances(self):
        """
        Return Random Forest feature importance.
        """

        return self.model.feature_importances_

    def save(self, path):
        """
        Save trained model.
        """

        path = Path(path)

        joblib.dump(
            self.model,
            path,
        )

    def load(self, path):
        """
        Load trained model.
        """

        path = Path(path)

        self.model = joblib.load(path)

        return self