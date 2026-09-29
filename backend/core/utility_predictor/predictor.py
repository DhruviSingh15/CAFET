"""
Utility Predictor — Phase 7 core module.

Model: RandomForestRegressor
  - Interpretable, handles mixed features, supports prediction variance
  - Uncertainty = std of individual tree predictions (not calibrated probability)

Utility definition:
    utility = candidate_validation_score - baseline_validation_score
    (NEVER test score)

LODO protocol enforced externally by caller.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor

# Fixed hyperparameters — not tuned for SOTA, chosen for mechanism validation
RF_CONFIG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


class UtilityPredictor:
    def __init__(self, config: Optional[Dict] = None):
        cfg = config or RF_CONFIG
        self.model = RandomForestRegressor(**cfg)
        self.config = cfg
        self._is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on (candidate_representation, observed_validation_gain) pairs."""
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            predicted_utility : mean across trees
            uncertainty       : std across trees (predictive spread, NOT calibrated)
        """
        if not self._is_trained:
            raise RuntimeError("Predictor must be fitted before prediction.")

        tree_preds = np.stack(
            [tree.predict(X) for tree in self.model.estimators_], axis=0
        )  # (n_trees, n_samples)

        predicted = tree_preds.mean(axis=0)
        uncertainty = tree_preds.std(axis=0)
        return predicted, uncertainty

    def feature_importances(self) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Not fitted.")
        return self.model.feature_importances_
