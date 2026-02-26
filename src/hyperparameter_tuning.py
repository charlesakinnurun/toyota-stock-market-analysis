"""
Module 6: Hyperparameter Tuning
Selects the best model and optimises its hyperparameters using GridSearchCV
with TimeSeriesSplit cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from typing import Any, Dict, Tuple


# Parameter grids for each tunable model
PARAM_GRIDS: Dict[str, Dict] = {
    "Random Forest": {
        "n_estimators":    [100, 200],
        "max_depth":       [10, 20, None],
        "min_samples_leaf": [2, 4],
    },
    "SVR (RBF Kernel)": {
        "C":     [1, 10, 100],
        "gamma": ["scale", 0.1],
    },
    "Decision Tree": {
        "max_depth":        [5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf":  [1, 2],
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "Linear Regression": {},  # No hyperparameters to tune
}


def tune_best_model(
    results_df:       pd.DataFrame,
    trained_models:   Dict[str, Any],
    X_train_scaled:   np.ndarray,
    y_train:          pd.Series,
) -> Tuple[Any, str]:
    """
    Select the best-performing model and tune its hyperparameters.

    Uses TimeSeriesSplit cross-validation to respect temporal ordering.
    If the best model has no parameters to tune, the already-fitted instance
    is returned unchanged.

    Args:
        results_df:      Comparison DataFrame sorted by RMSE (from model_training module)
        trained_models:  Dictionary of fitted model instances
        X_train_scaled:  Scaled training features
        y_train:         Training target values

    Returns:
        Tuple of (tuned_model, best_model_name)
    """
    best_model_name     = results_df.index[0]
    best_model_instance = trained_models[best_model_name]
    param_grid          = PARAM_GRIDS.get(best_model_name, {})

    print(f"\n[Hyperparameter Tuning] ----- Tuning: {best_model_name} -----")

    if not param_grid:
        print(f"[Hyperparameter Tuning] {best_model_name} has no parameters to tune. Using fitted model.")
        return best_model_instance, best_model_name

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(
        estimator=best_model_instance,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train_scaled, y_train)
    tuned_model = grid_search.best_estimator_

    print(f"[Hyperparameter Tuning] Best parameters found: {grid_search.best_params_}")
    return tuned_model, best_model_name
