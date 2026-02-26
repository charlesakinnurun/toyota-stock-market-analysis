"""
Module 5: Model Training & Comparison
Trains multiple regression models and compares their performance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple


def get_candidate_models() -> Dict:
    """
    Return a dictionary of candidate regression models with sensible defaults.

    Returns:
        Dictionary mapping model name → instantiated sklearn estimator
    """
    return {
        "Linear Regression": LinearRegression(),
        "Lasso":             Lasso(alpha=0.1, random_state=42),
        "Ridge":             Ridge(alpha=1.0, random_state=42),
        "SVR (RBF Kernel)":  SVR(kernel="rbf", C=1.0),
        "Decision Tree":     DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a set of predictions.

    Args:
        y_true: Ground-truth target values
        y_pred: Model predictions

    Returns:
        Dictionary with keys 'RMSE', 'MAE', 'R2'
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
    }


def train_and_compare(
    X_train_scaled: np.ndarray,
    X_test_scaled:  np.ndarray,
    y_train:        pd.Series,
    y_test:         pd.Series,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Train all candidate models and rank them by RMSE.

    Args:
        X_train_scaled: Scaled training features
        X_test_scaled:  Scaled test features
        y_train:        Training target values
        y_test:         Test target values

    Returns:
        Tuple of:
            - results_df: DataFrame of metrics sorted by RMSE (ascending)
            - trained_models: Dictionary mapping name → fitted model instance
    """
    print("\n[Model Training] ----- Starting Model Comparison -----")

    models = get_candidate_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"[Model Training] Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        results[name]       = evaluate_model(y_test, y_pred)
        trained_models[name] = model

    results_df = pd.DataFrame(results).T.sort_values(by="RMSE")

    print("\n[Model Training] ----- Initial Model Comparison Results -----")
    print(results_df.to_string())

    return results_df, trained_models
