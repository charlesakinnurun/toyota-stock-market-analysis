"""
Module 8: Prediction
Uses the trained model and scaler to generate a next-day price prediction
from the most recent available data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Any

from feature_engineering import FEATURE_COLUMNS


def predict_next_day(
    df:          pd.DataFrame,
    tuned_model: Any,
    scaler:      StandardScaler,
) -> float:
    """
    Predict the next trading day's closing price using the most recent features.

    The last row of the feature-engineered DataFrame (before target-shifting)
    holds the most up-to-date signals and is used as the prediction input.

    Args:
        df:          Feature-engineered DataFrame (output of feature_engineering module,
                     BEFORE target definition and final dropna)
        tuned_model: Fitted (possibly tuned) sklearn estimator
        scaler:      StandardScaler fitted on training data

    Returns:
        Predicted next-day closing price (float)
    """
    print("\n[Prediction] Preparing next-day prediction...")

    # Use the last available row (features are valid; Target may be NaN)
    last_row_features = df[FEATURE_COLUMNS].iloc[-1]

    print("[Prediction] Last available feature values:")
    print(last_row_features.to_string())

    # Scaler expects a 2-D array → reshape (1, n_features)
    last_row_scaled = scaler.transform(last_row_features.values.reshape(1, -1))

    next_day_price = tuned_model.predict(last_row_scaled)[0]

    last_actual_price = df["Close"].iloc[-1]

    print("\n[Prediction] ================================================")
    print(f"  Last actual  Close price : ${float(last_actual_price):.2f}")
    print(f"  Predicted next-day Close : ${next_day_price:.2f}")
    print("[Prediction] ================================================")

    return next_day_price
