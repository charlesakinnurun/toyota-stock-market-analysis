"""
Module 4: Data Preprocessing
Defines the target variable, splits the dataset, and scales features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from feature_engineering import FEATURE_COLUMNS


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable: next day's closing price.

    Args:
        df: Feature-engineered DataFrame

    Returns:
        DataFrame with an added 'Target' column (next day's Close), NaN rows dropped
    """
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)  # Tomorrow's price into today's row

    rows_before = len(df)
    df_clean = df.dropna()
    print(f"[Preprocessing] Target defined. Rows before/after final dropna: {rows_before} → {len(df_clean)}")

    return df_clean


def split_data(
    df_clean: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a sequential (time-ordered) train/test split.

    NOTE: Random shuffling is intentionally avoided to prevent data leakage
    in time-series data.

    Args:
        df_clean: Clean DataFrame with features and 'Target' column
        test_size: Fraction of data reserved for testing (default 0.20)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean["Target"]

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    print(f"[Preprocessing] Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardise features to zero mean and unit variance.

    The scaler is fitted ONLY on training data to prevent test-set leakage.

    Args:
        X_train: Training feature matrix
        X_test:  Testing feature matrix

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("[Preprocessing] Feature scaling complete (StandardScaler fitted on train set only).")
    return X_train_scaled, X_test_scaled, scaler
