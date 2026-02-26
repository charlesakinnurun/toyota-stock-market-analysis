"""
Module 2: Feature Engineering
Creates predictive features from raw stock data.
"""

import pandas as pd


# Global list of feature column names (used across modules)
FEATURE_COLUMNS = [
    'lag_1', 'lag_5', 'lag_10',
    'sma_10', 'sma_30', 'sma_60',
    'volatility_10', 'volatility_30', 'volatility_60',
    'price_change_1',
    'day_of_week', 'month', 'quarter', 'year'
]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical and temporal features from raw stock OHLCV data.

    Features created:
        - Lagged close prices (1, 5, 10 days)
        - Simple moving averages (10, 30, 60 days)
        - Rolling volatility / std deviation (10, 30, 60 days)
        - Daily price change (momentum)
        - Date-based features (day of week, month, quarter, year)

    Args:
        data: Raw OHLCV DataFrame (output of data_acquisition module)

    Returns:
        DataFrame with all original columns plus engineered features
    """
    print("[Feature Engineering] Starting feature engineering...")

    df = data.copy()

    # Feature 1: Lagged Features
    # Assumption: the price N days ago carries predictive signal
    lag_periods = [1, 5, 10]
    for lag in lag_periods:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Feature 2: Simple Moving Averages (SMA)
    # Smooths short-term fluctuations; highlights longer-term trends
    ma_windows = [10, 30, 60]
    for window in ma_windows:
        df[f"sma_{window}"] = df["Close"].rolling(window=window).mean()

    # Feature 3: Rolling Volatility (Standard Deviation)
    # Measures price variability; high volatility may be a predictive signal
    for window in ma_windows:
        df[f"volatility_{window}"] = df["Close"].rolling(window=window).std()

    # Feature 4: Price Change / Momentum
    # Daily difference in closing price
    df["price_change_1"] = df["Close"].diff(1)

    # Feature 5: Date-based Features
    # Captures seasonal and calendar patterns
    df["day_of_week"] = df.index.dayofweek  # 0 = Monday, 4 = Friday
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    # Drop rows with NaN values introduced by lagging/rolling operations
    df.dropna(inplace=True)

    print(f"[Feature Engineering] Complete. {len(df)} rows after dropping NaN values.")
    return df
