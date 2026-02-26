"""
Module 1: Data Acquisition
Handles downloading and loading stock data.
"""

import yfinance as yf
import pandas as pd


def download_stock_data(ticker: str = "TM", start: str = "2010-01-01", end: str = "2023-12-31") -> pd.DataFrame:
    """
    Download historical stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (default: 'TM' for Toyota Motor Corp)
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format

    Returns:
        DataFrame with OHLCV stock data, or empty DataFrame on failure
    """
    print(f"[Data Acquisition] Downloading {ticker} data from {start} to {end}...")

    try:
        data = yf.download(ticker, start=start, end=end)

        if data.empty:
            print("[Data Acquisition] ERROR: No data downloaded. Check ticker symbol or date range.")
            return pd.DataFrame()

        print(f"[Data Acquisition] Successfully downloaded {len(data)} rows of data.")
        return data

    except Exception as e:
        print(f"[Data Acquisition] ERROR downloading data: {e}")
        return pd.DataFrame()
