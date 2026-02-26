"""
Module 3: Pre-Training Visualization
Visualizes raw stock data and engineered features before modeling.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_and_sma(df: pd.DataFrame, save_path: str = "pre_training_visualization.png") -> None:
    """
    Plot the closing price alongside the 30-day Simple Moving Average.

    Args:
        df: DataFrame containing 'Close' and 'sma_30' columns
        save_path: File path to save the plot image
    """
    print("[Pre-Training Visualization] Generating price and SMA chart...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Toyota (TM) Closing Price & 30-Day SMA (Before Modeling)")
    ax.plot(df.index, df["Close"],  label="Closing Price", color="blue", alpha=0.8)
    ax.plot(df.index, df["sma_30"], label="30-Day Moving Average", color="orange", linestyle="--")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"[Pre-Training Visualization] Chart saved to '{save_path}'.")
