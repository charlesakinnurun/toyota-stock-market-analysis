"""
Module 7: Post-Training Evaluation & Visualization
Evaluates the tuned model and visualises actual vs predicted prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Any


def evaluate_tuned_model(
    tuned_model:     Any,
    X_test_scaled:   np.ndarray,
    y_test:          pd.Series,
    best_model_name: str,
    results_df:      pd.DataFrame,
) -> np.ndarray:
    """
    Evaluate the tuned model on the test set and compare with the baseline.

    Args:
        tuned_model:     Fitted (possibly tuned) sklearn estimator
        X_test_scaled:   Scaled test features
        y_test:          True test target values
        best_model_name: Name of the best model (for labelling output)
        results_df:      Original comparison results (for baseline comparison)

    Returns:
        Array of predictions on the test set
    """
    y_pred_tuned = tuned_model.predict(X_test_scaled)

    tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    tuned_mae  = mean_absolute_error(y_test, y_pred_tuned)
    tuned_r2   = r2_score(y_test, y_pred_tuned)

    print(f"\n[Evaluation] ----- Tuned Model Final Evaluation ({best_model_name}) -----")
    print(f"  RMSE : {tuned_rmse:.4f}")
    print(f"  MAE  : {tuned_mae:.4f}")
    print(f"  R²   : {tuned_r2:.4f}")

    # Comparison against original (pre-tuning) metrics
    original = results_df.loc[best_model_name]
    print(f"\n[Evaluation] ----- Comparison with Original (Pre-Tuning) -----")
    print(f"  Original RMSE : {original['RMSE']:.4f}  →  Tuned RMSE : {tuned_rmse:.4f}")
    print(f"  Original MAE  : {original['MAE']:.4f}  →  Tuned MAE  : {tuned_mae:.4f}")
    print(f"  Original R²   : {original['R2']:.4f}  →  Tuned R²   : {tuned_r2:.4f}")

    return y_pred_tuned


def plot_actual_vs_predicted(
    y_test:          pd.Series,
    y_pred_tuned:    np.ndarray,
    best_model_name: str,
    save_path:       str = "post_training_visualization.png",
) -> None:
    """
    Plot actual vs predicted prices for the test set.

    Args:
        y_test:          True test target values (with DatetimeIndex)
        y_pred_tuned:    Model predictions on the test set
        best_model_name: Model name used in the chart title
        save_path:       File path for saving the chart
    """
    print("\n[Post-Training Visualization] Generating actual vs predicted chart...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(f"Tuned {best_model_name} — Actual vs Predicted")
    ax.plot(y_test.index, y_test,         label="Actual Price",    color="blue", alpha=0.7)
    ax.plot(y_test.index, y_pred_tuned,   label="Predicted Price", color="red",  linestyle="--")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"[Post-Training Visualization] Chart saved to '{save_path}'.")
