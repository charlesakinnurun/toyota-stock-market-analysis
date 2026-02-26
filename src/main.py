"""
main.py — Stock Price Prediction Pipeline
==========================================
Connects all pipeline modules in sequence:

    1. Data Acquisition       → download raw OHLCV data
    2. Feature Engineering    → create technical & temporal features
    3. Pre-Training Viz       → inspect price trends before modelling
    4. Preprocessing          → define target, split, and scale data
    5. Model Training         → train & compare candidate models
    6. Hyperparameter Tuning  → optimise the best model
    7. Evaluation & Viz       → assess tuned model; plot actual vs predicted
    8. Prediction             → predict the next trading day's price

Usage:
    python main.py
"""

# ── Module imports ─────────────────────────────────────────────────────────────
from data_acquisition      import download_stock_data
from feature_engineering   import engineer_features
from visualization         import plot_price_and_sma
from preprocessing         import define_target, split_data, scale_features
from model_training        import train_and_compare
from hyperparameter_tuning import tune_best_model
from evaluation            import evaluate_tuned_model, plot_actual_vs_predicted
from prediction            import predict_next_day


def run_pipeline(
    ticker:    str   = "TM",
    start:     str   = "2010-01-01",
    end:       str   = "2023-12-31",
    test_size: float = 0.20,
) -> float:
    """
    Execute the full end-to-end stock prediction pipeline.

    Args:
        ticker:    Stock ticker symbol
        start:     Data download start date (YYYY-MM-DD)
        end:       Data download end date   (YYYY-MM-DD)
        test_size: Fraction of data held out for testing

    Returns:
        Predicted next-day closing price
    """

    # ── Step 1: Data Acquisition ──────────────────────────────────────────────
    raw_data = download_stock_data(ticker=ticker, start=start, end=end)

    if raw_data.empty:
        raise RuntimeError("Data acquisition failed. Aborting pipeline.")

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    df_features = engineer_features(raw_data)

    # ── Step 3: Pre-Training Visualization ────────────────────────────────────
    plot_price_and_sma(df_features, save_path="pre_training_visualization.png")

    # ── Step 4: Preprocessing ─────────────────────────────────────────────────
    df_clean = define_target(df_features)
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=test_size)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ── Step 5: Model Training & Comparison ───────────────────────────────────
    results_df, trained_models = train_and_compare(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # ── Step 6: Hyperparameter Tuning ─────────────────────────────────────────
    tuned_model, best_model_name = tune_best_model(
        results_df, trained_models, X_train_scaled, y_train
    )

    # ── Step 7: Post-Training Evaluation & Visualization ─────────────────────
    y_pred_tuned = evaluate_tuned_model(
        tuned_model, X_test_scaled, y_test, best_model_name, results_df
    )
    plot_actual_vs_predicted(
        y_test, y_pred_tuned, best_model_name,
        save_path="post_training_visualization.png"
    )

    # ── Step 8: Next-Day Prediction ───────────────────────────────────────────
    # NOTE: We pass `df_features` (before target shifting) so the last row
    # contains valid features for the most recent trading date.
    next_day_price = predict_next_day(df_features, tuned_model, scaler)

    return next_day_price


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    predicted_price = run_pipeline()
    print(f"\nPipeline complete. Next-day predicted price: ${predicted_price:.2f}")
