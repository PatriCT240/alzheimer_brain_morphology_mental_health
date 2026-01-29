# modeling.py

# ============================================================================= 
# IMPORTS 
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from src.visualization import plot_feature_importance

# ============================================================================= 
# LINEAR REGRESSION MODELING 
# =============================================================================

def fit_linear_regression(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str
) -> dict:

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results_df = X_test.copy()
    results_df[target_column] = y_test.values
    results_df["prediction"] = y_pred

    return {
        "model": model,
        "r2": r2,
        "coefficients": dict(zip(feature_columns, model.coef_)),
        "results_df": results_df
    }

# ============================================================================= 
# RANDOM FOREST MODELING 
# =============================================================================

def fit_random_forest(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    test_size: float = 0.2
) -> dict:
    """
    Train a Random Forest Regressor for a single target column.
    Returns model, R², predictions DF, and feature importances DF.
    """

    # Split data
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # Results DF
    results_df = X_test.copy()
    results_df[target_column] = y_test.values
    results_df["prediction"] = y_pred

    # Feature importances DF
    fi_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)   

    return {
        "model": rf,
        "r2": r2,
        "results_df": results_df,
        "feature_importances": fi_df
    }

# ============================================================================= 
# MODEL COMPARISON UTILITIES 
# =============================================================================

def compare_models(lr_results: list[dict], rf_results: list[dict]) -> pd.DataFrame:
    """
    Compare Linear Regression and Random Forest performance (R²)
    for each severity index.

    Parameters
    ----------
    lr_results : list of dict
        Output list from the LR loop in the notebook.
        Each dict must contain:
            - "severity_index"
            - "r2"
    rf_results : list of dict
        Output list from the RF loop in the notebook.
        Each dict must contain:
            - "severity_index"
            - "r2"

    Returns
    -------
    pd.DataFrame
        Table comparing LR vs RF R² for each severity index.
    """

    # Convert lists to DataFrames
    lr_df = pd.DataFrame([
        {"severity_index": r["severity_index"], "r2_lr": r["r2"]}
        for r in lr_results
    ])

    rf_df = pd.DataFrame([
        {"severity_index": r["severity_index"], "r2_rf": r["r2"]}
        for r in rf_results
    ])

    # Merge on severity index
    merged = lr_df.merge(rf_df, on="severity_index", how="inner")

    # Compute difference
    merged["abs_diff"] = (merged["r2_rf"] - merged["r2_lr"]).abs()

    # Sort by best RF performance (or choose LR if you prefer)
    merged = merged.sort_values("r2_rf", ascending=False).reset_index(drop=True)

    return merged
