# preprocessing.py

# ============================================================================= 
# IMPORTS 
# =============================================================================

import pandas as pd
from typing import List, Dict

# ============================================================================= 
# DATA LOADING
# =============================================================================

def read_data(file_path: str) -> pd.DataFrame:
    """
    Read CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

# ============================================================================= 
# DATA TYPE HANDLING
# =============================================================================

def convert_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> pd.DataFrame:
    """
    Convert all columns to numeric, except those in exclude.
    Automatically handles columns that can be numeric.
    """
    df = df.copy()
    exclude = exclude or []
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ============================================================================= 
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing_values(df: pd.DataFrame, numeric_strategy="median", categorical_strategy="mode") -> pd.DataFrame:
    """
    Fill missing values: numeric with median/mean, categorical with mode.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    if numeric_strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif numeric_strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

# ============================================================================= 
# DATA SAVING (NOTEBOOK-CONTROLLED)
# =============================================================================

def save_dataframe(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV."""
    df.to_csv(path, index=False)

# ============================================================================= 
# SUMMARY STATISTICS
# =============================================================================

def summary_statistics(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Return descriptive statistics for numeric columns rounded to the specified decimals.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    return df[numeric_cols].describe().round(decimals)
