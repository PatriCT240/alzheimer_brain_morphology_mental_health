# visualization.py

# ============================================================================= 
# IMPORTS 
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from typing import Dict, List, Optional

# ============================================================================= 
# GROUP MEAN PLOTS 
# =============================================================================

def plot_group_means(group_means, group_labels, 
                     title=None, xlabel=None, ylabel=None,
                     figsize=(10, 6), save_path=None):
    """
    Plot group means for multiple measures.

    Parameters
    ----------
    group_means : pd.DataFrame
        DataFrame with groups as index and measures as columns.
    title : str
        Plot title.
    xlabel : str
        Label for x-axis.
    ylabel : str
        Label for y-axis.
    figsize : tuple, optional
        Figure size (default (10, 6)).
    save_path : str, optional
        Path to save figure.
    """
    plt.figure(figsize=figsize)
    x = group_means.index
    for col in group_means.columns:
        plt.plot(x, group_means[col], marker='o', label=col)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()

# ============================================================================= 
# BAR PLOTS 
# =============================================================================

def plot_bar_side_by_side(
    df_means: pd.DataFrame, 
    group_labels: list, 
    title: str, 
    xlabel: str,
    ylabel: str, 
    save_path: None, 
    bar_width: float = 0.2, 
    figsize=(12,6)):
    """
     Plots side-by-side bar plot for each measure.
    Saves figure only if save_path is provided.
    """
    
    plt.figure(figsize=figsize)
    num_measures = df_means.shape[0]
    num_groups = df_means.shape[1]
    index = np.arange(num_measures)

    for i in range(num_groups):
        plt.bar(index + i * bar_width, df_means.iloc[:, i], bar_width, label=f'{group_labels[i]}')
        # anotacions
        for j, val in enumerate(df_means.iloc[:, i]):
            plt.annotate(round(val,2), (index[j] + i * bar_width, val),
                         ha='center', va='bottom', rotation=90)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width*(num_groups-1)/2, df_means.index, rotation=45)
    plt.legend()
    plt.tight_layout()

    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path, dpi=300)
    
    plt.show()

# ============================================================================= 
# REGRESSION PLOTS 
# =============================================================================

def plot_regression_results(
    df,
    target_column: str,
    prediction_column: str = "prediction",
    save_path: str = None,
    show: bool = True
):

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=target_column, y=prediction_column, data=df, alpha=0.6)
    plt.xlabel(f"Real value ({target_column})")
    plt.ylabel(f"Prediction ({prediction_column})")
    plt.title(f"Linear Regression – {target_column}")
    plt.tight_layout()

    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path, dpi=300)
    
    if show: 
        plt.show() 
    else: 
        plt.close()

# ============================================================================= 
# FEATURE IMPORTANCE PLOTS 
# =============================================================================

def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    target: str,
    model_name: str = "model",
    save_path: str = None,
    show: bool = True
):
    """
    Plots feature importances for a model and saves the figure as PNG.

    Args:
        feature_importance_df: DataFrame with columns ["feature", "importance"]
        target: target variable name
        model_name: name of the model (e.g., "Random Forest", "Linear Regression")
    """

    plt.figure(figsize=(12, 6))
    plt.bar(
        feature_importance_df["feature"],
        feature_importance_df["importance"],
        color="skyblue"
    )
    plt.xlabel("Brain Volume")
    plt.ylabel("Importance")
    plt.title(f"{model_name} Feature Importance for {target}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path, dpi=300)
    
    if show: 
        plt.show() 
    else: 
        plt.close()

# ============================================================================= 
# CORRELATION HEATMAPS 
# =============================================================================

def plot_correlation_matrix(
    correlation_df: pd.DataFrame, 
    title: str = "Correlation Matrix", 
    save_path: str = None
):
    """
    Plot a heatmap of the correlation matrix and save it.

    Parameters
    ----------
    correlation_df : pd.DataFrame
        DataFrame containing correlation coefficients.
    title : str
        Title of the plot.
    save_path : str
        Path to save the figure.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_p_values_matrix(
    p_values_df: pd.DataFrame, 
    title: str = "P-values Matrix", 
    save_path: str = None
):
    """
    Plot a heatmap of p-values and save it.

    Parameters
    ----------
    p_values_df : pd.DataFrame
        DataFrame containing p-values.
    title : str
        Title of the plot.
    save_path : str
        Path to save the figure.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(p_values_df.astype(float), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    
    if save_path: 
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
