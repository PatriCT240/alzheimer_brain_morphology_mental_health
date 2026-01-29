# analysis.py

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from src.config import brain_cols, severity_cols, activity_cols
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def pearson_correlation(
    df: pd.DataFrame, 
    cols: List[str]
) -> pd.DataFrame:
    """
    Calculate Pearson correlation between all pairs in cols.
    Returns a DataFrame with correlations.
    """
    return df[cols].corr(method='pearson')

def compute_correlation_matrix(
    df: pd.DataFrame, 
    volume_columns: list, 
    severity_columns: list
):
    """
    Compute Pearson correlation coefficients between volume and severity columns,
    and save the resulting matrices to CSV files in reports/tables.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the columns of interest.
    volume_columns : list
        List of brain volume column names.
    severity_columns : list
        List of symptom severity column names.

    Returns
    -------
    correlation_df : pd.DataFrame
        DataFrame of correlation coefficients (volume x severity).
    p_values_df : pd.DataFrame
        DataFrame of corresponding p-values.
    """
    # Initialize matrices
    num_vol = len(volume_columns)
    num_sev = len(severity_columns)
    
    corr_matrix = np.zeros((num_vol, num_sev))
    p_values_matrix = np.zeros((num_vol, num_sev))
    
    # Compute correlations and p-values
    for i, vol_col in enumerate(volume_columns):
        for j, sev_col in enumerate(severity_columns):
            r, p = stats.pearsonr(df[vol_col], df[sev_col])
            corr_matrix[i, j] = r
            p_values_matrix[i, j] = p
    
    # Convert to DataFrame with meaningful indices/columns
    correlation_df = pd.DataFrame(corr_matrix, index=volume_columns, columns=severity_columns)
    p_values_df = pd.DataFrame(p_values_matrix, index=volume_columns, columns=severity_columns)
        
    return correlation_df, p_values_df

# =============================================================================
# GROUP STATISTICS
# =============================================================================
    
def group_means(
    df: pd.DataFrame, 
    group_col: str, 
    value_cols: List[str]
) -> pd.DataFrame:
    """
    Calculate mean of value_cols grouped by group_col.
    Returns a DataFrame with group means.
    """
    return df.groupby(group_col)[value_cols].mean()

def compute_group_means_for_plot(
    df: pd.DataFrame, 
    group_col: str, 
    measure_cols: list
) -> pd.DataFrame:
    """
    Compute group means for the specified measures.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str
        Column name for grouping (categorical).
    measure_cols : list
        List of measure columns to compute means.

    Returns
    -------
    pd.DataFrame
        DataFrame with groups as index and measures as columns.
    """
    df_means = df.groupby(group_col)[measure_cols].mean()
    return df_means

def compute_group_means(
    df: pd.DataFrame, 
    group_col: str, 
    value_cols: list
) -> dict:
    """
    Compute mean values per group for each column in value_cols.
    Returns a dict {column: [mean_per_group]}.
    """
    means_dict = {}
    for col in value_cols:
        means_dict[col] = df.groupby(group_col)[col].mean().tolist()
    return means_dict

# =============================================================================
# SEVERITY AND COLUMN DETECTION UTILITIES
# =============================================================================

def severity_analysis(df: pd.DataFrame) -> List[str]:
    """
    Returns a list of severity index columns in the DataFrame.

    Professional approach:
    - Automatically detects columns ending with 'sev'.
    - Avoids manual hardcoding.
    """
    return [col for col in df.columns if col.endswith('sev')]

def daily_activity_columns(df: pd.DataFrame) -> list:
    """
    Return the list of columns corresponding to daily activities present in df.
    """
    return [col for col in activity_cols if col in df.columns]

def brain_volume_columns(df: pd.DataFrame) -> list:
    """
    Return a list of brain volume columns present in the DataFrame.
    Automatically detects common volume-related columns.
    """
    return [col for col in df.columns if any(k in col for k in brain_cols)]

# =============================================================================
# ANOVA ANALYSIS
# =============================================================================   

def anova(
    df: pd.DataFrame, 
    dependent_columns: list, 
    covariates: list
):
    """
    One-way or multi-factor ANOVA using correct group-based sums of squares.
    Assumes covariates are categorical.
    """
    results = {}

    for dep in dependent_columns:
        y = df[dep]

        # Combined groups
        groups = df.groupby(covariates)[dep]

        # Means
        grand_mean = y.mean()
        group_means = groups.mean()
        group_sizes = groups.size()

        # Sum of Squares
        ss_between = ((group_means - grand_mean) ** 2 * group_sizes).sum()
        ss_within = groups.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()

        # Degrees of freedom
        df_between = group_means.shape[0] - 1
        df_within = len(y) - group_means.shape[0]

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within

        f_statistic = ms_between / ms_within
        p_value = stats.f.sf(f_statistic, df_between, df_within)

        results[dep] = {
            "F_statistic": f_statistic,
            "p_value": p_value
        }

    return results

# =============================================================================
# TUKEY POST-HOC ANALYSIS
# =============================================================================

def tukey(
    df: pd.DataFrame, 
    group_col: str, 
    measure_cols: list
):
    """
    Performs Tukey's HSD post-hoc test for multiple dependent variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str
        Independent grouping variable.
    measure_cols : list
        List of dependent variables.

    Returns
    -------
    results : dict
        Keys = measure names, values = TukeyHSD results objects.
    """
    
    results = {}
    for col in measure_cols:
        data = df[col].to_numpy()
        labels = df[group_col].astype(str).to_numpy()
        tukey_result = pairwise_tukeyhsd(data, labels)
        results[col] = tukey_result
       
    return results

def tukey_significant_summary(
    df: pd.DataFrame,
    group_col: str,
    measure_cols: list,
    variable_type: str = "summary"
) -> pd.DataFrame:
    """
    Compute Tukey post-hoc tests and return a DataFrame with only
    statistically significant comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    group_col : str
        Grouping column.
    measure_cols : list
        List of measures to test.

    Returns
    -------
    df_significant : pd.DataFrame
        One row per significant comparison across all measures.
    summary_info : dict
        Info about number of significant comparisons and top 5 rows.
    """
    tukey_results = tukey(df, group_col, measure_cols)

    rows = []

    for measure, tukey_res in tukey_results.items():
        summary = tukey_res.summary()
        df_summary = pd.DataFrame(
            summary.data[1:],
            columns=summary.data[0]
        )

        # Keep only significant comparisons
        df_summary = df_summary[df_summary["reject"] == True].copy()
        df_summary["measure"] = measure

        rows.append(df_summary)

    if not rows:
        df_significant = pd.DataFrame()
        summary_info = {
            "n_significant": 0,
            "top_5": df_significant
        }
    else:
        df_significant = pd.concat(rows, ignore_index=True)

        df_significant.sort_values(
            by=["measure", "p-adj"],
            inplace=True
        )

        summary_info = {
            "n_significant": len(df_significant),
            "top_5": df_significant.head(5)
        }

    return df_significant, summary_info

# =============================================================================
# CHI-SQUARE ANALYSIS
# =============================================================================

def run_chi_square_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str
) -> List[Tuple[str, float]]:
    """
    Runs Chi-square test between each feature and the label column.
    Returns list of (feature_name, p_value).
    """

    results = []

    for col in feature_cols:
        contingency_table = pd.crosstab(df[col], df[label_col])

        # Skip variables with insufficient variation
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            continue

        _, p_value, _, _ = chi2_contingency(contingency_table)
        results.append((col, p_value))

    return results

def filter_significant_results(
    chi_square_results: List[Tuple[str, float]],
    alpha: float,
    save_path: str = None
) -> List[Tuple[str, float]]:
    """
    Filters chi-square results by significance level.

    Parameters
    ----------
    chi_square_results : List[Tuple[str, float]]
        List of (variable, p-value) tuples from chi-square test.
    alpha : float
        Significance level threshold.
    save_path : str, optional
        If provided, saves the significant results as CSV.

    Returns
    -------
    List[Tuple[str, float]]
       Filtered list of significant (variable, p-value) tuples.    
    """
    significant_results = [(var, p) for var, p in chi_square_results if p < alpha]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(significant_results, columns=["variable", "p_value"]).to_csv(
            save_path,
            index=False
        )

    return significant_results

# =============================================================================
# MANOVA ANALYSIS
# =============================================================================

def run_manova(
    df: pd.DataFrame, 
    dependent_columns: list, 
    independent_column: str
):
    """
    Perform MANOVA for given dependent and independent columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    dependent_columns : list
        List of dependent variables (numerical).
    independent_column : str
        Name of independent variable (categorical).

    Returns
    -------
    manova : MANOVA
        Fitted MANOVA object.
    mv_test_results : dict
        Dictionary containing MANOVA test results for each effect.
    """
    formula = f"{'+'.join(dependent_columns)} ~ {independent_column}"
    manova = MANOVA.from_formula(formula, data=df)

    mv_test_results = manova.mv_test().results 
    
    return manova, mv_test_results
