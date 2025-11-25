# --- Standard library imports ---
import glob
import gzip
import importlib.util
import io
import itertools
from itertools import combinations
import os
import re
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from tqdm.notebook import tqdm
import warnings
import logging
import json
import tempfile
from functools import reduce

# --- Display and plotting ---
from IPython.display import display, Image

# --- Typing ---
from typing import List, Tuple, Union, Optional, Dict, Any, Callable, Literal
from collections import defaultdict

# --- Scientific computing & data analysis ---
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import rankdata
from scipy.stats import kruskal
from scipy import linalg
from scipy.stats import t
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from scipy.stats import gmean
import dcor
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import quantile_transform

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D

# --- Machine learning & statistics ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import adjusted_rand_score
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --- MOFA2 & related ---
# import mofax
# from mofapy2.run.entry_point import entry_point

# --- Bioinformatics ---
import gff3_parser
from Bio import SeqIO

# --- PDF and Excel handling ---
from openpyxl import load_workbook
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# --- Network analysis ---
import networkx as nx
import community as community_louvain
import igraph as ig
import leidenalg
from ipycytoscape import CytoscapeWidget

# --- Cheminformatics ---
from rdkit import Chem

# --- Parallelization ---
from joblib import Parallel, delayed

# --- Plotly for interactive plots ---
import plotly.graph_objects as go
import plotly.io as pio
#pio.kaleido.scope.mathjax = None

# ====================================
# Helper functions for various tasks
# ====================================

log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    fmt = "\033[47m%(levelname)s - %(message)s\033[0m"
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

def clear_directory(dir_path: str) -> None:
    # Wipe out all contents of dir_path if generating new outputs
    if os.path.exists(dir_path):
        #log.info(f"Clearing existing contents of directory: {dir_path}")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                log.info(f'Failed to delete {file_path}. Reason: {e}')  

def write_integration_file(
    data: pd.DataFrame,
    output_dir: str,
    filename: str,
    indexing: bool = True,
    index_label: str = None
) -> None:
    """
    Write a DataFrame to a CSV file in the specified output directory.

    Args:
        data (pd.DataFrame): Data to write.
        output_dir (str): Output directory.
        filename (str): Output filename (without extension).
        indexing (bool): Whether to write the index.
        index_label (str, optional): Name for the index column.

    Returns:
        None
    """

    if output_dir:
        if ".csv" in filename:
            filename = filename.replace(".csv", "")
        fname = f"{output_dir}/{filename}.csv"
        if index_label is not None:
            data.index.name = index_label
        data.to_csv(fname, index=indexing)
        log.info(f"\tData saved to {fname}")
    else:
        log.info("Not saving data to disk.")

def list_persistent_configs():
    """List available persistent configuration sets in a directory with last write dates."""
    config_pattern = "/home/jovyan/work/output_data/*/configs/*.yml"
    config_files = glob.glob(config_pattern, recursive=True)
    
    # Group by hash combination
    config_sets = {}
    for config_file in config_files:
        config_file = Path(config_file)
        name_parts = config_file.stem.split('_')
        
        if len(name_parts) >= 5:
            data_hash = None
            analysis_hash = None
            
            for part in name_parts:
                if 'Processing--' in part:
                    data_hash = part.split('--')[1]
                elif 'Analysis--' in part:
                    analysis_hash = part.split('--')[1]
            
            if data_hash and analysis_hash:
                hash_combo = (data_hash, analysis_hash)
                
                if hash_combo not in config_sets:
                    config_sets[hash_combo] = []
                
                # Get file modification time
                mod_time = config_file.stat().st_mtime
                config_sets[hash_combo].append({
                    'file': config_file,
                    'mod_time': mod_time,
                    'timestamp': datetime.fromtimestamp(mod_time)
                })
    
    # Find most recent file for each hash combination and prepare table data
    table_data = []
    for (data_hash, analysis_hash), files in config_sets.items():
        # Sort by modification time and get the most recent
        most_recent = max(files, key=lambda x: x['mod_time'])
        
        config_types = [f['file'].stem.split('_')[-2] for f in files]
        complete = len(config_types) >= 3
        status = "Complete" if complete else f"Missing ({3-len(config_types)} files)"
        
        table_data.append({
            'Data Hash': data_hash,
            'Analysis Hash': analysis_hash,
            'Last Modified': most_recent['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Status': status,
            'File Count': len(files)
        })
    
    # Sort by last modified date (most recent first)
    table_data.sort(key=lambda x: x['Last Modified'], reverse=True)
    
    # Create and display DataFrame
    if table_data:
        df = pd.DataFrame(table_data)
        log.info("Available persistent configuration sets:")
        display(df)
    else:
        log.info("No persistent configuration sets found yet.")
    
    return

def list_project_configs() -> None:
    """
    List all saved configuration files for a project and print to standard output.
    """
    config_pattern = "/home/jovyan/work/output_data/*/*/configs/*.yml"
    config_files = glob.glob(config_pattern, recursive=True)
    default_config = "/home/jovyan/work/input_data/config/project_config.yml"
    config_files.append(default_config)

    config_info = []
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            if config_file == default_config:
                metadata = {
                    'created_at': 'Default',
                    'data_processing_tag': config.get('datasets', {}).get('data_processing_tag', 'Unknown'),
                    'data_analysis_tag': config.get('analysis', {}).get('data_analysis_tag', 'Unknown')
                }
            else:
                metadata = config.get('_metadata', {})
            config_info.append({
                'path': config_file,
                'filename': os.path.basename(config_file),
                'created_at': metadata.get('created_at', 'Unknown'),
                'data_processing_tag': metadata.get('data_processing_tag', 'Unknown'),
                'data_analysis_tag': metadata.get('data_analysis_tag', 'Unknown')
            })
        except Exception as e:
            log.warning(f"Could not read config {config_file}: {e}")

    config_info_sorted = sorted(config_info, key=lambda x: x['created_at'], reverse=True)
    print(f"{'Created At':40} {'Data Tag':20} {'Analysis Tag':20} {'Path'}")
    print("-" * 120)
    for cfg in config_info_sorted:
        print(f"{str(cfg['created_at']):40} {str(cfg['data_processing_tag']):20} {str(cfg['data_analysis_tag']):20} {cfg['path']}")

def load_project_config(config_path: str = None) -> dict:
    """
    Load a project configuration by path or by searching for tags.
    
    Args:
        config_path (str): Direct path to config file
        project_name (str): Project name to search within
        data_processing_tag (str): Data processing tag to find
        data_analysis_tag (str): Analysis tag to find
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is not None:
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        log.info(f"Loaded configuration from {config_path}")
    else:
        default_path = "/home/jovyan/work/input_data/config/project_config.yml"
        with open(default_path, 'r') as f:
            config = yaml.safe_load(f)
        log.info(f"Loaded default configuration file from {default_path}")
    
    return config

def load_openai_api_key(key_file_path: str = "/home/jovyan/work/input_data/dev/.openai_api_key") -> Optional[str]:
    """
    Load OpenAI API key from a hidden file.
    
    Args:
        key_file_path (str): Path to the API key file
        
    Returns:
        str or None: The API key if found and valid, None otherwise
    """
    try:
        if not os.path.exists(key_file_path):
            log.warning(f"OpenAI API key file not found at {key_file_path}")
            return None
            
        with open(key_file_path, 'r') as f:
            api_key = f.read().strip()
            
        if not api_key:
            log.warning("API key file is empty")
            return None
            
        log.info("OpenAI API key loaded successfully")
        return api_key
        
    except PermissionError:
        log.error(f"Permission denied reading API key file: {key_file_path}")
        return None
    except Exception as e:
        log.error(f"Error loading API key: {e}")
        return None

# ====================================
# Analysis step functions
# ====================================

def _fdr_correct(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Return (adjusted p-values, mask of significant)."""
    adj = multipletests(pvals, method="fdr_bh", alpha=alpha)[1]
    log.info("FDR correction applied.")
    return adj, adj <= alpha

def _subset_by_rank(
    scores: pd.DataFrame,
    score_col: str,
    max_features: int,
    ascending: bool = False,
) -> List[str]:
    """Return a list of top-`max_features` feature names sorted by `score_col`."""
    if scores.empty:
        return []
    ordered = scores[score_col].sort_values(ascending=ascending).index
    log.info(f"Selecting top {max_features} features by {score_col}.")
    return list(ordered[:max_features])

def variance_selection(
    data: pd.DataFrame,
    top_n: int = None,
    max_features: int = 5000,
) -> pd.DataFrame:
    """Select the `max_features` most variable features."""
    var_series = data.var(axis=1).rename("variance")
    top = _subset_by_rank(var_series.to_frame(), "variance", max_features, ascending=False)
    log.info(f"Variance selection: keeping top {max_features} most variable features.")
    return data.loc[top]

def glm_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    category: str,
    reference: str,
    significance_level: float = 0.05,
    log2fc_cutoff: float = 0.25,
    max_features: int = 10000,
) -> pd.DataFrame:
    """
    Fit a Gaussian GLM for each feature vs. a categorical metadata column.
    Returns a subset of features that satisfy BOTH a p-value threshold
    (FDR-adjusted) and an absolute log2-fold-change threshold.
    """
    
    # Merge once and prepare data
    df = data.T.join(metadata)
    if category not in df.columns:
        raise ValueError(f"Metadata column '{category}' not found.")
    
    # Force ordered categorical with reference first
    uniq = df[category].dropna().unique()
    if reference not in uniq:
        raise ValueError(f"Reference group '{reference}' not present in '{category}'.")
    ordered_cats = [reference] + [c for c in uniq if c != reference]
    df[category] = pd.Categorical(df[category], categories=ordered_cats, ordered=True)
    
    # Remove rows with missing category values
    df_clean = df.dropna(subset=[category]).copy()
    
    # Prepare design matrix using pandas get_dummies (much faster than repeated formula parsing)
    design_matrix = pd.get_dummies(df_clean[category], prefix=category, drop_first=True)
    design_matrix.insert(0, 'intercept', 1)
    
    # Get feature data as numpy array for vectorized operations
    feature_data = df_clean[data.index].values
    X = design_matrix.values.astype(np.float64)
    
    log.info(f"GLM: fitting {data.shape[0]} features against '{category}' (ref={reference})")

    # Vectorized GLM fitting using numpy/scipy
    try:
        XtX_inv = linalg.inv(X.T @ X)
        XtY = X.T @ feature_data
        coeffs_matrix = XtX_inv @ XtY
        
        # Calculate residuals and standard errors
        fitted_values = X @ coeffs_matrix
        residuals = feature_data - fitted_values
        
        # Degrees of freedom
        n, p = X.shape
        df_resid = n - p
        
        # Mean squared error for each feature
        mse = np.sum(residuals**2, axis=0) / df_resid
        
        # Standard errors for the first non-intercept coefficient (reference contrast)
        coeff_idx = 1
        se_coeff = np.sqrt(mse * XtX_inv[coeff_idx, coeff_idx])
        
        # Extract coefficients and calculate t-statistics
        coeffs = coeffs_matrix[coeff_idx, :]
        t_stats = coeffs / se_coeff
        
        # Calculate p-values
        pvals = 2 * (1 - t.cdf(np.abs(t_stats), df_resid))
        
    except Exception as e:
        # Fallback to individual fitting if matrix is singular
        log.warning(f"Matrix singular, falling back to individual GLM fitting: {e}")
        
        # Initialize arrays for fallback results
        coeffs = np.zeros(data.shape[0])
        pvals = np.ones(data.shape[0])
        
        # Fit individual GLMs
        for i, feature in enumerate(data.index):
            try:
                # Prepare data for this feature
                feature_df = df_clean[[category]].copy()
                feature_df['response'] = df_clean[feature]
                feature_df = feature_df.dropna()
                
                if len(feature_df) < 3:  # Need at least 3 observations
                    coeffs[i] = 0
                    pvals[i] = 1.0
                    continue
                
                # Fit GLM using statsmodels
                y = feature_df['response']
                X_feature = pd.get_dummies(feature_df[category], drop_first=True)
                X_feature = sm.add_constant(X_feature)
                
                model = sm.OLS(y, X_feature).fit()
                
                # Extract coefficient for first non-intercept term
                if len(model.params) > 1:
                    coeffs[i] = model.params.iloc[1]
                    pvals[i] = model.pvalues.iloc[1]
                else:
                    coeffs[i] = 0
                    pvals[i] = 1.0
                    
            except Exception as feature_error:
                coeffs[i] = 0
                pvals[i] = 1.0
    
    # Create series with results
    coeffs_s = pd.Series(coeffs, index=data.index, name="coef")
    pvals_s = pd.Series(pvals, index=data.index, name="pvalue")
    
    # Multiple-testing correction
    adj, sig_mask = _fdr_correct(pvals_s.values, alpha=significance_level)
    
    # Apply both filters
    passes = sig_mask & (np.abs(coeffs) >= log2fc_cutoff)
    selected = data.index[passes].tolist()
    
    if len(selected) > max_features:
        # Rank by absolute coefficient (largest effect first)
        selected = coeffs_s.loc[selected].abs().sort_values(ascending=False).index[:max_features].tolist()
    
    log.info(f"GLM selected {len(selected)} features (max_features={max_features})")
    return data.loc[selected]

def kruskal_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    category: str,
    significance_level: float = 0.05,
    lfc_cutoff: float = 0.5,
    max_features: int = 10000,
) -> pd.DataFrame:
    """
    Kruskal-Wallis test per feature against a categorical metadata column.
    Returns features that satisfy both an FDR-adjusted p-value < `significance_level`
    and an absolute median-difference (as a proxy for effect size) >= `lfc_cutoff`.
    """
    if category not in metadata.columns:
        raise ValueError(f"Metadata column '{category}' missing. Set it in your analysis config file.")
    
    groups = metadata[category].dropna().unique()
    if len(groups) < 2:
        raise ValueError(f"Need at least two groups in '{category}' for Kruskal-Wallis.")
    
    # Build dict of sample indices per group
    idx_by_grp = {g: metadata[metadata[category] == g].index for g in groups}
    
    log.info(f"Kruskal-Wallis test: evaluating {data.shape[0]} features for '{category}'")
    
    # Vectorized approach using scipy.stats
    try:
        # Pre-allocate arrays for results
        n_features = data.shape[0]
        pvals = np.full(n_features, np.nan)
        effects = np.full(n_features, np.nan)
        
        # Group data into arrays for vectorized processing
        group_data = []
        for g in groups:
            group_samples = [s for s in idx_by_grp[g] if s in data.columns]
            if group_samples:
                group_data.append(data[group_samples].values)
            else:
                raise ValueError(f"No valid samples found for group {g}")
        
        # Check for constant features across all groups
        all_data = np.concatenate(group_data, axis=1)
        constant_features = np.var(all_data, axis=1) < 1e-10
        
        # Vectorized Kruskal-Wallis test
        if not constant_features.all():
            # For non-constant features, run vectorized Kruskal-Wallis
            valid_features = ~constant_features
            valid_indices = np.where(valid_features)[0]
            
            for i, feat_idx in enumerate(valid_indices):
                samples_by_group = [group_data[j][feat_idx, :] for j in range(len(groups))]
                try:
                    _, p = kruskal(*samples_by_group)
                    pvals[feat_idx] = p
                    
                    # Calculate effect size (max pairwise median difference)
                    medians = [np.median(samples) for samples in samples_by_group]
                    max_diff = max(abs(m1 - m2) for i, m1 in enumerate(medians)
                                                  for m2 in medians[i + 1:])
                    effects[feat_idx] = max_diff
                except Exception:
                    pvals[feat_idx] = 1.0
                    effects[feat_idx] = 0.0
        
        # Set constant features to p=1, effect=0
        pvals[constant_features] = 1.0
        effects[constant_features] = 0.0
        
        # Alternative: Use joblib for parallel processing if many features
        if n_features > 5000:
            log.info("Large dataset detected, using parallel processing...")
            
            def process_feature(feat_idx):
                if constant_features[feat_idx]:
                    return 1.0, 0.0
                
                samples_by_group = [group_data[j][feat_idx, :] for j in range(len(groups))]
                try:
                    _, p = kruskal(*samples_by_group)
                    medians = [np.median(samples) for samples in samples_by_group]
                    max_diff = max(abs(m1 - m2) for i, m1 in enumerate(medians)
                                                  for m2 in medians[i + 1:])
                    return p, max_diff
                except Exception:
                    return 1.0, 0.0
            
            # Parallel processing
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_feature)(i) for i in range(n_features)
            )
            
            pvals, effects = zip(*results)
            pvals = np.array(pvals)
            effects = np.array(effects)
            
    except Exception as e:
        log.warning(f"Vectorized approach to KW test failed: {e}")

    # Create series with results
    pvals_s = pd.Series(pvals, index=data.index, name="pvalue")
    eff_s = pd.Series(effects, index=data.index, name="effect")
    
    # Multiple-testing correction
    adj, sig_mask = _fdr_correct(pvals_s.values, alpha=significance_level)
    
    # Apply both thresholds
    passes = sig_mask & (eff_s.abs() >= lfc_cutoff)
    selected = data.index[passes].tolist()
    
    if len(selected) > max_features:
        selected = eff_s.loc[selected].abs().sort_values(ascending=False).index[:max_features].tolist()
    
    log.info(f"Kruskal-Wallis selected {len(selected)} features")
    return data.loc[selected]

def feature_list_selection(
    data: pd.DataFrame,
    feature_list_file: Union[str, Path],
    max_features: int = 10000,
) -> pd.DataFrame:
    """
    Subset features by a user-provided list (one feature name per line).
    """
    path = Path(feature_list_file)
    if not path.is_file():
        raise ValueError(f"Feature list file not found: {path}")
    feature_list = pd.read_csv(path, header=None, sep="\s+", engine="python")[0].astype(str).tolist()
    intersect = list(set(data.index).intersection(feature_list))
    if not intersect:
        log.warning("No features from the list were present in the data matrix.")
        return pd.DataFrame()
    # Respect max_features - keep order as in the original list (if possible)
    ordered = [f for f in feature_list if f in intersect][:max_features]
    log.info(f"Feature list selection: using file {feature_list_file}, max {max_features} features.")
    return data.loc[ordered]

def lasso_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    category: str,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Fit LassoCV to predict a continuous category variable from all features.
    Returns the top-`max_features` features by absolute coefficient magnitude.
    """
    if category not in metadata.columns:
        raise ValueError(f"Target column '{category}' missing. Set it in your analysis config file.")

    # Check if category can be coerced to float
    try:
        y = metadata[category].astype(float).values
    except Exception:
        raise ValueError(f"Target column '{category}' is probably not continuous and cannot be coerced to float. Choose a different category or selection method.")

    X = data.T.values  # samples x features
    lasso = LassoCV(cv=5, n_jobs=-1, random_state=0).fit(X, y)
    coef = pd.Series(np.abs(lasso.coef_), index=data.index, name="lasso_importance")
    top = _subset_by_rank(coef.to_frame(), "lasso_importance", max_features, ascending=False)
    log.info(f"Lasso selection: fitting model for {category}, selecting top {max_features} features.")
    return data.loc[top]

def random_forest_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    category: str,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Train a random-forest model (regressor if target numeric, classifier otherwise)
    and return the top-`max_features` features by Gini importance.
    """
    if category not in metadata.columns:
        raise ValueError(f"Target column '{category}' missing. Set it in your analysis config file.")

    y_raw = metadata[category]
    is_numeric = pd.api.types.is_numeric_dtype(y_raw)
    y = y_raw.astype(float).values if is_numeric else y_raw.astype(str).values
    X = data.T.values

    if is_numeric:
        model = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=0
        )
    else:
        model = RandomForestClassifier(
            n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=0
        )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=data.index, name="rf_importance")
    top = _subset_by_rank(imp.to_frame(), "rf_importance", max_features, ascending=False)
    log.info(f"Random forest selection: fitting model for {category}, selecting top {max_features} features.")
    return data.loc[top]

def mutual_info_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    category: str,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Estimate (non-linear) mutual information between each feature and the target.
    Uses `mutual_info_regression` for continuous targets, otherwise
    `mutual_info_classif`. Returns the top-`max_features` features.
    """
    if category not in metadata.columns:
        raise ValueError(f"Target column '{category}' missing. Set it in your analysis config file.")

    y_raw = metadata[category]
    is_numeric = pd.api.types.is_numeric_dtype(y_raw)
    y = y_raw.astype(float).values if is_numeric else y_raw.astype(str).values
    X = data.T.values
    if is_numeric:
        mi = mutual_info_regression(X, y, random_state=0)
    else:
        mi = mutual_info_classif(X, y, random_state=0)
    mi_series = pd.Series(mi, index=data.index, name="mutual_info")
    top = _subset_by_rank(mi_series.to_frame(), "mutual_info", max_features, ascending=False)
    log.info(f"Mutual information selection: evaluating {data.shape[0]} features for {category}, selecting top {max_features}.")
    return data.loc[top]

def perform_feature_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Dict[str, Any],
    max_features: int = 5000,
    output_dir: str = None,
    output_filename: str = None
) -> pd.DataFrame:
    """
    High-level driver that looks at config["analysis"]["analysis_parameters"]
    ["feature_selection"] and calls the appropriate selection function.

    It validates that all required keys for the chosen method exist,
    writes the resulting subset to output_dir using the filename
    feature_selection_table.csv (or the name you set in the config),
    and returns the subsetted matrix.

    """

    method = config.get("selected_method")
    if not method:
        raise ValueError("'selected_method' not defined in config.")
    method = method.lower().strip()

    # Check that data and metadata have the same samples and subset accordingly
    common_samples = data.columns.intersection(metadata.index)
    if len(common_samples) < 3:
        raise ValueError("Data and metadata have fewer than 3 samples in common.")
    if len(common_samples) < data.shape[1]:
        log.info(f"Data and metadata have {len(common_samples)} samples in common, "
                    f"but data has {data.shape[1]} samples. Subsetting data.")
        data = data[common_samples]
    if len(common_samples) < metadata.shape[0]:
        log.info(f"Data and metadata have {len(common_samples)} samples in common, "
                  f"but metadata has {metadata.shape[0]} samples. Subsetting metadata.")
        metadata = metadata.loc[common_samples]

    # Map method name → function and its required keys
    method_map: Dict[str, Tuple[Callable, List[str]]] = {
        "variance": (variance_selection, ["variance"]),
        "glm": (glm_selection, ["glm"]),
        "kruskalwallis": (kruskal_selection, ["kruskalwallis"]),
        "feature_list": (feature_list_selection, ["feature_list"]),
        "lasso": (lasso_selection, ["lasso"]),
        "random_forest": (random_forest_selection, ["random_forest"]),
        "mutual_info": (mutual_info_selection, ["mutual_info"]),
    }

    if method not in method_map:
        raise ValueError(f"Unsupported feature-selection method '{method}'. "
                                   f"Supported: {list(method_map)}")

    selection_function, selection_parameters = method_map[method]

    kwargs: Dict[str, Any] = {}
    for block in selection_parameters:
        block_cfg = config.get(block, {})
        if not isinstance(block_cfg, dict):
            raise ValueError(f"Configuration for '{block}' must be a mapping.")
        kwargs.update(block_cfg)
    
    kwargs["max_features"] = max_features
    kwargs["data"] = data
    if selection_function is not feature_list_selection and \
       selection_function is not variance_selection:
        kwargs["metadata"] = metadata

    log.info(f"Performing feature selection using method '{method}'.")
    try:
        subset = selection_function(**kwargs)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Error while executing method '{method}': {exc}") from exc

    log.info(f"Feature selection complete.")
    log.info(f"Selected {subset.shape[0]} features out of {data.shape[0]} total.")
    write_integration_file(subset, output_dir, output_filename, indexing=True)
    return subset

def _block_pair(
    Z_i: np.ndarray,
    Z_j: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    scale: float,
    cutoff: float,
    keep_negative: bool,
) -> List[Tuple[int, int, float]]:
    """
    Z_i : (n_samples, b_i)   - transcript block
    Z_j : (n_samples, b_j)   - metabolite block
    idx_i / idx_j : global column indices of the two blocks (int arrays)
    scale : factor to turn the raw dot-product into the final similarity
    """
    # dot-product (fast BLAS)
    sim = (Z_i.T @ Z_j) * scale                # shape (b_i, b_j)

    if keep_negative:
        mask = np.abs(sim) >= cutoff
    else:
        mask = sim >= cutoff

    ii, jj = np.where(mask)                    # indices *inside* the block
    return [
        (int(idx_i[ii[k]]), int(idx_j[jj[k]]), float(sim[ii[k], jj[k]]))
        for k in range(ii.size)
    ]

def calculate_correlated_features(
    data: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    feature_prefixes: List[str] = None,
    method: str = "pearson",
    cutoff: float = 0.75,
    keep_negative: bool = False,
    block_size: int = 500,
    n_jobs: int = 1,
    corr_mode: str = "bipartite",
) -> pd.DataFrame:
    """
    Compute feature correlation on an already normalised feature-by-sample matrix.

    Parameters
    ------
    data : pd.DataFrame
        Rows = features, columns = samples.
        Must already be log-scaled / z-scored etc.
    feature_prefixes : list of str, optional
        List of feature prefixes to distinguish datatypes.
        If None, defaults to ["tx_", "mx_"].
        Example: ["tx_", "mx_", "px_"]
    method : str, default 'pearson'
        Similarity to compute - ``'pearson'``, ``'spearman'``,
        ``'cosine'``, ``'centered_cosine'``, ``'bicor'``,
        ``'dcor'``, or ``'sparse_partial'``.
    cutoff : float, default 0.75
        Minimum absolute similarity (or minimum similarity if
        ``keep_negative=False``) for a pair to be kept.
    keep_negative : bool, default False
        If True, also keep negative correlations whose absolute value
        exceeds ``cutoff``.
    block_size : int, default 500
        Number of features processed per block.
    n_jobs : int, default 1
        Parallelism over transcript blocks.  ``-1`` uses all cores.
    corr_mode : str, default "bipartite"
        Correlation mode:
        - "bipartite": Only compute correlations between different datatypes
        - "full": Compute all pairwise correlations
        - Any other string: Should match a feature prefix

    Returns
    ---
    pd.DataFrame
        Columns ``['feature_1', 'feature_2', 'correlation']``.
    """

    log.info("Starting feature correlation computation...")
    
    # Validate method
    valid_methods = {"pearson", "spearman", "cosine", "centered_cosine", "bicor", "dcor", "sparse_partial"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method '{method}'. Valid: {valid_methods}")
    
    if method in ["dcor"]:
        log.info(f"Using advanced correlation method: {method}")
        return _calculate_dcor_correlation(
            data=data,
            output_filename=output_filename,
            output_dir=output_dir,
            feature_prefixes=feature_prefixes,
            method=method,
            cutoff=cutoff,
            keep_negative=keep_negative,
            corr_mode=corr_mode,
        )

    if method == "sparse_partial":
        return _calculate_sparse_partial_correlations(
            data=data,
            feature_prefixes=feature_prefixes,
            alpha=0.01,
            cutoff=cutoff,
            corr_mode=corr_mode,
            max_iter=100
        )
    
    # Create feature type designation based on prefixes
    ftype = pd.Series(index=data.index, dtype=object)
    ftype[:] = None  # Initialize with None
    
    for i, prefix in enumerate(feature_prefixes):
        mask = data.index.str.startswith(prefix)
        ftype[mask] = f"datatype_{i}"  # Generic datatype labels
    
    if ftype.isnull().any():
        invalid = ftype[ftype.isnull()].index.tolist()
        log.info(f"Invalid feature names detected: {invalid}")
        raise ValueError(f"Feature names must start with one of {feature_prefixes}. Invalid: {invalid}")

    # Basic checks
    if not isinstance(ftype, pd.Series):
        raise TypeError("feature_type must be a pandas Series.")
    if not ftype.index.equals(data.index):
        raise ValueError("Index of feature_type must match data rows.")
    if method not in {
        "pearson", "spearman", "cosine", "centered_cosine", "bicor"
    }:
        raise ValueError(f"Unsupported method '{method}'.")

    log.info(f"Using method: {method}, cutoff: {cutoff}, keep_negative: {keep_negative}, block_size: {block_size}, n_jobs: {n_jobs}, corr_mode: {corr_mode}")

    # Identify feature indices for each datatype
    X = data.T.values.astype(np.float64, copy=False)   # (n_samples, n_features)
    feature_names = data.index.to_numpy()
    n_features = X.shape[1]

    # Create indices for each datatype
    datatype_indices = {}
    for i, prefix in enumerate(feature_prefixes):
        datatype_name = f"datatype_{i}"
        mask = ftype.eq(datatype_name).values
        datatype_indices[datatype_name] = np.where(mask)[0]
        log.info(f"Found {len(datatype_indices[datatype_name])} features with prefix '{prefix}'.")

    # Determine which pairs to compute based on corr_mode
    if corr_mode == "bipartite":
        # Only compute between different datatypes
        pairs = []
        datatypes = list(datatype_indices.keys())
        for i, dtype1 in enumerate(datatypes):
            for dtype2 in datatypes[i+1:]:  # Only unique pairs
                pairs.append((datatype_indices[dtype1], datatype_indices[dtype2]))
                pairs.append((datatype_indices[dtype2], datatype_indices[dtype1]))  # Both directions
        log.info(f"Computing bipartite correlations between {len(pairs)//2} datatype pairs.")
    elif corr_mode == "full":
        # All pairs (including within group)
        all_idx = np.arange(n_features)
        pairs = [(all_idx, all_idx)]
        log.info("Computing all pairwise correlations (including within datatype).")
    else:
        # Check if corr_mode matches a feature prefix
        matching_prefix = None
        for prefix in feature_prefixes:
            if corr_mode == prefix.rstrip('_'):  # Allow both "tx" and "tx_"
                matching_prefix = prefix
                break
        
        # Also check if corr_mode exactly matches any feature prefix including underscore
        if matching_prefix is None:
            for prefix in feature_prefixes:
                if corr_mode == prefix:
                    matching_prefix = prefix
                    break
        
        # Check if any features start with the corr_mode string
        if matching_prefix is None:
            feature_first_parts = [feature.split('_')[0] + '_' for feature in data.index]
            if any(corr_mode + '_' == part for part in feature_first_parts):
                matching_prefix = corr_mode + '_'
            elif any(corr_mode == part.rstrip('_') for part in feature_first_parts):
                matching_prefix = corr_mode + '_'
        
        if matching_prefix is None:
            raise ValueError(f"corr_mode '{corr_mode}' does not match 'bipartite', 'full', or any feature prefix from {feature_prefixes}")
        
        # Find the datatype index for the matching prefix
        target_datatype = None
        for i, prefix in enumerate(feature_prefixes):
            if prefix == matching_prefix or prefix.rstrip('_') == corr_mode:
                target_datatype = f"datatype_{i}"
                break
        
        if target_datatype is None or target_datatype not in datatype_indices:
            raise ValueError(f"No features found with prefix matching '{corr_mode}'")
        
        # Only compute correlations within this datatype
        target_indices = datatype_indices[target_datatype]
        pairs = [(target_indices, target_indices)]
        log.info(f"Computing intra-datatype correlations for prefix '{matching_prefix}' ({len(target_indices)} features).")

    # Transform data once according to the requested similarity
    log.info("Transforming data matrix for correlation computation...")
    Z, scale = _set_up_matrix(X, method)
    log.info("Data transformation complete.")

    # Block-wise computation
    def _process_block(i_idx, j_idx, t_start: int, t_end: int) -> List[Tuple[int, int, float]]:
        block_i_idx = i_idx[t_start:t_end]
        Z_i = Z[:, block_i_idx]
        block_results: List[Tuple[int, int, float]] = []
        for m_start in range(0, len(j_idx), block_size):
            m_end = min(m_start + block_size, len(j_idx))
            block_j_idx = j_idx[m_start:m_end]
            Z_j = Z[:, block_j_idx]
            block_results.extend(
                _block_pair(
                    Z_i,
                    Z_j,
                    block_i_idx,
                    block_j_idx,
                    scale,
                    cutoff,
                    keep_negative,
                )
            )
        #log.info(f"Processed block {t_start}:{t_end} against target features.")
        return block_results

    # Parallel over blocks
    all_pairs: List[Tuple[int, int, float]] = []
    for i_idx, j_idx in pairs:
        log.info(f"Processing {len(i_idx)} source features in blocks of {block_size}...")
        block_ranges = list(range(0, len(i_idx), block_size))
        if n_jobs == 1:
            for t_start in block_ranges:
                t_end = min(t_start + block_size, len(i_idx))
                all_pairs.extend(_process_block(i_idx, j_idx, t_start, t_end))
        else:
            n_jobs_eff = -1 if n_jobs == -1 else n_jobs
            log.info(f"Processing in parallel with {n_jobs_eff} jobs...")
            parallel = Parallel(n_jobs=n_jobs_eff, backend="loky", verbose=0)
            chunks = parallel(
                delayed(_process_block)(i_idx, j_idx, t_start, min(t_start + block_size, len(i_idx)))
                for t_start in block_ranges
            )
            log.info("Parallel block processing complete.")
            all_pairs.extend([pair for sublist in chunks for pair in sublist])

    if not all_pairs:
        empty_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        log.info("Warning: No pairs passed the correlation cutoff. Returning empty DataFrame.")
        write_integration_file(data=empty_df, output_dir=output_dir, filename=output_filename, indexing=True)
        return empty_df

    log.info(f"Total pairs passing cutoff: {len(all_pairs)}")
    tr_idx, met_idx, sims = zip(*all_pairs)
    df = pd.DataFrame(
        {
            "feature_1": feature_names[np.fromiter(tr_idx, dtype=int, count=len(tr_idx))],
            "feature_2": feature_names[np.fromiter(met_idx, dtype=int, count=len(met_idx))],
            "correlation": sims,
        }
    )
    df["abs_corr"] = np.abs(df["correlation"])
    df = df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")

    log.info(f"Writing correlation results to {output_dir}/{output_filename}.csv")
    write_integration_file(data=df, output_dir=output_dir, filename=output_filename, indexing=True)
    log.info("Correlation computation complete.")
    return df

def _set_up_matrix(
    X: np.ndarray,
    method: str,
    data_is_normalized: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Parameters
    ------
    X : (n_samples, n_features) ndarray
        Data matrix (already log2-scaled and z-scored).
    method : str
        One of  {'pearson', 'spearman', 'cosine',
                  'centered_cosine', 'bicor'}.
    data_is_normalized : bool, default False
        If True, assumes X is already log2-scaled and z-scored.
        For Pearson correlation, will still re-normalize after centering.

    Returns
    -------
    Z : ndarray, same shape as X
        Transformed matrix.
    scale : float
        Multiplicative factor that must be applied to the dot-product
        `Z_i.T @ Z_j` to obtain the final similarity.
        For Pearson / Spearman / bicor   → 1/(n_samples-1)
        For Cosine / Centered-Cosine   → 1
    """
    n = X.shape[0]
    
    if method in ["pearson", "spearman", "bicor"]:
        # Even if data is already z-scored, we need to:
        # 1. Center (mean = 0)
        # 2. Normalize by std (std = 1)
        # This ensures proper correlation calculation
        
        # Center the data
        mu = X.mean(axis=0, keepdims=True)
        Z = X - mu
        
        # Normalize by standard deviation
        # Use ddof=1 for sample standard deviation
        std = Z.std(axis=0, keepdims=True, ddof=1)
        
        # Avoid division by zero for constant features
        std[std == 0] = 1.0
        
        # Normalize
        Z = Z / std
        
        # Scale factor for correlation
        scale = 1.0 / (n - 1)
        
    elif method == "cosine":
        # Normalize to unit length (L2 norm)
        norm = np.linalg.norm(X, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        Z = X / norm
        scale = 1.0
        
    elif method == "centered_cosine":
        # Center then normalize to unit length
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        norm = np.linalg.norm(Xc, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        Z = Xc / norm
        scale = 1.0
        
    else:
        raise ValueError(
            f"Method '{method}' not recognised. Choose "
            "'pearson', 'spearman', 'cosine', 'centered_cosine' or 'bicor'."
        )
    
    return Z, scale

def _calculate_sparse_partial_correlations(
    data: pd.DataFrame,
    feature_prefixes: List[str],
    alpha: float = 0.1, #higher for larger datasets?
    cutoff: float = 0.3,
    corr_mode: str = "full",
    max_iter: int = 100
) -> pd.DataFrame:
    """
    Fast non-bivariate correlation using Graphical Lasso.
        
    Args:
        data: Feature matrix (features x samples)
        alpha: Sparsity parameter (0.001-0.1 typical range)
               Lower = denser network, slower computation
        cutoff: Minimum absolute partial correlation to keep
        max_iter: Maximum iterations (lower = faster but less accurate)
    """
    
    # Subset features based on mode
    if corr_mode == "full":
        features = data.index.tolist()
    elif corr_mode == "bipartite":
        features = data.index.tolist()
    else:
        features = [f for f in data.index if f.startswith(f"{corr_mode}_")]
    
    log.info(f"Computing sparse partial correlations for {len(features)} features...")
    
    # Prepare data (samples x features for sklearn)
    X = data.loc[features].T.fillna(0).values
    
    # Standardize features (important for GraphicalLasso)
    X = StandardScaler().fit_transform(X)
    
    log.info(f"  Fitting Graphical Lasso (alpha={alpha}, max_iter={max_iter})...")
    
    # Fit model with fixed alpha (no cross-validation = much faster)
    model = GraphicalLasso(
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-3,  # Slightly relaxed tolerance for speed
        verbose=1,  # Show progress
        assume_centered=False
    )
    
    model.fit(X)
    
    log.info(f"  Converged in {model.n_iter_} iterations")
    
    # Convert precision matrix to partial correlations
    precision = model.precision_
    
    # Partial correlation formula: 
    # ρ_ij = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    diag = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial_corr, 0)  # Remove self-correlations
    
    # Extract edges above cutoff
    log.info(f"  Extracting edges above cutoff {cutoff}...")
    correlation_data = []
    feature_names = data.loc[features].index.tolist()
    
    # Only upper triangle to avoid duplicates
    rows, cols = np.triu_indices_from(partial_corr, k=1)
    
    for i, j in zip(rows, cols):
        corr_val = partial_corr[i, j]
        if abs(corr_val) >= cutoff:
            correlation_data.append({
                'feature_1': feature_names[i],
                'feature_2': feature_names[j],
                'correlation': corr_val
            })
    
    results_df = pd.DataFrame(correlation_data)
    log.info(f"  Found {len(results_df):,} correlations above cutoff")
    
    return results_df

def _calculate_dcor_correlation(
    data: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    feature_prefixes: List[str],
    method: str,
    cutoff: float,
    keep_negative: bool,
    corr_mode: str,
) -> pd.DataFrame:
    """
    Calculate correlations using advanced methods (dcor).
    
    These methods don't benefit from the block-wise optimization used for
    standard correlations, so we compute them directly.
    """
    
    log.info(f"Computing {method} correlations for {data.shape[0]} features...")
    
    # Create feature type designation
    ftype = pd.Series(index=data.index, dtype=object)
    ftype[:] = None
    
    for i, prefix in enumerate(feature_prefixes):
        mask = data.index.str.startswith(prefix)
        ftype[mask] = f"datatype_{i}"
    
    if ftype.isnull().any():
        invalid = ftype[ftype.isnull()].index.tolist()
        raise ValueError(f"Feature names must start with one of {feature_prefixes}. Invalid: {invalid}")
    
    # Identify feature indices for each datatype
    X = data.T.values.astype(np.float64, copy=False)
    feature_names = data.index.to_numpy()
    n_features = X.shape[1]
    
    datatype_indices = {}
    for i, prefix in enumerate(feature_prefixes):
        datatype_name = f"datatype_{i}"
        mask = ftype.eq(datatype_name).values
        datatype_indices[datatype_name] = np.where(mask)[0]
        log.info(f"Found {len(datatype_indices[datatype_name])} features with prefix '{prefix}'.")
    
    # Determine which pairs to compute
    if corr_mode == "bipartite":
        pairs = []
        datatypes = list(datatype_indices.keys())
        for i, dtype1 in enumerate(datatypes):
            for dtype2 in datatypes[i+1:]:
                pairs.append((datatype_indices[dtype1], datatype_indices[dtype2]))
    elif corr_mode == "full":
        all_idx = np.arange(n_features)
        pairs = [(all_idx, all_idx)]
    else:
        # Find matching datatype
        matching_prefix = None
        for prefix in feature_prefixes:
            if corr_mode == prefix.rstrip('_') or corr_mode == prefix:
                matching_prefix = prefix
                break
        
        if matching_prefix is None:
            raise ValueError(f"corr_mode '{corr_mode}' does not match any feature prefix")
        
        target_datatype = None
        for i, prefix in enumerate(feature_prefixes):
            if prefix == matching_prefix or prefix.rstrip('_') == corr_mode:
                target_datatype = f"datatype_{i}"
                break
        
        if target_datatype not in datatype_indices:
            raise ValueError(f"No features found with prefix matching '{corr_mode}'")
        
        target_indices = datatype_indices[target_datatype]
        pairs = [(target_indices, target_indices)]
    
    # Compute correlations based on method
    all_results = []
    
    for i_idx, j_idx in pairs:
        if method == "dcor":
            results = _compute_distance_correlation(X, i_idx, j_idx, cutoff, keep_negative)
        
        all_results.extend(results)
    
    if not all_results:
        empty_df = pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])
        log.info("Warning: No pairs passed the correlation cutoff.")
        write_integration_file(data=empty_df, output_dir=output_dir, filename=output_filename, indexing=True)
        return empty_df
    
    # Convert to DataFrame
    log.info(f"Total pairs passing cutoff: {len(all_results)}")
    tr_idx, met_idx, sims = zip(*all_results)
    df = pd.DataFrame({
        "feature_1": feature_names[np.array(tr_idx)],
        "feature_2": feature_names[np.array(met_idx)],
        "correlation": sims,
    })
    df["abs_corr"] = np.abs(df["correlation"])
    df = df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
    
    write_integration_file(data=df, output_dir=output_dir, filename=output_filename, indexing=True)
    log.info("Correlation computation complete.")
    return df

def _compute_distance_correlation(
    X: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    cutoff: float,
    keep_negative: bool,
) -> List[Tuple[int, int, float]]:
    """
    Compute distance correlation between features using precomputed distance 
    matrices AND parallel processing for maximum performance.
    
    This approach:
    1. Precomputes distance matrices for i_idx features once
    2. Processes j features in parallel for each i feature
    3. Uses the fast manual distance correlation calculation
    
    Args:
        X: Data matrix (n_samples, n_features)
        i_idx: Indices of source features
        j_idx: Indices of target features
        cutoff: Minimum correlation threshold
        keep_negative: Whether to keep negative correlations (ignored for dcor)
        
    Returns:
        List of (i_index, j_index, correlation) tuples passing the cutoff
    """
    
    log.info("  Precomputing distance matrices for source features...")
    distance_matrices_i = {}
    for i in tqdm(i_idx, desc="Precomputing distances", unit="feature"):
        x_i = X[:, i]
        distances = np.abs(x_i[:, np.newaxis] - x_i[np.newaxis, :])
        distance_matrices_i[i] = distances
    
    log.info("  Computing distance correlations in parallel...")
    
    def process_i_feature(i):
        """Process all j features for a single i feature in parallel."""
        dist_i = distance_matrices_i[i]
        feature_results = []
        
        for j in j_idx:
            if i == j:
                continue
            
            x_j = X[:, j]
            dist_j = np.abs(x_j[:, np.newaxis] - x_j[np.newaxis, :])
            
            # Use fast manual computation
            dc = _fast_distance_correlation(dist_i, dist_j)
            
            # Distance correlation is always >= 0
            if dc >= cutoff:
                feature_results.append((int(i), int(j), float(dc)))
        
        return feature_results
    
    # Parallel processing over i features
    all_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_i_feature)(i)
        for i in tqdm(i_idx, desc="Distance correlation", unit="feature")
    )
    
    # Flatten the nested results
    results = [pair for feature_results in all_results for pair in feature_results]
    
    log.info(f"  Found {len(results)} correlations passing cutoff")
    
    return results

def _fast_distance_correlation(dist_A: np.ndarray, dist_B: np.ndarray) -> float:
    """
    Fast computation of distance correlation from precomputed distance matrices.
    
    This implements the distance correlation formula directly to avoid
    overhead from the dcor library's repeated calculations.
    """
    n = dist_A.shape[0]
    
    # Double center the distance matrices
    # This is the key operation in distance correlation
    row_mean_A = dist_A.mean(axis=1, keepdims=True)
    col_mean_A = dist_A.mean(axis=0, keepdims=True)
    grand_mean_A = dist_A.mean()
    A_centered = dist_A - row_mean_A - col_mean_A + grand_mean_A
    
    row_mean_B = dist_B.mean(axis=1, keepdims=True)
    col_mean_B = dist_B.mean(axis=0, keepdims=True)
    grand_mean_B = dist_B.mean()
    B_centered = dist_B - row_mean_B - col_mean_B + grand_mean_B
    
    # Compute distance covariance and variances
    dcov_AB = np.sqrt(np.sum(A_centered * B_centered) / (n * n))
    dvar_A = np.sqrt(np.sum(A_centered * A_centered) / (n * n))
    dvar_B = np.sqrt(np.sum(B_centered * B_centered) / (n * n))
    
    # Distance correlation
    if dvar_A > 0 and dvar_B > 0:
        return dcov_AB / np.sqrt(dvar_A * dvar_B)
    else:
        return 0.0

def _make_prefix_maps(prefixes: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return two dicts:
        prefix → color (hex)
        prefix → shape (networkx-compatible string)
    colors are taken from a viridis palette and are reproducible.
    """
    palette = viridis(np.linspace(0, 1, max(3, len(prefixes))))
    colors = [to_hex(c) for c in palette]
    shape_map = {}
    for i, pref in enumerate(prefixes):
        shape_map[pref] = "circle" if i == 0 else "diamond"
    color_map = {pref: colors[i % len(colors)] for i, pref in enumerate(prefixes)}
    return color_map, shape_map

def _build_sparse_adj(
    corr_df: pd.DataFrame,
    prefixes: List[str],
) -> sp.coo_matrix:
    """
    Returns a COO-format sparse matrix where the data are the *edge weights*.

    Includes all non-zero correlations from corr_df (diagonal excluded).
    """
    # Ensure NaNs become zeros
    corr_vals = corr_df.fillna(0.0).values

    # Keep all non-zero entries (the correlation caller already applied any cutoff)
    mask = corr_vals != 0.0
    np.fill_diagonal(mask, False)

    # Extract sparse representation
    row_idx, col_idx = np.where(mask)
    weights = corr_vals[row_idx, col_idx]
    n = corr_df.shape[0]

    return sp.coo_matrix((weights, (row_idx, col_idx)), shape=(n, n))

def _graph_from_sparse(
    adj: sp.coo_matrix,
    node_names: np.ndarray,
) -> nx.Graph:
    """
    Convert a COO adjacency matrix to a networkx Graph with the original
    node names as labels.
    """
    G = nx.from_scipy_sparse_array(adj, edge_attribute="weight", create_using=nx.Graph())
    mapping = dict(enumerate(node_names))
    return nx.relabel_nodes(G, mapping)

def _assign_node_attributes(
    G: nx.Graph,
    prefixes: List[str],
    color_map: Dict[str, str],
    shape_map: Dict[str, str],
    annotation_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Mutates G in-place - adds:
        * datatype_color, datatype_shape, node_size
        * All annotation columns as double semicolon-separated strings (if annotation_df supplied)
        * Handles multiple annotations per feature by storing as ";;"-separated strings
    """
    # Color / shape based on prefix
    node_names = np.array(list(G.nodes()))
    # Vectorised prefix lookup
    node_pref = np.empty(len(node_names), dtype=object)
    node_pref[:] = ""
    for pref in prefixes:
        hits = np.char.startswith(node_names, pref)
        node_pref[hits] = pref

    # Build dicts for networkx.set_node_attributes
    color_dict = {name: color_map.get(p, "gray") for name, p in zip(node_names, node_pref)}
    shape_dict = {name: shape_map.get(p, "Rectangle") for name, p in zip(node_names, node_pref)}
    nx.set_node_attributes(G, color_dict, "datatype_color")
    nx.set_node_attributes(G, shape_dict, "datatype_shape")
    nx.set_node_attributes(G, 10, "node_size")

    # Add all annotation columns as separate node attributes
    if annotation_df is not None and not annotation_df.empty:
        log.info(f"Processing {len(annotation_df)} annotation rows for {len(G.nodes())} nodes...")
        
        # Group annotations by feature_id to handle multiple annotations per feature
        annotation_groups = annotation_df.groupby('feature_id')
        
        # Initialize annotation dictionaries for each column
        annotation_columns = [col for col in annotation_df.columns if col != 'feature_id']
        node_annotations = {col: {} for col in annotation_columns}
        
        # Process each feature and its annotations
        for feature_id, feature_annotations in annotation_groups:
            if feature_id in G.nodes():
                for col in annotation_columns:
                    # Get all non-null, non-"Unassigned" values for this column
                    values = feature_annotations[col].dropna()
                    values = values[values != 'Unassigned'].unique().tolist()
                    
                    if values:
                        # Convert list to double semicolon-separated string
                        node_annotations[col][feature_id] = ";;".join(str(v) for v in values)
                    else:
                        # No valid annotations for this column
                        node_annotations[col][feature_id] = "Unassigned"
        
        # Add nodes without any annotations
        for node in G.nodes():
            for col in annotation_columns:
                if node not in node_annotations[col]:
                    node_annotations[col][node] = "Unassigned"
        
        # Set node attributes for each annotation column
        for col, node_attr_dict in node_annotations.items():
            nx.set_node_attributes(G, node_attr_dict, col)
        
        log.info(f"Added {len(annotation_columns)} annotation attributes to {len(G.nodes())} nodes")
        
        # Summary of annotations per node
        nodes_with_annotations = sum(1 for node in G.nodes() 
                                   if any(G.nodes[node].get(col, "Unassigned") != "Unassigned" 
                                         for col in annotation_columns))
        log.info(f"Nodes with at least one annotation: {nodes_with_annotations}")

def _detect_submodules(
    G: nx.Graph,
    method: str,
    **kwargs,
) -> List[Tuple[str, nx.Graph]]:
    """
    Returns a list of (module_name, subgraph) tuples.
    Supported ``method`` values:
        * "subgraphs" - simple connected components
        * "louvain" - python-louvain
        * "leiden" - leidenalg (requires igraph)
        * "wgcna" - soft-threshold → TOM → hierarchical clustering
        * "k_clique" - k-clique communities
        * "greedy_modularity" - greedy modularity maximization
        * "label_propagation" - asynchronous label propagation
        * "girvan_newman" - Girvan-Newman method
    ``kwargs`` are forwarded to the specific implementation.
    """
    if method == "subgraphs":
        comps = nx.connected_components(G)
        return [(f"submodule_{i+1}", G.subgraph(c).copy())
                for i, c in enumerate(comps)]

    elif method == "louvain":
        partition = community_louvain.best_partition(G, weight="weight")
        modules: Dict[int, List[str]] = {}
        for node, comm in partition.items():
            modules.setdefault(comm, []).append(node)
        return [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                for i, (comm, nodes) in enumerate(sorted(modules.items()))]

    elif method == "leiden":
        # Convert to igraph (preserves node names)
        ig_g = ig.Graph.from_networkx(G)
        node_names = list(G.nodes())
        ig_g.vs["name"] = node_names
        
        partition = leidenalg.find_partition(
            ig_g, leidenalg.ModularityVertexPartition, weights="weight"
        )
        modules = []
        for i, community in enumerate(partition):
            nodes = [ig_g.vs[idx]["name"] for idx in community]
            modules.append((f"submodule_{i+1}", G.subgraph(nodes).copy()))
        return modules

    elif method == "wgcna":
        # build adjacency from the correlation matrix (soft-threshold)
        beta: int = kwargs.get("beta", 6)
        min_mod_sz: int = kwargs.get("min_module_size", 10)
        dist_cut: float = kwargs.get("distance_cutoff", 0.25)

        # original correlation dataframe (must be symmetric)
        corr = nx.to_pandas_adjacency(G, weight="weight")
        raw_corr = np.sqrt(corr.values) / 100.0
        raw_corr = np.clip(raw_corr, -1, 1) * np.sign(corr.values)   # keep sign

        # soft-threshold adjacency
        adj = np.abs(raw_corr) ** beta

        # topological overlap matrix (TOM)
        A = sp.csr_matrix(adj)
        A2 = A @ A
        k = np.asarray(A.sum(axis=1)).ravel()
        min_k = np.minimum.outer(k, k)
        denom = min_k + 1 - adj
        tom = (A2 + A) / denom
        
        # Convert to dense array before using fill_diagonal
        tom = tom.toarray()
        np.fill_diagonal(tom, 1.0)

        # hierarchical clustering on TOM-distance
        dist = 1.0 - tom
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        cluster_labels = fcluster(Z, t=dist_cut, criterion="distance")
        
        # discard clusters that are too small and create submodules
        modules = {}
        for lbl, node in zip(cluster_labels, corr.index):
            modules.setdefault(lbl, []).append(node)
        
        # Filter modules by minimum size and create subgraphs
        kept = []
        for i, (lbl, nodes) in enumerate(sorted(modules.items())):
            if len(nodes) >= min_mod_sz:
                submodule_name = f"submodule_{i+1}"
                subgraph = G.subgraph(nodes).copy()
                kept.append((submodule_name, subgraph))
        
        return kept

    elif method == "k_clique":
        # K-clique percolation
        k: int = kwargs.get("k", 3)
        communities = nx.community.k_clique_communities(G, k)
        return [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                for i, nodes in enumerate(communities)]

    elif method == "greedy_modularity":
        # Greedy modularity maximization
        weight = kwargs.get("weight", "weight")
        resolution = kwargs.get("resolution", 1)
        cutoff = kwargs.get("cutoff", 1)
        best_n = kwargs.get("best_n", None)
        
        communities = nx.community.greedy_modularity_communities(
            G, 
            weight=weight,
            resolution=resolution,
            cutoff=cutoff,
            best_n=best_n
        )
        return [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                for i, nodes in enumerate(communities)]

    elif method == "label_propagation":
        # Asynchronous label propagation
        weight = kwargs.get("weight", "weight")
        seed = kwargs.get("seed", None)
        
        communities = nx.community.asyn_lpa_communities(G, weight=weight, seed=seed)
        return [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                for i, nodes in enumerate(communities)]

    elif method == "girvan_newman":
        # Girvan-Newman method (divisive)
        level: int = kwargs.get("level", 0)
        most_valuable_edge = kwargs.get("most_valuable_edge", None)
        
        communities_generator = nx.community.girvan_newman(
            G, 
            most_valuable_edge=most_valuable_edge
        )
        
        # Iterate to the specified level
        for i, communities in enumerate(communities_generator):
            if i == level:
                break
        
        return [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                for i, nodes in enumerate(communities)]

    else:
        valid_methods = [
            "subgraphs", "louvain", "leiden", "wgcna",
            "k_clique", "greedy_modularity",
            "label_propagation", "girvan_newman"
        ]
        raise ValueError(
            f"Invalid submodule method '{method}'. "
            f"Choose from: {', '.join(valid_methods)}"
        )

def plot_correlation_network(
    corr_table: pd.DataFrame,
    integrated_data: pd.DataFrame,
    integrated_metadata: pd.DataFrame,
    output_dir: str,
    output_filenames: Dict[str, str],
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    submodule_mode: str = "louvain",
    show_plot: bool = False,
    interactive_layout: str = None,
    wgcna_params: dict = {}
) -> None:
    """
    Build a correlation graph from a long-format table, optionally
    detect submodules (connected components, Louvain, Leiden or WGCNA)
    and write everything to disk.

    Parameters are identical to your original function; the only
    behavioural change is the expanded ``submodule_mode`` options.
    
    datasets : List, optional
        List of dataset objects with annotation_table attributes
    """

    # Get all unique features
    all_features = pd.Index(sorted(set(corr_table["feature_1"]).union(set(corr_table["feature_2"]))))
    all_prefixes = [ds.dataset_name + "_" for ds in datasets]
    present_prefixes = [p for p in all_prefixes if any(f.startswith(p) for f in all_features)]

    # Pivot and reindex to ensure square matrix
    correlation_df = corr_table.pivot(index="feature_2", columns="feature_1", values="correlation").reindex(index=all_features, columns=all_features, fill_value=0.0)

    # Build sparse adjacency (edges only above cutoff)
    sparse_adj = _build_sparse_adj(correlation_df, present_prefixes)

    # Create networkx graph (node names = original feature IDs)
    G = _graph_from_sparse(sparse_adj, correlation_df.index.to_numpy())
    log.info(f"Graph built -  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Node aesthetics (color / shape) and optional annotation
    log.info("Assigning node attributes...")
    color_map, shape_map = _make_prefix_maps(present_prefixes)
    _assign_node_attributes(G, present_prefixes, color_map, shape_map, annotation_df)

    # Remove tiny isolated components
    tiny = [c for c in nx.connected_components(G) if len(c) < 3]
    if tiny:
        G.remove_nodes_from({n for comp in tiny for n in comp})
        log.info(f"\tRemoved {len(tiny)} tiny components (<3 nodes).")

    # Export the raw graph (before submodule annotation)
    nx.write_graphml(G, os.path.join(output_dir, output_filenames["graph"]))
    edge_table = nx.to_pandas_edgelist(G)
    node_table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
    write_integration_file(data=edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")
    log.info("\tRaw graph, node table and edge table written to disk.")

    # submodule detection
    if submodule_mode != "none":
        log.info(f"Detecting submodules using '{submodule_mode}'...")
        submods = _detect_submodules(
            G,
            method=submodule_mode,
            beta=wgcna_params["beta"],
            min_module_size=wgcna_params["min_module_size"],
            distance_cutoff=wgcna_params["distance_cutoff"],
        )
        
        # Always annotate nodes with submodule information, even if only one submodule
        if submods:
            _annotate_and_save_submodules(
                submods,
                G,
                output_filenames,
                integrated_data,
                integrated_metadata,
                save_plots=True,
            )
        else:
            # If no submodules were found, assign all nodes to a single submodule
            log.info("No submodules detected, assigning all nodes to 'submodule_1'")
            for node in G.nodes():
                G.nodes[node]["submodule"] = "submodule_1"
                G.nodes[node]["submodule_color"] = "#440154"  # First viridis color

        # Re-write the *main* graph (now enriched with submodule attributes)
        nx.write_graphml(G, os.path.join(output_dir, output_filenames["graph"]))
        edge_table = nx.to_pandas_edgelist(G)
        node_table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
        write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
        write_integration_file(data=edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")
        log.info("\tMain graph updated with submodule annotations and written to disk.")

    # interactive plot
    if show_plot:
        log.info("Rendering interactive network in notebook…")
        color_attr = "submodule_color" if submodule_mode != "none" else "datatype_color"
        log.info("Pre-computing network layout...")
        widget = _nx_to_plotly_widget(
            G,
            node_color_attr=color_attr,
            node_size_attr="node_size",
            layout=interactive_layout,
            seed=42,
        )
        display(widget)

    return node_table, edge_table

def display_existing_network(
    graph_file: str,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    interactive_layout: str = None
) -> None:
    """
    Display an existing network visualization from saved files.
    
    Parameters
    ----------
    graph_file : str
        Path to the saved GraphML file (not used, kept for compatibility)
    node_table : pd.DataFrame
        Node table DataFrame with node attributes
    edge_table : pd.DataFrame
        Edge table DataFrame with edge information
    interactive_layout : str, optional
        Layout algorithm for the interactive plot
    """
    
    try:
        # Clean up the edge table - remove any unnamed or meaningless index columns
        edge_df = edge_table.copy()
        
        # Remove unnamed index columns (columns that start with 'Unnamed:')
        unnamed_cols = [col for col in edge_df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            edge_df = edge_df.drop(columns=unnamed_cols)
            log.info(f"Removed unnamed index columns: {unnamed_cols}")
        
        # Check for and remove meaningless index columns (integer sequences starting from 0)
        index_like_cols = []
        for col in edge_df.columns:
            # Check if column name suggests it's an index
            if any(keyword in str(col).lower() for keyword in ['index', 'idx', '_id']) and col not in ['source', 'target']:
                # Check if it's just a sequence of integers starting from 0
                try:
                    col_values = edge_df[col].dropna()
                    if (col_values.dtype in ['int64', 'int32'] and 
                        len(col_values) > 0 and 
                        col_values.min() == 0 and 
                        col_values.max() == len(col_values) - 1 and
                        len(col_values.unique()) == len(col_values)):  # All unique values
                        index_like_cols.append(col)
                except:
                    continue
        
        if index_like_cols:
            edge_df = edge_df.drop(columns=index_like_cols)
            log.info(f"Removed meaningless index columns: {index_like_cols}")
        
        # Now the edge table should have exactly the columns we expect: source, target, weight
        expected_cols = ['source', 'target', 'weight']
        if not all(col in edge_df.columns for col in expected_cols[:2]):  # at least source and target
            log.error(f"Edge table missing required columns. Expected at least 'source' and 'target', got: {edge_df.columns.tolist()}")
            raise ValueError("Edge table must have 'source' and 'target' columns")
        
        # Get edge attribute columns (everything except source and target)
        edge_attr_cols = [col for col in edge_df.columns if col not in ['source', 'target']]
        
        # Create graph from edge list
        if edge_attr_cols:
            G = nx.from_pandas_edgelist(
                edge_df, 
                source='source', 
                target='target', 
                edge_attr=edge_attr_cols
            )
        else:
            G = nx.from_pandas_edgelist(
                edge_df, 
                source='source', 
                target='target'
            )
        
        log.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Add all node attributes from node_table
        for node_id, row in node_table.iterrows():
            if node_id in G.nodes():
                # Add all attributes from the row to the node
                for col, value in row.items():
                    G.nodes[node_id][col] = value
        
        # Determine color attribute based on whether submodule info exists
        if 'submodule_color' in node_table.columns and not node_table['submodule_color'].isna().all():
            color_attr = "submodule_color"
        else:
            color_attr = "datatype_color"
        
        log.info("Rendering existing network visualization...")
        log.info("Pre-computing network layout...")
        
        # Create the interactive widget
        widget = _nx_to_plotly_widget(
            G,
            node_color_attr=color_attr,
            node_size_attr="node_size",
            layout=interactive_layout,
            seed=42,
        )
        
        # Display the widget
        display(widget)
        
    except Exception as e:
        log.error(f"Error displaying existing network: {e}")
        log.info("Network files exist but could not be displayed. You may need to regenerate the network.")
        # Print debug info
        log.info(f"Original edge table columns: {edge_table.columns.tolist()}")
        log.info(f"Node table columns: {node_table.columns.tolist()}")
        log.info(f"Edge table head:\n{edge_table.head()}")
        raise e

def _nx_to_plotly_widget(
    G,
    node_color_attr="submodule_color",
    node_size_attr="node_size",
    node_shape_attr="datatype_shape",
    layout=None,
    seed=42,
):
    """
    Convert a NetworkX graph to a Plotly FigureWidget.

    Parameters
    ----------
    G : networkx.Graph
        Graph to visualise.
    node_color_attr : str, optional
        Node attribute containing a CSS colour string.
    node_size_attr : str, optional
        Node attribute containing a numeric size (in pts).
    layout : {"spring","circular","kamada_kawai","random"}, optional
        Layout algorithm used to compute (x, y) coordinates.
    seed : int, optional
        Random seed for deterministic layouts (spring & random).

    Returns
    -------
    plotly.graph_objects.FigureWidget
        Interactive widget ready for display in JupyterLab.
    """
    # Compute node positions
    log.info(f"Using layout '{layout}' for interactive Plotly network.")
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed, k=10/np.sqrt(len(G.nodes())), weight="weight", iterations=100)
    elif layout == "bipartite":
        pos = nx.bipartite_layout(G, nodes=[n for n, d in G.nodes(data=True) if d.get("datatype_shape") == "circle"])
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(G)
    elif layout == "pydot":
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    elif layout == "force":
        pos = nx.forceatlas2_layout(G, store_pos_as="pos")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "pydot":
        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        pos = nx.nx_pydot.pydot_layout(H, prog="neato")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G, seed=seed)
    else:
        raise ValueError(f"Unsupported layout `{layout}`")

    # Build edge trace
    log.info("Building edge traces...")
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        showlegend=False,
    )

    # Build node trace (colour / size / hover text)
    log.info("Building node traces...")
    node_x, node_y, node_color, node_size, node_shape, hover_txt = [], [], [], [], [], []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_color.append(data.get(node_color_attr, "#1f78b4"))
        node_shape.append(data.get(node_shape_attr, "Circle"))
        node_size.append(data.get(node_size_attr, 10))

        # Build hover text with node annotation
        hover_parts = [str(n)]
        
        # Add submodule if present
        submodule = data.get("submodule", None)
        if submodule:
            hover_parts.append(f"{submodule.replace('_','')}")
        
        # Determine which annotation to show based on node prefix
        annotation_text = "Unassigned_Annotation"
        display_text = "Unassigned_Name"
        
        # Check for metabolomics compound annotation and name
        if str(n).startswith('mx_'):
            mx_compound_name = data.get("mx_Compound_Name", "Unassigned")
            mx_display_name = data.get('mx_display_name', "Unassigned")
            if mx_compound_name != "Unassigned":
                annotation_text = mx_compound_name
            if mx_display_name != "Unassigned":
                display_text = mx_display_name

        # Check for transcriptomics annotation and name
        elif str(n).startswith('tx_'):
            tx_go_acc = data.get("tx_go_acc", "Unassigned")
            tx_display_name = data.get('tx_display_name', "Unassigned")
            if tx_go_acc != "Unassigned":
                annotation_text = tx_go_acc
            if tx_display_name != "Unassigned":
                display_text = tx_display_name
        
        # Add annotation to hover text
        hover_parts.append(f"{annotation_text}")
        hover_parts.append(f"{display_text}")
        hover_txt.append("<br>".join(hover_parts))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(
            size=node_size,
            color=node_color,
            symbol=node_shape,
            opacity=0.9,
            line=dict(width=1, color="#222"),
        ),
        hoverinfo="text",
        text=hover_txt,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        customdata=list(G.nodes()),
    )

    # Assemble figure widget
    log.info("Assembling network widget...")
    fig = go.FigureWidget(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Network graph (interactive)",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=900,
            width=900,
            clickmode="event+select",
        ),
    )
    return fig

def _annotate_and_save_submodules(
    submodules: List[Tuple[str, nx.Graph]],
    main_graph: nx.Graph,
    output_filenames: Dict[str, str],
    integrated_data: pd.DataFrame,
    metadata: pd.DataFrame,
    save_plots: bool = True,
) -> None:
    """
    - Annotates every node in ``main_graph`` with ``submodule`` and its color.
    - Writes each submodule as its own GraphML + node/edge CSV.
    - (Optionally) draws a violin-strip plot of the *mean* abundance of all
      members of the submodule across the group variable in ``metadata``.
    """
    # colors for submodules (shuffle for visual separation)
    n_mod = len(submodules)
    sub_cols = viridis(np.linspace(0, 1, n_mod))
    sub_hex = [to_hex(c) for c in sub_cols]
    random.shuffle(sub_hex)

    sub_path = output_filenames.get("submodule_path", "submodules")
    os.makedirs(sub_path, exist_ok=True)

    graph_name = os.path.basename(output_filenames["graph"]).replace(".graphml", "")

    for idx, (mod_name, sub_g) in enumerate(submodules, start=1):
        # annotate nodes in main_graph and subgraph
        col = sub_hex[(idx - 1) % len(sub_hex)]
        for n in sub_g.nodes():
            main_graph.nodes[n]["submodule"] = mod_name
            main_graph.nodes[n]["submodule_color"] = col
            sub_g.nodes[n]["submodule"] = mod_name
            sub_g.nodes[n]["submodule_color"] = col

        # write submodule files
        subgraph_file = f"{sub_path}/{graph_name}_submodule{idx}.graphml"
        nx.write_graphml(sub_g, subgraph_file)

        node_tbl = pd.DataFrame.from_dict(dict(sub_g.nodes(data=True)), orient="index")
        edge_tbl = nx.to_pandas_edgelist(sub_g)
        node_tbl.to_csv(f"{sub_path}/{graph_name}_submodule{idx}_nodes.csv")
        edge_tbl.to_csv(f"{sub_path}/{graph_name}_submodule{idx}_edges.csv")

        # abundance plot
        if save_plots:
            # mean abundance of all members of the submodule per sample
            members = list(sub_g.nodes())
            # intersect with integrated_data columns (some nodes might have been filtered out)
            members = [m for m in members if m in integrated_data.columns]
            if not members:
                continue
            mean_abund = integrated_data[members].mean(axis=1).rename("abundance")
            plot_df = pd.concat([mean_abund, metadata], axis=1, join="inner")
            plt.figure(figsize=(10, 6))
            sns.violinplot(
                x="group", y="abundance", data=plot_df,
                inner=None, palette="viridis"
            )
            sns.stripplot(
                x="group", y="abundance", data=plot_df,
                color="k", alpha=0.5, size=3
            )
            plt.title(f"Mean abundance of {mod_name}")
            plt.xlabel("group")
            plt.ylabel("mean abundance")
            plt.tight_layout()
            plt.savefig(
                f"{sub_path}/{graph_name}_submodule{idx}_abundance.pdf",
                dpi=300,
            )
            plt.close()


##############################################
## Comparing Network Topologies
##############################################

def _try_load_existing_networks(
    output_dir: str,
    output_filename_prefix: str,
    feature_prefixes: List[str]
) -> Dict[str, nx.Graph]:
    """
    Try to load existing networks from disk.
    
    Returns empty dict if any required network files are missing.
    """
    networks = {}
    
    # Define expected network modes
    mode_names = ['full'] + [f"{prefix.rstrip('_')}_only" for prefix in feature_prefixes]
    
    # Check if all required network files exist
    for mode_name in mode_names:
        graphml_path = os.path.join(
            output_dir, 
            f"{output_filename_prefix}_{mode_name}_network.graphml"
        )
        
        if not os.path.exists(graphml_path):
            log.info(f"Network file not found. Proceeding with correlation and network construction.")
            return {}  # Return empty dict if any file missing
    
    # All files exist, load them
    log.info("Found existing network files, loading...")
    for mode_name in mode_names:
        graphml_path = os.path.join(
            output_dir, 
            f"{output_filename_prefix}_{mode_name}_network.graphml"
        )
        
        try:
            G = nx.read_graphml(graphml_path)
            
            # Convert node attributes back to proper types if needed
            # (GraphML stores everything as strings)
            for node in G.nodes():
                if 'node_size' in G.nodes[node]:
                    G.nodes[node]['node_size'] = int(G.nodes[node]['node_size'])
            
            networks[mode_name] = G
            log.info(f"Loaded {mode_name}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            
        except Exception as e:
            log.warning(f"Error loading {graphml_path}: {e}")
            return {}  # Return empty if any loading fails
    
    return networks

def _save_networks_to_disk(
    networks: Dict[str, nx.Graph],
    output_dir: str,
    output_filename_prefix: str
) -> None:
    """
    Save networks to disk as GraphML and CSV tables.
    
    Only saves essential files needed for downstream analysis.
    """
    log.info("Saving networks to disk for future reuse...")
    os.makedirs(output_dir, exist_ok=True)
    
    for mode_name, G in networks.items():
        base_filename = f"{output_filename_prefix}_{mode_name}"
        
        # Save GraphML (most important - preserves graph structure and all attributes)
        graphml_path = os.path.join(output_dir, f"{base_filename}_network.graphml")
        nx.write_graphml(G, graphml_path)
        log.info(f"Saved {mode_name} network to {graphml_path}")
        
        # Save node and edge tables (useful for quick inspection)
        node_table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
        edge_table = nx.to_pandas_edgelist(G)
        
        write_integration_file(
            node_table,
            output_dir,
            f"{base_filename}_nodes",
            indexing=True,
            index_label="node_id"
        )
        write_integration_file(
            edge_table,
            output_dir,
            f"{base_filename}_edges",
            indexing=False
        )

def compare_network_topologies(
    integrated_data: pd.DataFrame,
    feature_prefixes: List[str] = None,
    correlation_params: dict = None,
    network_params: dict = None,
    annotation_input: pd.DataFrame = None,
    output_dir: str = None,
    output_filename_prefix: str = "network_comparison",
    overwrite: bool = False,
    plot_interactive: bool = False,
) -> Dict[str, Any]:
    """
    Compare network topologies between independent and integrated multi-omics datasets.
        
    Returns:
        Dictionary containing networks, statistics, and analysis results
    """
    
    # Initialize results structure
    results = {
        'topology_comparison': {},
        'preservation_analysis': {},
        'cross_omics_connectivity': {},
        'conditional_entropy': {},
        'motif_analysis': {},
        'submodule_enrichment': {},
        'edge_enrichment': {},
    }
    
    # Default annotations for downstream tests
    annotation_categories = {
        'tx': ['tx_go_acc'],
        'mx': ['mx_subclass']
    }

    # Try to load existing networks
    networks = _try_load_existing_networks(
        output_dir, 
        output_filename_prefix, 
        feature_prefixes
    )
    if networks and overwrite is False:
        log.info(f"Loaded {len(networks)} existing networks from disk")
    else:
        # Calculate correlations
        log.info("=" * 80)
        log.info("Calculating Correlation Matrices")
        log.info("=" * 80)
        correlation_tables = _calculate_all_correlations(
            integrated_data=integrated_data,
            feature_prefixes=feature_prefixes,
            correlation_params=correlation_params,
            output_filename_prefix=output_filename_prefix
        )
        
        # Build networks from correlations
        log.info("=" * 80)
        log.info("Creating Networks")
        log.info("=" * 80)
        networks = _build_all_networks(
            correlation_tables=correlation_tables,
            feature_prefixes=feature_prefixes,
            network_params=network_params,
            annotation_input=annotation_input,
            output_dir=output_dir,
            output_filename_prefix=output_filename_prefix
        )
    
    # Compare network topologies
    log.info("=" * 80)
    log.info("Network Topology Comparison")
    log.info("=" * 80)
    topology_stats = _compare_network_structures(
        networks=networks,
        feature_prefixes=feature_prefixes,
    )
    results['topology_comparison'] = topology_stats
    
    # Analyze submodule preservation
    log.info("=" * 80)
    log.info("Submodule Preservation Analysis")
    log.info("=" * 80)
    preservation_stats = _analyze_submodule_preservation(
        networks=networks,
        feature_prefixes=feature_prefixes,
    )
    results['preservation_analysis'] = preservation_stats
    
    # Cross-omics connectivity
    log.info("=" * 80)
    log.info("Cross-omics Connectivity Analysis")
    log.info("=" * 80)
    xomics_conn = analyze_cross_omics_connectivity(
        full_network=networks['full'],
        feature_prefixes=feature_prefixes
    )
    results['cross_omics_connectivity'] = xomics_conn
    
    # Conditional entropy reduction
    log.info("=" * 80)
    log.info("Conditional Entropy Reduction")
    log.info("=" * 80)
    conditional_entropy = calculate_conditional_entropy_reduction(
        full_network=networks['full'],
        independent_networks=networks,
        feature_prefixes=feature_prefixes
    )
    results['conditional_entropy'] = conditional_entropy
    
    # Motif analysis
    log.info("=" * 80)
    log.info("Cross-omics Motif Analysis")
    log.info("=" * 80)
    motif_analysis = analyze_cross_omics_motifs(
        full_network=networks['full'],
        feature_prefixes=feature_prefixes
    )
    results['motif_analysis'] = motif_analysis
    
    # Perform submodule enrichment analysis
    log.info("=" * 80)
    log.info("Submodule Enrichment Analysis")
    log.info("=" * 80)
    submod_enrichment_results = analyze_submodule_functional_enrichment(
        networks=networks,
        annotation_categories=annotation_categories,
    )
    results['submodule_enrichment'] = submod_enrichment_results
    
    # Perform edge enrichment analysis
    log.info("=" * 80)
    log.info("Edge Enrichment Analysis")
    log.info("=" * 80)
    edge_enrichment_results = analyze_cross_omics_edge_enrichment(
        full_network=networks['full'],
        annotation_categories=annotation_categories,
        feature_prefixes=feature_prefixes,
    )
    results['edge_enrichment'] = edge_enrichment_results

    # Generate independent plots
    log.info("=" * 80)
    log.info("Generating Independent Networks")
    log.info("=" * 80)
    plot_independent_networks(
        networks=networks,
        feature_prefixes=feature_prefixes,
        layout=network_params.get('interactive_layout', 'spring')
    )

    # Generate network comparison plots
    log.info("=" * 80)
    log.info("Generating Comparison Network")
    log.info("=" * 80)
    plot_network_comparison(
        networks=networks,
        feature_prefixes=feature_prefixes,
        layout=network_params.get('interactive_layout', 'spring')
    )

    if plot_interactive:
        widget = plot_interactive_network_comparison(
            networks=networks,
            feature_prefixes=feature_prefixes,
            layout=network_params.get('interactive_layout', 'spring')
        )
        display(widget)

    log.info("Analysis complete")
    
    return

def _calculate_all_correlations(
    integrated_data: pd.DataFrame,
    feature_prefixes: List[str],
    correlation_params: dict,
    output_filename_prefix: str
) -> Dict[str, pd.DataFrame]:
    """Calculate correlations for all network types (independent + full)."""
    
    log.info("Calculating feature correlations...")
    correlation_tables = {}
    
    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="network_comparison_") as temp_dir:
        
        # Calculate independent dataset correlations
        for prefix in feature_prefixes:
            dataset_name = prefix.rstrip('_')
            log.info(f"Computing {dataset_name} correlations...")
            
            corr_table = calculate_correlated_features(
                data=integrated_data,
                output_filename=f"{output_filename_prefix}_{dataset_name}_correlations",
                output_dir=temp_dir,
                feature_prefixes=feature_prefixes,
                corr_mode=dataset_name,
                method=correlation_params.get('corr_method', 'pearson'),
                cutoff=correlation_params.get('corr_cutoff', 0.75),
                keep_negative=correlation_params.get('keep_negative', False),
                block_size=correlation_params.get('block_size', 500),
                n_jobs=correlation_params.get('cores', -1),
            )
            
            if not corr_table.empty:
                correlation_tables[f"{dataset_name}_only"] = corr_table
                log.info(f"Found {len(corr_table):,} correlations")
        
        # Calculate full integrated correlations
        log.info("Computing full integrated correlations...")
        full_corr_table = calculate_correlated_features(
            data=integrated_data,
            output_filename=f"{output_filename_prefix}_full_correlations",
            output_dir=temp_dir,
            feature_prefixes=feature_prefixes,
            corr_mode="full",
            method=correlation_params.get('corr_method', 'pearson'),
            cutoff=correlation_params.get('corr_cutoff', 0.75),
            keep_negative=correlation_params.get('keep_negative', False),
            block_size=correlation_params.get('block_size', 500),
            n_jobs=correlation_params.get('cores', -1),
        )
        
        if not full_corr_table.empty:
            correlation_tables['full'] = full_corr_table
            log.info(f"Found {len(full_corr_table):,} correlations")
    
    return correlation_tables

def _build_all_networks(
    correlation_tables: Dict[str, pd.DataFrame],
    feature_prefixes: List[str],
    network_params: dict,
    annotation_input: pd.DataFrame,
    output_dir: str,
    output_filename_prefix: str = "networks"
) -> Dict[str, nx.Graph]:
    """Build networks from correlation tables."""
    
    log.info("Building networks from correlations...")
    networks = {}
    
    for mode_name, corr_table in correlation_tables.items():
        log.info(f"Building {mode_name} network...")
        
        # Create graph from correlation table
        G = _build_single_network(
            corr_table=corr_table,
            feature_prefixes=feature_prefixes,
            annotation_input=annotation_input
        )
        
        # Detect submodules
        submodule_mode = network_params['submodule_mode']
        if submodule_mode != 'none':
            log.info(f"Detecting submodules using '{submodule_mode}'...")
            G = _annotate_submodules(
                G=G,
                submodule_mode=submodule_mode,
                wgcna_params=network_params['wgcna_params']
            )
        
        networks[mode_name] = G
        log.info(f"Final: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    _save_networks_to_disk(networks, output_dir, output_filename_prefix)

    return networks

def _build_single_network(
    corr_table: pd.DataFrame,
    feature_prefixes: List[str],
    annotation_input: pd.DataFrame
) -> nx.Graph:
    """Build a single network from a correlation table."""
    
    # Get all unique features
    all_features = pd.Index(sorted(
        set(corr_table["feature_1"]).union(set(corr_table["feature_2"]))
    ))
    
    # Pivot to matrix format
    correlation_df = corr_table.pivot(
        index="feature_2",
        columns="feature_1",
        values="correlation"
    ).reindex(index=all_features, columns=all_features, fill_value=0.0)
    
    # Build sparse adjacency and create graph
    sparse_adj = _build_sparse_adj(correlation_df, feature_prefixes)
    G = _graph_from_sparse(sparse_adj, correlation_df.index.to_numpy())
    
    # Add node attributes
    color_map, shape_map = _make_prefix_maps(feature_prefixes)
    _assign_node_attributes(G, feature_prefixes, color_map, shape_map, annotation_input)
    
    # Remove tiny components
    tiny = [c for c in nx.connected_components(G) if len(c) < 3]
    if tiny:
        G.remove_nodes_from({n for comp in tiny for n in comp})
    
    return G

def _annotate_submodules(
    G: nx.Graph,
    submodule_mode: str,
    wgcna_params: dict
) -> nx.Graph:
    """Detect and annotate submodules in a network."""
    
    submods = _detect_submodules(G, method=submodule_mode, **wgcna_params)
    
    if submods:
        n_mod = len(submods)
        sub_cols = viridis(np.linspace(0, 1, n_mod))
        sub_hex = [to_hex(c) for c in sub_cols]
        
        for idx, (mod_name, sub_g) in enumerate(submods, start=1):
            col = sub_hex[(idx - 1) % len(sub_hex)]
            for n in sub_g.nodes():
                G.nodes[n]["submodule"] = mod_name
                G.nodes[n]["submodule_color"] = col
        
        log.info(f"Detected {len(submods)} submodules")
    else:
        # Assign all to single submodule
        for node in G.nodes():
            G.nodes[node]["submodule"] = "submodule_1"
            G.nodes[node]["submodule_color"] = "#440154"
    
    return G

def _compare_network_structures(
    networks: Dict[str, nx.Graph],
    feature_prefixes: List[str],
) -> Dict[str, Any]:
    """Compare network structures between independent and integrated networks."""
    
    full_network = networks['full']
    comparison_stats = {
        'network_sizes': {},
        'gained_elements': {},
        'lost_elements': {},
        'overlap_percentages': {}
    }
    
    # Network sizes
    for mode_name, G in networks.items():
        comparison_stats['network_sizes'][mode_name] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        }
    
    # Compare each independent network to full network
    for prefix in feature_prefixes:
        dataset_name = prefix.rstrip('_')
        mode_name = f"{dataset_name}_only"
        
        if mode_name not in networks:
            continue
        
        log.info(f"Comparing {dataset_name} vs full network:")
        
        independent_network = networks[mode_name]
        
        # Calculate gained and lost elements
        gained, lost = _calculate_network_differences(
            independent_network, full_network, feature_prefixes
        )
        
        comparison_stats['gained_elements'][dataset_name] = gained
        comparison_stats['lost_elements'][dataset_name] = lost
        
        # Calculate overlaps
        node_overlap = len(set(independent_network.nodes()) & set(full_network.nodes()))
        edge_overlap = len(set(independent_network.edges()) & set(full_network.edges()))
        
        comparison_stats['overlap_percentages'][dataset_name] = {
            'nodes': (node_overlap / full_network.number_of_nodes() * 100) 
                     if full_network.number_of_nodes() > 0 else 0,
            'edges': (edge_overlap / full_network.number_of_edges() * 100) 
                     if full_network.number_of_edges() > 0 else 0
        }
    
        log.info(f"Elements GAINED in full network:")
        log.info(f"Nodes: {len(gained['nodes']):,}")
        for node_type, count in gained['node_counts'].items():
            log.info(f"{node_type}: {count:,}")
        log.info(f"Edges: {len(gained['edges']):,}")
        for edge_type, count in gained['edge_counts'].items():
            log.info(f"{edge_type}: {count:,}")
        log.info(f"Elements LOST from independent network:")
        log.info(f"Nodes: {len(lost['nodes']):,}")
        for node_type, count in lost['node_counts'].items():
            log.info(f"{node_type}: {count:,}")
        log.info(f"Edges: {len(lost['edges']):,}")
        for edge_type, count in lost['edge_counts'].items():
            log.info(f"{edge_type}: {count:,}")
    
    return comparison_stats

def _calculate_network_differences(
    independent_network: nx.Graph,
    full_network: nx.Graph,
    feature_prefixes: List[str]
) -> Tuple[Dict, Dict]:
    """Calculate nodes and edges gained/lost between networks."""
    
    # Nodes
    nodes_gained = set(full_network.nodes()) - set(independent_network.nodes())
    nodes_lost = set(independent_network.nodes()) - set(full_network.nodes())
    
    # Edges (accounting for undirected graph)
    edges_gained = set(full_network.edges()) - set(independent_network.edges())
    edges_gained = {e for e in edges_gained 
                    if (e[1], e[0]) not in independent_network.edges()}
    
    edges_lost = set(independent_network.edges()) - set(full_network.edges())
    edges_lost = {e for e in edges_lost 
                  if (e[1], e[0]) not in full_network.edges()}
    
    # Categorize by feature type
    gained = {
        'nodes': list(nodes_gained),
        'edges': list(edges_gained),
        'node_counts': _count_by_prefix(nodes_gained, feature_prefixes),
        'edge_counts': _count_edge_types(edges_gained, feature_prefixes)
    }
    
    lost = {
        'nodes': list(nodes_lost),
        'edges': list(edges_lost),
        'node_counts': _count_by_prefix(nodes_lost, feature_prefixes),
        'edge_counts': _count_edge_types(edges_lost, feature_prefixes)
    }
    
    return gained, lost

def _count_by_prefix(
    nodes: set,
    feature_prefixes: List[str]
) -> Dict[str, int]:
    """Count nodes by feature type prefix."""
    counts = {}
    for prefix in feature_prefixes:
        prefix_name = prefix.rstrip('_')
        counts[prefix_name] = sum(1 for n in nodes if n.startswith(prefix))
    return counts

def _count_edge_types(
    edges: set,
    feature_prefixes: List[str]
) -> Dict[str, int]:
    """Count edges by connection type (within/between data types)."""
    counts = {}
    
    # Get prefix pairs for classification
    prefixes = [p.rstrip('_') for p in feature_prefixes]
    
    for i, prefix1 in enumerate(prefixes):
        # Within-type edges
        within_count = sum(1 for u, v in edges 
                          if u.startswith(feature_prefixes[i]) and 
                             v.startswith(feature_prefixes[i]))
        counts[f'within_{prefix1}'] = within_count
        
        # Between-type edges
        for prefix2 in prefixes[i+1:]:
            between_count = sum(1 for u, v in edges 
                               if (u.startswith(f'{prefix1}_') and v.startswith(f'{prefix2}_')) or
                                  (u.startswith(f'{prefix2}_') and v.startswith(f'{prefix1}_')))
            counts[f'{prefix1}_to_{prefix2}'] = between_count
    
    return counts

def _analyze_submodule_preservation(
    networks: Dict[str, nx.Graph],
    feature_prefixes: List[str],
) -> Dict[str, Dict]:
    """Analyze how submodules are preserved between networks."""
    
    preservation_results = {}
    
    for prefix in feature_prefixes:
        dataset_name = prefix.rstrip('_')
        mode_name = f"{dataset_name}_only"
        
        if mode_name not in networks:
            continue
        
        log.info(f"Analyzing {dataset_name} submodule preservation...")
        
        # Run preservation analyses
        preservation_results[dataset_name] = {
            'jaccard_scores': calculate_submodule_preservation(
                networks[mode_name], networks['full'], 
                f"{dataset_name}_", feature_prefixes
            ),
            'permutation_test': permutation_test_preservation(
                networks[mode_name], networks['full'], 
                f"{dataset_name}_", n_permutations=1000
            )
        }

    return preservation_results

def analyze_submodule_functional_enrichment(
    networks: Dict[str, nx.Graph],
    annotation_categories: Dict[str, List[str]],
    p_value_threshold: float = 0.05,
    correction_method: str = "fdr_bh",
    min_annotation_count: int = 2
) -> Dict[str, Any]:
    """
    Perform functional enrichment analysis on network submodules.
    
    Args:
        networks: Dictionary of networks (must include 'full' and independent networks)
        annotation_categories: Dict mapping dataset names to lists of annotation column names
                             e.g., {'tx': ['tx_go_acc', 'tx_kegg_acc'], 
                                    'mx': ['mx_npclassifier_class']}
        p_value_threshold: Significance threshold for enrichment
        correction_method: Multiple testing correction method
        min_annotation_count: Minimum count for an annotation to be tested
        
    Returns:
        Dictionary with enrichment analysis results
    """
    
    enrichment_results = {}
    
    for dataset_name, annotation_columns in annotation_categories.items():
        log.info(f"Analyzing functional enrichment for {dataset_name}...")
        
        # Get networks
        indep_name = f"{dataset_name}_only"
        if indep_name not in networks:
            log.warning(f"Independent network {indep_name} not found, skipping")
            continue
        
        independent_network = networks[indep_name]
        full_network = networks['full']
        
        # Get node tables with annotations
        indep_node_table = pd.DataFrame.from_dict(
            dict(independent_network.nodes(data=True)), 
            orient="index"
        )
        full_node_table = pd.DataFrame.from_dict(
            dict(full_network.nodes(data=True)), 
            orient="index"
        )
        
        # Run enrichment for each annotation category
        dataset_results = {}
        for annotation_col in annotation_columns:
            log.info(f"Testing enrichment for annotation: {annotation_col}")
            
            # Skip if column not in node tables
            if annotation_col not in indep_node_table.columns:
                log.warning(f"Column {annotation_col} not in independent network, skipping")
                continue
            if annotation_col not in full_node_table.columns:
                log.warning(f"Column {annotation_col} not in full network, skipping")
                continue
            
            # Run enrichment on independent network
            log.info(f"Running enrichment on independent network...")
            indep_enrichment = perform_functional_enrichment(
                node_table=indep_node_table,
                annotation_column=annotation_col,
                p_value_threshold=p_value_threshold,
                correction_method=correction_method,
                min_annotation_count=min_annotation_count,
                output_dir=None
            )
            
            # Run enrichment on full network
            log.info(f"Running enrichment on full network...")
            full_enrichment = perform_functional_enrichment(
                node_table=full_node_table,
                annotation_column=annotation_col,
                p_value_threshold=p_value_threshold,
                correction_method=correction_method,
                min_annotation_count=min_annotation_count,
                output_dir=None
            )
            
            # Compare results
            comparison = _compare_enrichment_results(
                indep_enrichment,
                full_enrichment,
                dataset_name,
                annotation_col
            )
            
            dataset_results[annotation_col] = {
                'independent': indep_enrichment,
                'full': full_enrichment,
                'comparison': comparison
            }
        
        enrichment_results[dataset_name] = dataset_results
    
    return enrichment_results

def analyze_cross_omics_edge_enrichment(
    full_network: nx.Graph,
    annotation_categories: Dict[str, List[str]],
    feature_prefixes: List[str],
    p_value_threshold: float = 0.05,
    correction_method: str = "fdr_bh",
    min_annotation_count: int = 2
) -> Dict[str, Any]:
    """
    Test if annotations are enriched in nodes with cross-omics edges.
    
    Args:
        full_network: Full integrated network with node annotations
        annotation_categories: Dict mapping dataset names to annotation column lists
                              e.g., {'tx': ['tx_go_acc', 'tx_kegg_acc'], 
                                     'mx': ['mx_class', 'mx_subclass']}
        feature_prefixes: List of feature prefixes (e.g., ['tx_', 'mx_'])
        p_value_threshold: Significance threshold
        correction_method: Multiple testing correction method
        min_annotation_count: Minimum count for testing an annotation
        
    Returns:
        Dictionary with enrichment results per dataset and annotation category
    """
    
    # Get edge table
    edge_table = nx.to_pandas_edgelist(full_network)
    
    # Categorize nodes by whether they have cross-omics edges
    nodes_with_cross_edges = set()
    nodes_without_cross_edges = set()
    
    for _, row in edge_table.iterrows():
        source = row['source']
        target = row['target']
        
        # Check if edge connects different data types
        source_type = next((p for p in feature_prefixes if source.startswith(p)), None)
        target_type = next((p for p in feature_prefixes if target.startswith(p)), None)
        
        if source_type != target_type:
            # Cross-omics edge
            nodes_with_cross_edges.add(source)
            nodes_with_cross_edges.add(target)
    
    # All nodes in network that DON'T have cross-omics edges
    all_nodes = set(full_network.nodes())
    nodes_without_cross_edges = all_nodes - nodes_with_cross_edges
    
    log.info(f"Network edge statistics:")
    log.info(f"Total nodes: {len(all_nodes)}")
    log.info(f"Nodes with cross-omics edges: {len(nodes_with_cross_edges)}")
    log.info(f"Nodes without cross-omics edges: {len(nodes_without_cross_edges)}")
    
    # Get node table with annotations
    node_table = pd.DataFrame.from_dict(
        dict(full_network.nodes(data=True)), 
        orient="index"
    )
    
    # Perform enrichment for each dataset and annotation category
    enrichment_results = {}
    
    for dataset_name, annotation_columns in annotation_categories.items():
        log.info(f"Analyzing {dataset_name} annotations...")
        dataset_results = {}
        
        for annotation_col in annotation_columns:
            log.info(f"Testing enrichment for annotation: {annotation_col}")
            # Filter to nodes from this dataset
            dataset_prefix = f"{dataset_name}_"
            dataset_nodes = node_table[node_table.index.str.startswith(dataset_prefix)].copy()
            
            # Add cross-edge status
            dataset_nodes['has_cross_edges'] = dataset_nodes.index.isin(nodes_with_cross_edges)
            
            # Run enrichment test
            enrichment_df = _test_annotation_enrichment_in_cross_edges(
                node_table=dataset_nodes,
                annotation_column=annotation_col,
                p_value_threshold=p_value_threshold,
                correction_method=correction_method,
                min_annotation_count=min_annotation_count
            )
            
            if not enrichment_df.empty:
                dataset_results[annotation_col] = enrichment_df
                
                # Log significant results
                sig_results = enrichment_df[enrichment_df['significant']]
                if not sig_results.empty:
                    log.info(f"Found {len(sig_results)} significant enrichments")
                    log.info(f"Top enrichments of {annotation_col}:")
                    for _, row in sig_results.head(20).iterrows():
                        log.info(f"{row['annotation_term']}: "
                                 f"Nodes with cross edges={row['nodes_with_cross_edges']}, "
                                 f"Nodes without cross edges={row['nodes_without_cross_edges']}, "
                                 f"Fold Enrichment={row['fold_enrichment']:.2f}, "
                                 f"Corrected p-value={row['p_value_corr']:.2e}")
                else:
                    log.info(f"No significant enrichment of {annotation_col} found")
        
        if dataset_results:
            enrichment_results[dataset_name] = dataset_results
    
    return enrichment_results

def _compare_enrichment_results(
    indep_enrichment: pd.DataFrame,
    full_enrichment: pd.DataFrame,
    dataset_name: str,
    annotation_col: str
) -> Dict[str, Any]:
    """
    Compare enrichment results between independent and full networks.
    
    Now compares by annotation_term only (not submodule), since submodule IDs
    can differ between networks even when representing similar biological groups.
    """
    
    log.info(f"Comparing enrichment results...")
    
    comparison = {
        'dataset': dataset_name,
        'annotation_category': annotation_col
    }
    
    # Basic statistics
    comparison['n_tests_independent'] = len(indep_enrichment)
    comparison['n_tests_full'] = len(full_enrichment)
    comparison['n_significant_independent'] = indep_enrichment['significant'].sum() if not indep_enrichment.empty else 0
    comparison['n_significant_full'] = full_enrichment['significant'].sum() if not full_enrichment.empty else 0
    
    # Identify gained and lost enrichments BY ANNOTATION TERM ONLY
    # (ignoring submodule ID since they can differ between networks)
    if not indep_enrichment.empty and not full_enrichment.empty:
        # Get unique significant annotation terms (not caring about which submodule)
        indep_sig_terms = set(
            indep_enrichment[indep_enrichment['significant']]['annotation_term'].unique()
        )
        
        full_sig_terms = set(
            full_enrichment[full_enrichment['significant']]['annotation_term'].unique()
        )
        
        gained_terms = full_sig_terms - indep_sig_terms
        lost_terms = indep_sig_terms - full_sig_terms
        preserved_terms = indep_sig_terms & full_sig_terms
        
        comparison['n_gained_enrichments'] = len(gained_terms)
        comparison['n_lost_enrichments'] = len(lost_terms)
        comparison['n_preserved_enrichments'] = len(preserved_terms)
        
        # Store the actual terms for inspection
        comparison['gained_enrichment_terms'] = list(gained_terms)
        comparison['lost_enrichment_terms'] = list(lost_terms)
        comparison['preserved_enrichment_terms'] = list(preserved_terms)
        
        # Log summary
        log.info(f"Enrichment comparison summary (by annotation term):")
        log.info(f"Gained enrichments: {comparison['n_gained_enrichments']}")
        if gained_terms:
            log.info(f"Terms: {', '.join(sorted(gained_terms))}")
        log.info(f"Lost enrichments: {comparison['n_lost_enrichments']}")
        if lost_terms:
            log.info(f"Terms: {', '.join(sorted(lost_terms))}")
        log.info(f"Preserved enrichments: {comparison['n_preserved_enrichments']}")
        if preserved_terms:
            log.info(f"Terms: {', '.join(sorted(preserved_terms))}")
    
    else:
        comparison['n_gained_enrichments'] = 0
        comparison['n_lost_enrichments'] = 0
        comparison['n_preserved_enrichments'] = 0
        comparison['gained_enrichment_terms'] = []
        comparison['lost_enrichment_terms'] = []
        comparison['preserved_enrichment_terms'] = []
    
    return comparison

def _test_annotation_enrichment_in_cross_edges(
    node_table: pd.DataFrame,
    annotation_column: str,
    p_value_threshold: float,
    correction_method: str,
    min_annotation_count: int
) -> pd.DataFrame:
    """
    Test if each annotation term is enriched in nodes with cross-omics edges.
    
    Simple Fisher's exact test:
    - Rows: [has cross edges, no cross edges]
    - Cols: [has annotation, no annotation]
    """
    
    # Parse annotations (handle semicolon-separated values)
    annotation_data = []
    for node_id, row in node_table.iterrows():
        has_cross = row['has_cross_edges']
        annotations_str = row[annotation_column]
        
        if pd.isna(annotations_str) or str(annotations_str).strip() in ['', 'Unassigned', 'NA']:
            continue
        
        # Split on ;; and clean
        annotations = [a.strip() for a in str(annotations_str).split(';;') 
                      if a.strip() and a.strip() not in ['Unassigned', 'NA', '']]
        
        for ann in annotations:
            annotation_data.append({
                'node_id': node_id,
                'has_cross_edges': has_cross,
                'annotation': ann
            })
    
    if not annotation_data:
        return pd.DataFrame()
    
    ann_df = pd.DataFrame(annotation_data)
    
    # Filter by minimum count
    annotation_counts = ann_df['annotation'].value_counts()
    valid_annotations = annotation_counts[annotation_counts >= min_annotation_count].index
    ann_df = ann_df[ann_df['annotation'].isin(valid_annotations)]
    
    if ann_df.empty:
        return pd.DataFrame()
    
    # Count nodes in each category
    total_with_cross = node_table['has_cross_edges'].sum()
    total_without_cross = (~node_table['has_cross_edges']).sum()
    
    # Test each annotation
    results = []
    for annotation in valid_annotations:
        nodes_with_ann = set(ann_df[ann_df['annotation'] == annotation]['node_id'].unique())
        
        # Build 2x2 table
        with_cross_and_ann = sum(node_table.loc[node_table.index.isin(nodes_with_ann), 'has_cross_edges'])
        with_cross_no_ann = total_with_cross - with_cross_and_ann
        without_cross_and_ann = len(nodes_with_ann) - with_cross_and_ann
        without_cross_no_ann = total_without_cross - without_cross_and_ann
        
        contingency_table = np.array([
            [with_cross_and_ann, with_cross_no_ann],
            [without_cross_and_ann, without_cross_no_ann]
        ])
        
        # Expected count
        expected = (total_with_cross * len(nodes_with_ann)) / len(node_table)
        fold_enrichment = (with_cross_and_ann / expected) if expected > 0 else 0
        
        # Fisher's exact test
        try:
            odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
        except:
            odds_ratio, p_value = np.nan, 1.0
        
        results.append({
            'annotation_term': annotation,
            'nodes_with_cross_edges': with_cross_and_ann,
            'nodes_without_cross_edges': without_cross_and_ann,
            'total_nodes_with_annotation': len(nodes_with_ann),
            'expected_count': expected,
            'fold_enrichment': fold_enrichment,
            'odds_ratio': odds_ratio,
            'p_value': p_value
        })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction
    valid_pvals = ~results_df['p_value'].isna()
    if valid_pvals.sum() > 0:
        pval_results = multipletests(
            results_df.loc[valid_pvals, 'p_value'],
            method=correction_method,
            alpha=p_value_threshold
        )
        results_df['p_value_corr'] = np.nan
        results_df.loc[valid_pvals, 'p_value_corr'] = pval_results[1]
        results_df['significant'] = False
        results_df.loc[valid_pvals, 'significant'] = pval_results[0]
    else:
        results_df['p_value_corr'] = np.nan
        results_df['significant'] = False
    
    results_df = results_df.sort_values('p_value_corr')
    
    return results_df

def calculate_submodule_preservation(
    independent_network: nx.Graph,
    full_network: nx.Graph,
    dataset_prefix: str = "tx_",
    feature_prefixes: List[str] = ["tx_", "mx_"]
) -> pd.DataFrame:
    """
    Calculate Jaccard similarity between submodules in independent vs full network
    to measure preservation of modular structure.
    
    Returns DataFrame with preservation scores for each independent submodule.
    """
    
    log.info(f"Calculating submodule preservation for dataset: {dataset_prefix}")
    
    # Get nodes for this dataset only
    dataset_nodes = {n for n in full_network.nodes() if n.startswith(dataset_prefix)}
    log.info(f"Found {len(dataset_nodes)} nodes with prefix '{dataset_prefix}' in full network")
    
    # Get submodule assignments from independent network
    independent_modules = {}
    for node in independent_network.nodes():
        if node in dataset_nodes:
            submod = independent_network.nodes[node].get('submodule', 'unknown')
            independent_modules.setdefault(submod, set()).add(node)
    
    log.info(f"Independent network contains {len(independent_modules)} submodules")
    
    # Get submodule assignments from full network
    full_modules = {}
    for node in full_network.nodes():
        if node in dataset_nodes:
            submod = full_network.nodes[node].get('submodule', 'unknown')
            full_modules.setdefault(submod, set()).add(node)
    
    log.info(f"Full network contains {len(full_modules)} submodules that include {dataset_prefix} dataset nodes")
    
    # Track which full network submodules were matched
    matched_full_submodules = set()
    
    # Calculate Jaccard similarity for each independent module vs all full modules
    results = []
    log.info(f"Computing Jaccard similarities for module pairs...")
    
    for indep_name, indep_nodes in independent_modules.items():
        best_match = None
        best_jaccard = 0
        
        for full_name, full_nodes in full_modules.items():
            # Jaccard = intersection / union
            intersection = len(indep_nodes & full_nodes)
            union = len(indep_nodes | full_nodes)
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = full_name
        
        # Track that this full submodule was matched
        if best_match:
            matched_full_submodules.add(best_match)
        
        # Get counts for the matched full network submodule
        matched_full_nodes = full_modules.get(best_match, set())
        nodes_in_matched_full = len(matched_full_nodes)
        nodes_preserved = len(indep_nodes & matched_full_nodes)
        preservation_pct = (nodes_preserved / len(indep_nodes) * 100) if len(indep_nodes) > 0 else 0
        
        # Track fate of non-preserved nodes
        non_preserved_nodes = indep_nodes - matched_full_nodes
        
        # Find where non-preserved nodes ended up
        non_preserved_destinations = {}
        nodes_not_in_any_submodule = set()
        
        for node in non_preserved_nodes:
            # Check if node exists in full network
            if node not in full_network.nodes():
                nodes_not_in_any_submodule.add(node)
                continue
            
            # Get the submodule assignment in full network
            node_submodule = full_network.nodes[node].get('submodule', None)
            
            if node_submodule and node_submodule != best_match:
                # Node went to a different submodule
                non_preserved_destinations[node_submodule] = non_preserved_destinations.get(node_submodule, 0) + 1
            elif not node_submodule:
                # Node has no submodule assignment
                nodes_not_in_any_submodule.add(node)
        
        # Format destination information for logging and storage
        destination_info = []
        if non_preserved_destinations:
            for dest_submod, count in sorted(non_preserved_destinations.items(), key=lambda x: x[1], reverse=True):
                destination_info.append(f"{dest_submod} ({count})")
                
        # Break down composition of full network submodule by data type
        all_nodes_in_full_submodule = {
            n for n in full_network.nodes() 
            if full_network.nodes[n].get('submodule') == best_match
        }
        
        # Count nodes by data type
        composition = {}
        for prefix in feature_prefixes:
            prefix_count = sum(1 for n in all_nodes_in_full_submodule if n.startswith(prefix))
            composition[f'{prefix.rstrip("_")}_nodes_in_full'] = prefix_count
        
        # Enhanced logging with non-preserved node destinations
        composition_str = ", ".join([f"{k.replace('_nodes_in_full', '')}: {v}"
                                    for k, v in composition.items()])
        
        log.info(f"{indep_name} ({len(indep_nodes)} nodes) → {best_match} "
                f"({nodes_in_matched_full} {dataset_prefix.rstrip('_')} nodes; "
                f"full composition: {composition_str}): "
                f"Jaccard={best_jaccard:.3f}, {preservation_pct:.1f}% nodes preserved")
        if destination_info:
            log.info(f"    {len(non_preserved_nodes)} non-preserved nodes went to: {';;'.join(destination_info)}")
        
        # Build result dictionary
        result = {
            'independent_submodule': indep_name,
            'matched_full_submodule': best_match,
            'jaccard_similarity': best_jaccard,
            'nodes_in_independent': len(indep_nodes),
            'nodes_in_matched_full': nodes_in_matched_full,
            'nodes_preserved': nodes_preserved,
            'preservation_rate': preservation_pct / 100,
            'nodes_not_preserved': len(non_preserved_nodes),
            'non_preserved_destinations': destination_info
        }
        
        # Add composition columns
        result.update(composition)
        result['total_nodes_in_full_submodule'] = len(all_nodes_in_full_submodule)
        results.append(result)
    
    results_df = pd.DataFrame(results).sort_values('jaccard_similarity', ascending=False)
    
    # Summary statistics
    avg_jaccard = results_df['jaccard_similarity'].mean()
    avg_preservation = results_df['preservation_rate'].mean()
    
    log.info(f"Summary statistics:")
    log.info(f"Average Jaccard similarity: {avg_jaccard:.3f}")
    log.info(f"Average node preservation rate: {avg_preservation:.1%}")
    log.info(f"Modules with Jaccard > 0.7 (strong preservation): {(results_df['jaccard_similarity'] > 0.7).sum()} ({(results_df['jaccard_similarity'] > 0.7).sum() / len(results_df) * 100:.1f}%)")
    log.info(f"Modules with Jaccard > 0.3 (moderate preservation): {(results_df['jaccard_similarity'] > 0.3).sum()} ({(results_df['jaccard_similarity'] > 0.3).sum() / len(results_df) * 100:.1f}%)")
    
    return results_df

def permutation_test_preservation(
    independent_network: nx.Graph,
    full_network: nx.Graph,
    dataset_prefix: str = "tx_",
    n_permutations: int = 1000
) -> dict:
    """
    Test if submodule preservation is statistically significant via permutation test.
    
    Null hypothesis: Module assignments in full network are random with respect 
    to independent network structure.
    """
    
    log.info(f"Calculating observed ARI for {dataset_prefix} preservation...")
    
    # Get common nodes
    common_nodes = sorted([
        n for n in independent_network.nodes() 
        if n in full_network.nodes() and n.startswith(dataset_prefix)
    ])
    
    if not common_nodes:
        log.warning(f"No common nodes found for {dataset_prefix}")
        return {
            'observed_ari': np.nan,
            'mean_null_ari': np.nan,
            'std_null_ari': np.nan,
            'p_value': np.nan,
            'z_score': np.nan
        }
    
    # Get cluster labels for common nodes
    independent_labels = [
        independent_network.nodes[n].get('submodule', 'unknown') 
        for n in common_nodes
    ]
    full_labels = [
        full_network.nodes[n].get('submodule', 'unknown') 
        for n in common_nodes
    ]
    
    # Calculate observed ARI
    observed_ari = adjusted_rand_score(independent_labels, full_labels)
    
    # Permutation test
    log.info(f"Running permutation test with {n_permutations} iterations...")
    null_aris = []
    
    for i in range(n_permutations):
        # Shuffle full network labels
        shuffled_full_labels = np.random.permutation(full_labels)
        null_ari = adjusted_rand_score(independent_labels, shuffled_full_labels)
        null_aris.append(null_ari)
    
    # Calculate p-value (proportion of null ARIs >= observed)
    p_value = np.sum(np.array(null_aris) >= observed_ari) / n_permutations
    
    # Calculate summary statistics
    mean_null = np.mean(null_aris)
    std_null = np.std(null_aris)
    z_score = (observed_ari - mean_null) / std_null if std_null > 0 else np.inf
    
    results = {
        'observed_ari': observed_ari,
        'mean_null_ari': mean_null,
        'std_null_ari': std_null,
        'p_value': p_value,
        'z_score': z_score
    }
    
    log.info(f"Permutation test results:")
    log.info(f"Observed ARI: {observed_ari:.3f}")
    log.info(f"Mean null ARI: {mean_null:.3f} ± {std_null:.3f}")
    log.info(f"P-value: {p_value:.4f}")
    log.info(f"Z-score: {z_score:.2f}")
    
    return results

def analyze_cross_omics_connectivity(
    full_network: nx.Graph,
    feature_prefixes: List[str] = ["tx_", "mx_"]
) -> dict:
    """
    Measure connectivity between omics layers in the integrated network.
    
    Args:
        full_network: Full integrated network
        feature_prefixes: List of feature prefixes for different omics types
        
    Returns:
        Dictionary with cross-omics connectivity metrics
    """
    
    log.info("Analyzing cross-omics connectivity...")
    
    # Partition nodes by omics type
    omics_nodes = {}
    for prefix in feature_prefixes:
        prefix_name = prefix.rstrip('_')
        omics_nodes[prefix_name] = {n for n in full_network.nodes() if n.startswith(prefix)}
        log.info(f"Found {len(omics_nodes[prefix_name])} {prefix_name} nodes")
    
    # Count edge types
    edge_counts = {}
    within_edges = {}
    
    # Count within-omics edges
    for prefix_name, nodes in omics_nodes.items():
        count = sum(1 for u, v in full_network.edges() if u in nodes and v in nodes)
        within_edges[prefix_name] = count
        edge_counts[f'within_{prefix_name}'] = count
    
    # Count between-omics edges
    between_counts = {}
    for i, prefix1 in enumerate(feature_prefixes):
        name1 = prefix1.rstrip('_')
        for prefix2 in feature_prefixes[i+1:]:
            name2 = prefix2.rstrip('_')
            count = sum(1 for u, v in full_network.edges() 
                       if (u in omics_nodes[name1] and v in omics_nodes[name2]) or 
                          (u in omics_nodes[name2] and v in omics_nodes[name1]))
            between_counts[f'{name1}_to_{name2}'] = count
            edge_counts[f'between_{name1}_{name2}'] = count
    
    # Calculate connectivity densities
    max_edges = {}
    densities = {}
    
    for prefix_name, nodes in omics_nodes.items():
        n = len(nodes)
        max_edges[f'within_{prefix_name}'] = n * (n - 1) / 2
        if max_edges[f'within_{prefix_name}'] > 0:
            densities[f'within_{prefix_name}'] = within_edges[prefix_name] / max_edges[f'within_{prefix_name}']
    
    # Between-omics max edges and density
    total_between_edges = sum(between_counts.values())
    for i, prefix1 in enumerate(feature_prefixes):
        name1 = prefix1.rstrip('_')
        for prefix2 in feature_prefixes[i+1:]:
            name2 = prefix2.rstrip('_')
            max_between = len(omics_nodes[name1]) * len(omics_nodes[name2])
            max_edges[f'between_{name1}_{name2}'] = max_between
            if max_between > 0:
                densities[f'between_{name1}_{name2}'] = between_counts[f'{name1}_to_{name2}'] / max_between
    
    # Calculate integration ratio (cross-omics edges / intra-omics edges)
    total_within_edges = sum(within_edges.values())
    integration_ratio = total_between_edges / total_within_edges if total_within_edges > 0 else 0
    
    # Count nodes with cross-omics neighbors
    nodes_with_cross_neighbors = {}
    for prefix_name, nodes in omics_nodes.items():
        count = 0
        other_nodes = set()
        for other_prefix in feature_prefixes:
            if other_prefix.rstrip('_') != prefix_name:
                other_nodes.update(omics_nodes[other_prefix.rstrip('_')])
        
        for node in nodes:
            neighbors = set(full_network.neighbors(node))
            if neighbors & other_nodes:
                count += 1
        
        nodes_with_cross_neighbors[prefix_name] = count
    
    results = {
        'edge_counts': edge_counts,
        'max_possible_edges': max_edges,
        'densities': densities,
        'integration_ratio': integration_ratio,
        'total_within_edges': total_within_edges,
        'total_between_edges': total_between_edges,
        'nodes_with_cross_omics_neighbors': nodes_with_cross_neighbors
    }
    
    # Log summary
    log.info("Cross-omics connectivity summary:")
    log.info(f"Total cross-omics edges: {total_between_edges:,}")
    log.info(f"Total within-omics edges: {total_within_edges:,}")
    log.info(f"Integration ratio: {integration_ratio:.3f}")
    
    for prefix_name, count in nodes_with_cross_neighbors.items():
        total = len(omics_nodes[prefix_name])
        pct = (count / total * 100) if total > 0 else 0
        log.info(f"{prefix_name} nodes with cross-omics neighbors: {count:,}/{total:,} ({pct:.1f}%)")
    
    for key, density in densities.items():
        log.info(f"{key} density: {density:.6f}")
    
    return results

def calculate_conditional_entropy_reduction(
    full_network: nx.Graph,
    independent_networks: Dict[str, nx.Graph],
    feature_prefixes: List[str]
) -> Dict[str, float]:
    """
    Calculate how much uncertainty about one layer is reduced by knowing the other.
    H(X|Y) = H(X,Y) - H(Y)
    Reduction = H(X) - H(X|Y)
    """
    
    log.info("Calculating conditional entropy reduction...")
    
    results = {}
    
    for prefix in feature_prefixes:
        dataset_name = prefix.rstrip('_')
        indep_name = f"{dataset_name}_only"
        
        if indep_name not in independent_networks:
            continue
        
        # Get nodes for this dataset
        dataset_nodes = {n for n in full_network.nodes() if n.startswith(prefix)}
        
        # Get nodes from OTHER datasets
        other_nodes = {n for n in full_network.nodes() if not n.startswith(prefix)}
        
        # Calculate H(X) - entropy of this layer alone
        indep_degrees = [independent_networks[indep_name].degree(n) 
                        for n in dataset_nodes 
                        if n in independent_networks[indep_name]]
        h_x = _degree_entropy(indep_degrees)
        
        # Calculate H(X|Y) - conditional entropy given other layers
        # Approximate by looking at degree distribution conditioned on cross-layer edges
        conditional_entropies = []
        for node in dataset_nodes:
            if node not in full_network:
                continue
                
            # Count edges to other omics types
            cross_edges = sum(1 for neighbor in full_network.neighbors(node) 
                            if neighbor in other_nodes)
            
            conditional_entropies.append((node, cross_edges))
        
        # Group by number of cross-edges and calculate entropy within each group
        groups = defaultdict(list)
        for node, cross_count in conditional_entropies:
            groups[cross_count].append(full_network.degree(node))
        
        # Weighted average of entropies
        h_x_given_y = 0
        total_nodes = len(conditional_entropies)
        for cross_count, degrees in groups.items():
            if degrees:
                prob_group = len(degrees) / total_nodes
                group_entropy = _degree_entropy(degrees)
                h_x_given_y += prob_group * group_entropy
        
        # Information gain
        reduction = h_x - h_x_given_y
        
        results[dataset_name] = {
            'h_independent': h_x,
            'h_conditional': h_x_given_y,
            'entropy_reduction': reduction,
            'reduction_percent': (reduction / h_x * 100) if h_x > 0 else 0
        }
        
        log.info(f"{dataset_name} conditional entropy reduction:")
        log.info(f"Independent entropy H(X): {h_x:.3f}")
        log.info(f"Conditional entropy H(X|Y): {h_x_given_y:.3f}")
        log.info(f"Reduction: {reduction:.3f} ({results[dataset_name]['reduction_percent']:.1f}%)")
    
    return results

def _degree_entropy(degrees):
    """Helper function for calculating Shannon entropy of degree distribution."""
    if not degrees:
        return 0
    degree_counts = pd.Series(degrees).value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in degree_counts if p > 0)

def analyze_cross_omics_motifs(
    full_network: nx.Graph,
    feature_prefixes: List[str],
    max_motif_size: int = 3
) -> Dict[str, int]:
    """
    Count specific cross-omics network motifs (small subgraph patterns).
    Example motifs:
    - One transcript connected to multiple metabolites (1-to-many)
    - Multiple transcripts connected to one metabolite (many-to-1)
    - Fully connected cross-omics triangles
    """
    
    log.info("Analyzing cross-omics network motifs...")

    motif_counts = {
        'transcript_hub': 0,  # 1 transcript → many metabolites
        'metabolite_hub': 0,  # 1 metabolite → many transcripts
    }
    
    # Identify nodes by type
    omics_nodes = {}
    for prefix in feature_prefixes:
        name = prefix.rstrip('_')
        omics_nodes[name] = {n for n in full_network.nodes() if n.startswith(prefix)}
    
    if len(omics_nodes) != 2:  # Currently designed for 2 omics types
        log.warning(f"Motif analysis requires exactly 2 omics types, found {len(omics_nodes)}. Returning empty counts.")
        return motif_counts
    
    type1, type2 = list(omics_nodes.keys())
    nodes1, nodes2 = omics_nodes[type1], omics_nodes[type2]
    
    log.info(f"Searching for motifs between {len(nodes1)} {type1} nodes and {len(nodes2)} {type2} nodes")
    log.info(f"Hub motif threshold: ≥3 cross-omics connections")
    
    # Count hub motifs (nodes with ≥3 connections to the other omics type)
    log.info(f"Counting {type1} hub motifs (1 {type1} node → many {type2} nodes)...")
    for node in nodes1:
        neighbors_type2 = [n for n in full_network.neighbors(node) if n in nodes2]
        if len(neighbors_type2) >= 3:
            motif_counts['transcript_hub'] += 1
    
    log.info(f"Counting {type2} hub motifs (1 {type2} node → many {type1} nodes)...")
    for node in nodes2:
        neighbors_type1 = [n for n in full_network.neighbors(node) if n in nodes1]
        if len(neighbors_type1) >= 3:
            motif_counts['metabolite_hub'] += 1
    
    log.info(f"Cross-omics motif analysis complete:")
    log.info(f"{type1} hubs: {motif_counts['transcript_hub']} nodes with ≥3 {type2} connections")
    log.info(f"{type2} hubs: {motif_counts['metabolite_hub']} nodes with ≥3 {type1} connections")
    
    # Interpretation
    total_hubs = motif_counts['transcript_hub'] + motif_counts['metabolite_hub']
    if total_hubs > 0:
        log.info(f"Interpretation: Found {total_hubs} hub nodes that bridge omics layers")
        log.info(f"These represent key regulatory or metabolic nodes in the integrated network")
    
    return motif_counts

def plot_independent_networks(
    networks: Dict[str, nx.Graph],
    feature_prefixes: List[str],
    layout: str = "spring",
    seed: int = 42,
    show_plot: bool = True
) -> None:
    """
    Create static matplotlib visualizations for each independent network.
    
    Uses the same visual style as plot_network_comparison:
    - All nodes in gray, shaped by data type (circle/square/triangle)
    - Black edges with consistent styling
    
    Args:
        networks: Dictionary containing independent networks (e.g., 'tx_only', 'mx_only')
        feature_prefixes: List of feature prefixes for different data types
        layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'fr')
        seed: Random seed for layout reproducibility
        show_plot: Whether to display plots inline
        
    Returns:
        None (displays plots)
    """
    
    log.info("Creating static network visualizations for independent networks...")
    
    # Filter to only independent networks (exclude 'full')
    independent_networks = {
        name: G for name, G in networks.items() 
        if name != 'full'
    }
    
    if not independent_networks:
        log.warning("No independent networks found to plot")
        return
    
    # Plot each independent network
    for network_name, network in independent_networks.items():
        log.info(f"Plotting {network_name} network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
        
        # Compute layout
        log.info(f"  Computing {layout} layout...")
        if layout == "spring":
            pos = nx.spring_layout(network, seed=seed, k=1/np.sqrt(len(network.nodes())))
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(network)
        elif layout == "circular":
            pos = nx.circular_layout(network)
        elif layout == "fr":
            pos = nx.fruchterman_reingold_layout(network, seed=seed)
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Draw nodes by data type with matching styles
        for i, prefix in enumerate(feature_prefixes):
            nodes_of_type = [n for n in network.nodes() if n.startswith(prefix)]
            
            if not nodes_of_type:
                continue
            
            # Use circle for first type, triangle for others (matching comparison plot)
            node_shape = 'o' if i == 0 else '^'
            
            nx.draw_networkx_nodes(
                network, pos,
                nodelist=nodes_of_type,
                node_color='#808080',  # Gray
                node_size=15,
                node_shape=node_shape,
                alpha=0.4,
                linewidths=0.5,
                edgecolors='white',
                ax=ax,
                label=f'{prefix.rstrip("_")} nodes'
            )
        
        # Draw all edges in black (matching the "preserved" style from comparison plot)
        nx.draw_networkx_edges(
            network, pos,
            edgelist=network.edges(),
            edge_color='#000000',  # Black
            width=0.25,
            alpha=0.25,
            ax=ax
        )
        
        # Create legend
        legend_elements = []
        
        # Add node type legends
        for i, prefix in enumerate(feature_prefixes):
            if any(n.startswith(prefix) for n in network.nodes()):
                marker = 'o' if i == 0 else '^'
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color='w', 
                          label=f'{prefix.rstrip("_")} nodes', 
                          markerfacecolor='#808080', markersize=8, alpha=0.4)
                )
        
        # Add edge legend
        legend_elements.append(
            Line2D([0], [0], color='#000000', linewidth=2, 
                  label=f'Edges ({network.number_of_edges()})')
        )
        
        # Add legend to plot
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Format network name for title
        display_name = network_name.replace('_only', '').replace('_', ' ').title()
        
        # Title
        ax.set_title(
            f"{display_name} Independent Network",
            fontsize=14,
            pad=20
        )
        ax.axis('off')
        
        if show_plot:
            plt.show()
        plt.close(fig)
    
    log.info("Independent network plotting complete")
    
    return

def plot_network_comparison(
    networks: Dict[str, nx.Graph],
    feature_prefixes: List[str],
    layout: str = "spring",
    seed: int = 42,
    show_plot: bool = True
) -> None:
    """
    Create a static matplotlib visualization comparing independent and integrated networks.
    
    Shows the FULL network with:
    - All nodes in gray, shaped by data type (circle/square)
    - Green edges: Only in full network (gained through integration)
    - Black edges: In any independent network (preserved across integration)
    
    Args:
        networks: Dictionary containing 'full' and independent networks
        feature_prefixes: List of feature prefixes for different data types
        output_dir: Directory to save visualization
        output_filename: Output filename (without extension)
        layout: Layout algorithm ('spring', 'kamada_kawai', 'circular')
        seed: Random seed for layout reproducibility
        show_plot: Whether to display plot inline
        
    Returns:
        None (saves plot to disk)
    """
    
    if 'full' not in networks:
        raise ValueError("Networks dictionary must contain 'full' network")
    
    log.info("Creating static network comparison visualization...")
    
    full_network = networks['full']
    
    # Collect all edges from independent networks
    independent_edges = set()
    for name, network in networks.items():
        if name != 'full':
            independent_edges.update(network.edges())
    
    log.info(f"Full network: {full_network.number_of_nodes()} nodes, {full_network.number_of_edges()} edges")
    log.info(f"Independent networks: {len(independent_edges)} total edges")
    
    # Compute layout on full network
    log.info(f"Computing {layout} layout...")
    if layout == "spring":
        pos = nx.spring_layout(full_network, seed=seed, k=1/np.sqrt(len(full_network.nodes())))
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(full_network)
    elif layout == "circular":
        pos = nx.circular_layout(full_network)
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(full_network, seed=seed)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    # Helper function to normalize edges for undirected comparison
    def normalize_edge(edge):
        return tuple(sorted(edge))
    
    # Normalize independent edges for comparison
    normalized_independent = {normalize_edge(e) for e in independent_edges}
    
    # Categorize edges in full network
    black_edges = []  # In independent networks
    green_edges = []  # Only in full network
    
    for edge in full_network.edges():
        if normalize_edge(edge) in normalized_independent:
            black_edges.append(edge)
        else:
            green_edges.append(edge)
    
    log.info(f"Black edges (in independent): {len(black_edges)}")
    log.info(f"Green edges (gained in full): {len(green_edges)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw nodes by data type
    for i, prefix in enumerate(feature_prefixes):
        nodes_of_type = [n for n in full_network.nodes() if n.startswith(prefix)]
        
        if not nodes_of_type:
            continue
        
        # Draw nodes of this type
        nx.draw_networkx_nodes(
            full_network, pos,
            nodelist=nodes_of_type,
            node_color='#808080',
            node_size=15,
            node_shape='o' if i == 0 else '^',
            alpha=0.4,
            linewidths=0.5,
            edgecolors='white',
            ax=ax,
            label=f'{prefix.rstrip("_")} nodes'
        )

    # Draw edges by preserved vs new
    if black_edges:
        nx.draw_networkx_edges(
            full_network, pos,
            edgelist=black_edges,
            edge_color='#000000',
            width=0.25,
            alpha=0.25,
            ax=ax
        )
    if green_edges:
        nx.draw_networkx_edges(
            full_network, pos,
            edgelist=green_edges,
            edge_color='#008000',
            width=0.5,
            alpha=0.5,
            ax=ax
        )
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], color='#000000', linewidth=2, label=f'Edges shared with independent networks ({len(black_edges)})'),
        Line2D([0], [0], color='#008000', linewidth=2, label=f'Edges unique to integrated network ({len(green_edges)})'),
        Line2D([0], [0], marker='o', color='w', label=f'{feature_prefixes[0].rstrip("_")} nodes', markerfacecolor='#808080', markersize=8, alpha=0.4),
        Line2D([0], [0], marker='^', color='w', label=f'{feature_prefixes[1].rstrip("_")} nodes', markerfacecolor='#808080', markersize=8, alpha=0.4)
    ]
        
    # Combine legends
    leg = ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Title
    ax.set_title(
        f"Full Integrated Network",
        fontsize=14,
        pad=20
    )
    ax.axis('off')
    plt.show()
    plt.close(fig)
    
    return

def plot_interactive_network_comparison(
    networks: Dict[str, nx.Graph],
    feature_prefixes: List[str],
    layout: str = "spring",
    seed: int = 42,
) -> go.FigureWidget:
    """
    Create an interactive Plotly visualization comparing independent and integrated networks.
    
    Shows the FULL network with:
    - Nodes colored by submodule assignment
    - Node shapes differ by data type (circle/triangle)
    - Gray edges: Within same data type (intra-omics)
    - Black edges: Between different data types (cross-omics)
    - Preserved vs gained edges shown by line style
    
    Args:
        networks: Dictionary containing 'full' and independent networks
        feature_prefixes: List of feature prefixes for different data types
        layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'fr')
        seed: Random seed for layout reproducibility
        
    Returns:
        go.FigureWidget: Interactive Plotly widget
    """
    
    if 'full' not in networks:
        raise ValueError("Networks dictionary must contain 'full' network")
    
    log.info("Creating interactive network comparison visualization...")
    
    full_network = networks['full']
    
    # Collect all edges from independent networks
    independent_edges = set()
    for name, network in networks.items():
        if name != 'full':
            independent_edges.update(network.edges())
    
    log.info(f"Full network: {full_network.number_of_nodes()} nodes, {full_network.number_of_edges()} edges")
    log.info(f"Independent networks: {len(independent_edges)} total edges")
    
    # Compute layout on full network
    log.info(f"Computing {layout} layout...")
    if layout == "spring":
        pos = nx.spring_layout(full_network, seed=seed, k=10/np.sqrt(len(full_network.nodes())), weight="weight", iterations=100)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(full_network)
    elif layout == "circular":
        pos = nx.circular_layout(full_network)
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(full_network, seed=seed)
    else:
        raise ValueError(f"Unsupported layout: {layout}")
    
    # Helper function to normalize edges for undirected comparison
    def normalize_edge(edge):
        return tuple(sorted(edge))
    
    # Normalize independent edges for comparison
    normalized_independent = {normalize_edge(e) for e in independent_edges}
    
    # Helper function to determine edge type
    def get_edge_type(u, v):
        """Determine if edge is within or between data types"""
        u_type = next((p for p in feature_prefixes if u.startswith(p)), None)
        v_type = next((p for p in feature_prefixes if v.startswith(p)), None)
        return "cross" if u_type != v_type else "intra"
    
    # Categorize edges in full network
    gray_preserved_edges = []  # Intra-omics edges in independent networks
    gray_gained_edges = []     # Intra-omics edges only in full network
    black_preserved_edges = [] # Cross-omics edges in independent networks
    black_gained_edges = []    # Cross-omics edges only in full network
    
    for edge in full_network.edges():
        edge_type = get_edge_type(edge[0], edge[1])
        is_preserved = normalize_edge(edge) in normalized_independent
        
        if edge_type == "intra":
            if is_preserved:
                gray_preserved_edges.append(edge)
            else:
                gray_gained_edges.append(edge)
        else:  # cross-omics
            if is_preserved:
                black_preserved_edges.append(edge)
            else:
                black_gained_edges.append(edge)
    
    log.info(f"Gray edges (intra-omics preserved): {len(gray_preserved_edges)}")
    log.info(f"Gray edges (intra-omics gained): {len(gray_gained_edges)}")
    log.info(f"Black edges (cross-omics preserved): {len(black_preserved_edges)}")
    log.info(f"Black edges (cross-omics gained): {len(black_gained_edges)}")
    
    # Build edge traces
    def create_edge_trace(edges, color, width, dash, name):
        edge_x, edge_y = [], []
        for u, v in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        
        return go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=width, color=color, dash=dash),
            hoverinfo="none",
            showlegend=True,
            name=name
        )
    
    # Create edge traces (preserved = solid, gained = dashed)
    edge_traces = [
        create_edge_trace(gray_preserved_edges, "#888888", 1, "solid", 
                         f"Intra-omics preserved ({len(gray_preserved_edges)})"),
        create_edge_trace(gray_gained_edges, "#888888", 1, "dash", 
                         f"Intra-omics gained ({len(gray_gained_edges)})"),
        create_edge_trace(black_preserved_edges, "#000000", 2, "solid", 
                         f"Cross-omics preserved ({len(black_preserved_edges)})"),
        create_edge_trace(black_gained_edges, "#000000", 2, "dash", 
                         f"Cross-omics gained ({len(black_gained_edges)})"),
    ]
    
    # Get unique submodules and create color palette
    log.info("Building node traces...")
    submodules = set()
    for n in full_network.nodes():
        submod = full_network.nodes[n].get('submodule', 'unassigned')
        if submod:
            submodules.add(submod)
    
    n_submodules = len(submodules)
    submodule_colors = viridis(np.linspace(0, 1, n_submodules))
    submodule_color_map = {
        submod: to_hex(submodule_colors[i]) 
        for i, submod in enumerate(sorted(submodules))
    }
    
    # Create separate node traces for each data type (to get different shapes)
    node_traces = []
    
    # Map Plotly symbols to match matplotlib: circle for first type, triangle-up for others
    plotly_symbols = ['circle', 'triangle-up', 'square', 'diamond', 'cross']
    
    for i, prefix in enumerate(feature_prefixes):
        # Get nodes of this type
        nodes_of_type = [n for n in full_network.nodes() if n.startswith(prefix)]
        
        if not nodes_of_type:
            continue
        
        node_x, node_y, node_color, node_size, hover_txt = [], [], [], [], []
        
        for n in nodes_of_type:
            data = full_network.nodes[n]
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            
            # Color by submodule
            submodule = data.get("submodule", "unassigned")
            node_color.append(submodule_color_map.get(submodule, "#808080"))
            node_size.append(data.get("node_size", 10))
            
            # Build hover text
            hover_parts = [str(n)]
            if submodule:
                hover_parts.append(f"{submodule.replace('_', ' ')}")
            
            # Add annotations
            display_name = data.get('display_name', 'Unassigned')
            if display_name != 'Unassigned':
                hover_parts.append(display_name)
            
            hover_txt.append("<br>".join(hover_parts))
        
        # Create trace for this data type with specific symbol
        symbol = plotly_symbols[i % len(plotly_symbols)]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                symbol=symbol,
                opacity=0.9,
                line=dict(width=1, color="#222"),
            ),
            hoverinfo="text",
            text=hover_txt,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
            name=f"{prefix.rstrip('_')} nodes",
            customdata=nodes_of_type,
        )
        
        node_traces.append(node_trace)
    
    # Assemble figure widget
    log.info("Assembling interactive network widget...")
    fig = go.FigureWidget(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title="Integrated Network Comparison (interactive)",
            title_x=0.5,
            showlegend=True,
            hovermode="closest",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=900,
            width=1200,
            clickmode="event+select",
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        ),
    )
    
    log.info("Interactive network comparison complete")
    return fig

# ====================================
# Dataset acquisition functions
# ====================================     

def find_mx_parent_folder(
    pid: str,
    pi_name: str,
    mx_dir: str,
    polarity: str,
    datatype: str,
    chromatography: str,
    filtered_mx: bool = False,
    overwrite: bool = False,
    superuser: bool = False
) -> str:
    """
    Find the parent folder for metabolomics (MX) data on Google Drive using rclone.

    Args:
        pid (str): Proposal ID.
        pi_name (str): PI name.
        mx_dir (str): Local MX data directory.
        polarity (str): Polarity ('positive', 'negative', 'multipolarity').
        datatype (str): Data type ('peak-height', 'peak-area', etc.).
        chromatography (str): Chromatography type.
        filtered_mx (bool): Whether to use filtered data.
        overwrite (bool): Overwrite existing results.

    Returns:
        str: Path to the final results folder, or None if not found.
    """
    
    if datatype == "peak-area":
        datatype = "quant"
    if filtered_mx and datatype == "quant":
        log.info("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None
    if polarity == "multipolarity":
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        log.info(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
        return None
    if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
        log.info("MX folder already downloaded and linked.")
        return None
    elif glob.glob(os.path.expanduser(mx_data_pattern)) and overwrite:
        if not superuser:
            raise ValueError("You are not currently authorized to download or overwrite metabolomics data from source. Please contact your JGI project manager for access.")
        elif superuser:
            log.info("MX folder already downloaded and linked. Overwriting as per user request...")
    elif not glob.glob(os.path.expanduser(mx_data_pattern)):
        if not superuser:
            raise ValueError("MX folder not found locally. Exiting...")
        elif superuser:
            log.info("MX folder not found locally. Proceeding to find and link MX data from Google Drive...")

    # Find project folder
    cmd = f"rclone lsd JGI_Metabolomics_Projects: | grep -E '{pid}|{pi_name}'"
    log.info("Finding MX parent folders...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        data = [line.split()[:5] for line in result.stdout.strip().split('\n')]
        mx_parent = pd.DataFrame(data, columns=["dir", "date", "time", "size", "folder"])
        mx_parent = mx_parent[["date", "time", "folder"]]
        mx_parent["ix"] = range(1, len(mx_parent) + 1)
        mx_parent = mx_parent[["ix", "date", "time", "folder"]]
        mx_final_folders = []
        # For each possible project folder (some will not be the right "final" folder)
        log.info("Finding MX final folders...")
        for project_folder in mx_parent["folder"].values:
            cmd = f"rclone lsd --max-depth 2 JGI_Metabolomics_Projects:{project_folder}"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            except:
                continue
            if result.stdout or result.stderr:
                output = result.stdout if result.stdout else result.stderr
                data = [line.split()[:5] for line in output.strip().split('\n')]
                mx_final = pd.DataFrame(data, columns=["dir", "date", "time", "size", "folder"])
                mx_final = mx_final[["date", "time", "folder"]]
                mx_final["ix"] = range(1, len(mx_final) + 1)
                mx_final = mx_final[["ix", "date", "time", "folder"]]
                mx_final['parent_folder'] = project_folder
                mx_final_folders.append(mx_final)
            else:
                return None

        mx_final_combined = pd.concat(mx_final_folders, ignore_index=True)
        untargeted_mx_final = mx_final_combined[
            mx_final_combined["folder"].str.contains("Untargeted", case=False) & 
            mx_final_combined["folder"].str.contains("final", case=False) & 
            ~mx_final_combined["folder"].str.contains("pilot", case=False)
        ]
        if untargeted_mx_final.shape[0] > 1:
            log.info("Warning! Multiple untargeted MX final folders found:")
            log.info(untargeted_mx_final)
            return None
        elif untargeted_mx_final.shape[0] == 0:
            log.info("Warning! No untargeted MX final folders found.")
            return None
        else:
            final_results_folder = f"{untargeted_mx_final['parent_folder'].values[0]}/{untargeted_mx_final['folder'].values[0]}"

        script_dir = f"{mx_dir}/scripts"
        os.makedirs(script_dir, exist_ok=True)
        script_name = f"{script_dir}/find_mx_files.sh"
        with open(script_name, "w") as script_file:
            script_file.write(f"cd {script_dir}/\n")
            script_file.write(f"rclone lsd --max-depth 2 JGI_Metabolomics_Projects:{untargeted_mx_final['parent_folder'].values[0]}")
        
        log.info("Using the following metabolomics final results folder for further analysis:")
        log.info(untargeted_mx_final)
        return final_results_folder
    else:
        log.info(f"Warning! No folders could be found with rclone lsd command: {cmd}")
        return None

def gather_mx_files(
    mx_untargeted_remote: str,
    mx_dir: str,
    polarity: str,
    datatype: str,
    chromatography: str,
    filtered_mx: bool = False,
    extract: bool = False,
    overwrite: bool = False
) -> tuple:
    """
    Link MX files from Google Drive to the local directory using rclone, and optionally extract them.

    Args:
        mx_untargeted_remote (str): Remote MX folder path.
        mx_dir (str): Local MX data directory.
        polarity (str): Polarity.
        datatype (str): Data type.
        chromatography (str): Chromatography type.
        filtered_mx (bool): Use filtered data.
        extract (bool): Extract archives after linking.
        overwrite (bool): Overwrite existing results.

    Returns:
        tuple: DataFrames of archives and extractions, or None.
    """
    
    if datatype == "peak-area":
        datatype = "quant"
    if filtered_mx and datatype == "quant":
        log.info("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None
    if polarity == "multipolarity":
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        log.info(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
        return None
    if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
        log.info("MX data already linked.")
        return "Archive already linked", "Archive already extracted"
    else:
        if superuser:
            log.info("You are a superuser. Proceeding with the download...")
        else:
            raise ValueError("You are not currently authorized to download metabolomics data from source. Please contact your JGI project manager for access.")
    
    script_dir = f"{mx_dir}/scripts"
    os.makedirs(script_dir, exist_ok=True)
    script_name = f"{script_dir}/gather_mx_files.sh"
    if chromatography == "C18":
        chromatography = "C18_" # This (hopefully) removes C18-Lipid

    # Create the script to link MX files
    with open(script_name, "w") as script_file:
        script_file.write(f"cd {script_dir}/\n")
        script_file.write(f"rclone copy --include '*{chromatography}*.zip' --stats-one-line -v --max-depth 1 JGI_Metabolomics_Projects:{mx_untargeted_remote} {mx_dir};")
    
    log.info("Linking MX files...")
    result = subprocess.run(f"chmod +x {script_name} && {script_name}", shell=True, check=True, capture_output=True, text=True)

    if result.stdout or result.stderr:
        output = result.stdout if result.stdout else result.stderr
        data = [line.split() for line in output.strip().split('\n')]
        archives = pd.DataFrame(data)
        if extract is True:
            extractions = extract_mx_archives(mx_dir, chromatography)
            display(extractions)
            return archives, extractions
        else:
            return archives
    else:
        log.info(f"Warning! No files could be found with rclone copy command in {script_name}")
        return None

def extract_mx_archives(mx_dir: str, chromatography: str) -> pd.DataFrame:
    """
    Extract MX zip archives in the specified directory.

    Args:
        mx_dir (str): Directory containing MX archives.
        chromatography (str): Chromatography type.

    Returns:
        pd.DataFrame: DataFrame of extracted files.
    """
    
    cmd = f"for archive in {mx_dir}/*{chromatography}*zip; do newdir={mx_dir}/$(basename $archive .zip); rm -rf $newdir; mkdir -p $newdir; unzip -j $archive -d $newdir; done"
    log.info("Extracting the following archive to be used for MX data input:")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        data = [line.split() for line in result.stdout.strip().split('\n')]
        df = pd.DataFrame(data)
        df = df[df.iloc[:, 0].str.contains('Archive', na=False)]
        df.iloc[:, 1] = df.iloc[:, 1].str.replace(f"{mx_dir}/", "", regex=False)
        return df
    else:
        log.info(f"No archives could be decompressed with unzip command: {cmd}")
        return None


def find_tx_files(
    pid: str,
    tx_dir: str,
    tx_index: int,
    overwrite: bool = False,
    superuser: bool = False
) -> pd.DataFrame:
    """
    Find TX files for a project using JAMO report select.

    Args:
        pid (str): Proposal ID.
        tx_dir (str): TX data directory.
        tx_index (int): Index of analysis project to use.
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: DataFrame of TX file information.
    """
    
    if os.path.exists(f"{tx_dir}/all_tx_portal_files.txt") and overwrite is False:
        log.info("TX files already found.")
        return pd.DataFrame()
    else:
        if superuser:
            log.info("You are a superuser. Proceeding with finding TX files...")
        else:
            raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")
    
    file_list = f"{tx_dir}/all_tx_portal_files.txt"
    script_dir = f"{tx_dir}/scripts"
    os.makedirs(script_dir, exist_ok=True)
    script_name = f"{script_dir}/find_tx_files.sh"

    if not os.path.exists(os.path.dirname(file_list)):
        os.makedirs(os.path.dirname(file_list))
    if not os.path.exists(os.path.dirname(script_name)):
        os.makedirs(os.path.dirname(script_name))
    
    log.info("Creating script to find TX files...")
    script_content = (
        f"jamo report select _id,metadata.analysis_project.analysis_project_id,metadata.library_name,metadata.analysis_project.status_name where "
        f"metadata.proposal_id={pid} file_name=counts.txt "
        f"| sed 's/\\[//g' | sed 's/\\]//g' | sed 's/u'\\''//g' | sed 's/'\\''//g' | sed 's/ //g' > {file_list}"
    )

    log.info("Finding TX files...")
    subprocess.run(
        f"echo \"{script_content}\" > {script_name} && chmod +x {script_name} && module load jamo && source {script_name}",
        shell=True, check=True
    )
    
    files = pd.read_csv(file_list, header=None, sep="\t")
    files.columns = ["fileID.counts", "APID", "libIDs", "status"]
    files["nLibs"] = files["libIDs"].apply(lambda ll: len(ll.split(",")))
    
    def get_tpm_file_id(apid):
        result = subprocess.run(
            f"module load jamo; jamo report select _id where file_name=tpm_counts.txt,metadata.analysis_project.analysis_project_id={apid}",
            shell=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    
    files["fileID.tpm"] = files["APID"].apply(get_tpm_file_id)
    
    def fetch_refs(apid):
        try:
            cmd = (
                f"x=$(curl https://rqc.jgi.lbl.gov/api/seq_jat_import/apid_to_ref/{apid}); "
                f"echo $(echo $x | jq .name)','$(echo $x | jq .Genome[].file_path)','$(echo $x | jq .Transcriptome[].file_path)','$(echo $x | jq .Annotation[].file_path)','$(echo $x | jq .KEGG[].file_path)"
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            ref_data = pd.read_csv(io.StringIO(result.stdout), header=None, sep=",")
            ref_data.columns = ["ref_name", "ref_genome", "ref_transcriptome", "ref_gff", "ref_protein_kegg"]
            ref_data["APID"] = apid
            return ref_data
        except Exception as e:
            log.info(f"Error fetching refs for {apid}: {e}")
            return pd.DataFrame({"ref_name": [np.nan], "ref_genome": [np.nan], "ref_transcriptome": [np.nan], "ref_gff": [np.nan], "ref_protein_kegg": [np.nan], "APID": [apid]})
    
    refs = pd.concat([fetch_refs(apid) for apid in files["APID"].unique()], ignore_index=True)
    files = files.merge(refs, on="APID", how="left").sort_values(by=["ref_name", "APID"]).reset_index(drop=True)
    files["ix"] = files.index + 1
    # Move "ix" column to the beginning
    cols = ["ix"] + [col for col in files.columns if col != "ix"]
    files = files[cols]
    files.reset_index(drop=True)
    
    if files.shape[0] > 0:
        log.info(f"Using the value of 'tx_index' ({tx_index}) from the config file to choose the correct 'ix' column (change if incorrect): ")
        files.to_csv(f"{tx_dir}/all_tx_portal_files.txt", sep="\t", index=False)
        display(files)
        return files
    else:
        log.info("No files found.")
        return None

def gather_tx_files(
    file_list: pd.DataFrame,
    tx_index: int = None,
    tx_dir: str = None,
    overwrite: bool = False
) -> str:
    """
    Link TX files to the working directory using JAMO.

    Args:
        file_list (pd.DataFrame): DataFrame of TX file information.
        tx_index (int): Index of analysis project to use.
        tx_dir (str): TX data directory.
        overwrite (bool): Overwrite existing results.

    Returns:
        str: Analysis project ID (APID).
    """

    if glob.glob(f"{tx_dir}/*counts.txt") and os.path.exists(f"{tx_dir}/all_tx_portal_files.txt") and overwrite is False:
        log.info("TX files already linked.")
        tx_files = pd.read_csv(f"{tx_dir}/all_tx_portal_files.txt", sep="\t")
        apid = int(tx_files.iloc[tx_index-1:,2].values)
        return apid
    else:
        if superuser:
            log.info("You are a superuser. Proceeding with linking TX files...")
        else:
            raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")


    if tx_index is None:
        log.info("There may be multiple APIDS or analyses for a given PI/Proposal ID and you have not specified which one to use!")
        log.info("Please set 'tx_index' in the project config file by choosing the correct row from the table above.")
        sys.exit(1)

    script_dir = f"{tx_dir}/scripts"
    os.makedirs(script_dir, exist_ok=True)
    script_name = f"{script_dir}/gather_tx_files.sh"
    log.info("Linking TX files...")

    with open(script_name, "w") as script_file:
        script_file.write(f"cd {script_dir}/\n")
        
        file_list = file_list[file_list["ix"] == tx_index]
        apid = file_list["APID"].values[0]
        for _, row in file_list.iterrows():
            for file_type in ["counts","tpm"]: # "tpm_counts" may be needed for older projects
                file_id = row[f"fileID.{file_type}"]
                apid = row['APID']
                filename = f"{tx_dir}/{row['ix']}_{apid}.{file_type}.txt"
                if file_type == "tpm":
                    file_type = "tpm_counts"
                script_file.write(f"""
                    module load jamo;
                    if [ $(jamo info id {file_id} | cut -f3 -d' ') == 'PURGED' ]; then 
                        echo file purged! fetching {file_id} - rerun later to link; jamo fetch id {file_id} 2>&1 > /dev/null
                    else 
                        if [ $(jamo info id {file_id} | cut -f3 -d' ') == 'RESTORE_IN_PROGRESS' ]; then 
                            echo restore in progress for file id {file_id} - rerun later to link
                        else
                            echo "\t{file_type} file for APID {apid} with ID {file_id} linked to {filename}";
                            jamo link id {file_id} 2>&1 > /dev/null;
                            mv {file_id}.{file_type}.txt {filename}
                        fi
                    fi
                    """)
            for file_type in ["genome","transcriptome","gff","protein_kegg"]:
                file_path = row[f"ref_{file_type}"]
                try:
                    apid = row['APID']
                    filename = f"{tx_dir}/{os.path.basename(file_path)}"
                    if file_path:
                        script_file.write(f"""
                            echo "\t{file_type} file for APID {apid} to {file_path}";
                            ln -sf {file_path} {tx_dir}/
                        """)
                except:
                    log.info(f"\tError linking {file_type} file for APID {apid}. File does not exist or you may need to wait for files to be restored.")
                    continue

    subprocess.run(f"chmod +x {script_name} && {script_name}", shell=True, check=True)
    
    if apid:
        with open(f"{tx_dir}/apid.txt", "w") as f:
            f.write(str(apid))
        log.info(f"Working with APID: {apid} from tx_index {tx_index}.")
        return apid
    else:
        log.info("Warning: Did not find APID. Check the index you selected from the tx_files object.")
        return None

def get_mx_data(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    chromatography: str,
    polarity: str,
    datatype: str = "peak-height",
    filtered_mx: bool = False,
    superuser: bool = False,
) -> pd.DataFrame:
    """
    Load MX data from extracted files, optionally filtered.

    Args:
        input_dir (str): Input directory containing MX data.
        output_dir (str): MX data directory.
        output_filename (str): Output filename for MX data.
        chromatography (str): Chromatography type.
        polarity (str): Polarity.
        datatype (str): Data type.
        filtered_mx (bool): Use filtered data.
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: MX data matrix.
    """

    if datatype == "peak-area":
        datatype = "quant"
    if filtered_mx and datatype == "quant":
        log.info("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None
    if polarity == "multipolarity":
        mx_data_pattern = f"{input_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{input_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{input_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{input_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        log.info(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
        return None

    mx_data_files = glob.glob(os.path.expanduser(mx_data_pattern))
    if mx_data_files:
        if len(mx_data_files) > 1:
            multipolarity_datasets = []
            for mx_data_file in mx_data_files:
                file_polarity = (
                    "positive" if "positive" in mx_data_file 
                    else "negative" if "negative" in mx_data_file 
                    else "NO_POLARITY"
                )
                mx_dataset = pd.read_csv(mx_data_file)
                mx_data = mx_dataset.copy()
                if datatype == "peak-height":
                    mx_data.columns = mx_data.columns.str.replace(' Peak height', '')
                if datatype == "peak-area":
                    mx_data.columns = mx_data.columns.str.replace(' Peak area', '')
                mx_data.columns = mx_data.columns.str.replace('.mzML', '')
                mx_data = mx_data.rename(columns={mx_data.columns[0]: 'CompoundID'})
                mx_data = mx_data.drop(columns=['row m/z', 'row retention time'])
                mx_data = mx_data.drop(columns=[col for col in mx_data.columns if "Unnamed" in col])
                if pd.api.types.is_numeric_dtype(mx_data['CompoundID']):
                    mx_data['CompoundID'] = 'mx_' + mx_data['CompoundID'].astype(str) + "_" + str(file_polarity)
                multipolarity_datasets.append(mx_data)
            multipolarity_data = pd.concat(multipolarity_datasets, axis=0)
            log.info(f"MX data loaded from {mx_data_files}:")
            #display(multipolarity_data.head())
            write_integration_file(data=multipolarity_data, output_dir=output_dir, filename=output_filename, indexing=False)
            return multipolarity_data
        elif len(mx_data_files) == 1:
            mx_data_filename = mx_data_files[0]  # Assuming you want the first match
            mx_dataset = pd.read_csv(mx_data_filename)
            mx_data = mx_dataset.copy()
            if datatype == "peak-height":
                mx_data.columns = mx_data.columns.str.replace(' Peak height', '')
            if datatype == "peak-area":
                mx_data.columns = mx_data.columns.str.replace(' Peak area', '')
            mx_data.columns = mx_data.columns.str.replace('.mzML', '')
            mx_data = mx_data.rename(columns={mx_data.columns[0]: 'CompoundID'})
            mx_data = mx_data.drop(columns=['row m/z', 'row retention time'])
            mx_data = mx_data.drop(columns=[col for col in mx_data.columns if "Unnamed" in col])
            if pd.api.types.is_numeric_dtype(mx_data['CompoundID']):
                mx_data['CompoundID'] = 'mx_' + mx_data['CompoundID'].astype(str)
            log.info(f"MX data loaded from {mx_data_filename}:")
            write_integration_file(data=mx_data, output_dir=output_dir, filename=output_filename, indexing=False)
            #display(mx_data.head())
            return mx_data
        else:
            log.info("No MX data files found.")
            return None

    else:
        log.warning(f"MX data file matching pattern {mx_data_pattern} not found.")
        log.warning("Have you run _get_raw_metadata() to extract the MX data files?")
        return None

def get_mx_metadata(
    output_filename: str,
    output_dir: str,
    input_dir: str,
    chromatography: str,
    polarity: str
) -> pd.DataFrame:
    """
    Load MX metadata from extracted files, optionally filtered.

    Args:
        output_filename (str): Output filename for MX metadata.
        output_dir (str): MX data directory.
        input_dir (str): Input directory containing MX data.
        chromatography (str): Chromatography type.
        polarity (str): Polarity.

    Returns:
        pd.DataFrame: MX metadata DataFrame.
    """

    if polarity == "multipolarity":
        mx_metadata_pattern = f"{input_dir}/*{chromatography}*/*_metadata.tab"
        mx_metadata_files = glob.glob(os.path.expanduser(mx_metadata_pattern))
    elif polarity in ["positive", "negative"]:
        mx_metadata_pattern = f"{input_dir}/*{chromatography}*/*{polarity}_metadata.tab"
        mx_metadata_files = glob.glob(os.path.expanduser(mx_metadata_pattern))

    if mx_metadata_files:
        if len(mx_metadata_files) > 1:
            multiploarity_metadata = []
            for mx_metadata_file in mx_metadata_files:
                mx_metadataset = pd.read_csv(mx_metadata_file, sep='\t')
                mx_metadata = mx_metadataset.copy()
                mx_metadata.columns = mx_metadata.columns.str.replace('ATTRIBUTE_sampletype', 'full_sample_metadata')
                mx_metadata['filename'] = mx_metadata['filename'].str.replace('.mzML', '', regex=False)
                mx_metadata = mx_metadata.rename(columns={mx_metadata.columns[0]: 'file'})
                mx_metadata.insert(0, 'ix', mx_metadata.index + 1)
                multiploarity_metadata.append(mx_metadata)
            multiploarity_metadatum = pd.concat(multiploarity_metadata, axis=0)
            multiploarity_metadatum.drop_duplicates(inplace=True)
            log.info(f"MX metadata loaded from {mx_metadata_files}")
            write_integration_file(data=multiploarity_metadatum, output_dir=output_dir, filename=output_filename, indexing=False)
            #display(multiploarity_metadatum.head())
            return multiploarity_metadatum
        elif len(mx_metadata_files) == 1:
            mx_metadata_filename = mx_metadata_files[0]  # Assuming you want the first match
            mx_metadata = pd.read_csv(mx_metadata_filename, sep='\t')
            mx_metadata.columns = mx_metadata.columns.str.replace('ATTRIBUTE_sampletype', 'full_sample_metadata')
            mx_metadata['filename'] = mx_metadata['filename'].str.replace('.mzML', '', regex=False)
            mx_metadata = mx_metadata.rename(columns={mx_metadata.columns[0]: 'file'})
            mx_metadata.insert(0, 'ix', mx_metadata.index + 1)
            log.info(f"MX metadata loaded from {mx_metadata_filename}")
            log.info("Writing MX metadata to file...")
            write_integration_file(data=mx_metadata, output_dir=output_dir, filename=output_filename, indexing=False)
            #display(mx_metadata.head())
            return mx_metadata
    else:
        log.info(f"MX data file matching pattern {mx_metadata_pattern} not found.")
        return None

def get_tx_data(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    type: str = "counts",
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Load TX data from linked files.

    Args:
        input_dir (str): Directory containing TX data files.
        output_dir (str): TX data directory.
        type (str): Data type ('counts', etc.).
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: TX data matrix.
    """

    tx_data_pattern = f"{input_dir}/{type}.txt"
    tx_data_files = glob.glob(os.path.expanduser(tx_data_pattern))
    
    if tx_data_files:
        if len(tx_data_files) > 1:
            log.info(f"Multiple TX data files found matching pattern {tx_data_pattern}.")
            log.info("Please specify the correct file.")
            return None
        tx_data_filename = tx_data_files[0]  # Assuming you want the first (and only) match
        tx_data = pd.read_csv(tx_data_filename, sep='\t')
        tx_data = tx_data.rename(columns={tx_data.columns[0]: 'GeneID'})
        
        # Add prefix 'tx_' if not already present
        tx_data['GeneID'] = tx_data['GeneID'].apply(lambda x: x if str(x).startswith('tx_') else f'tx_{x}')
        
        log.info(f"TX data loaded from {tx_data_filename} and processing...")
        write_integration_file(data=tx_data, output_dir=output_dir, filename=output_filename, indexing=False)
        #display(tx_data.head())
        return tx_data
    else:
        log.warning(f"TX data file matching pattern {tx_data_pattern} not found.")
        log.warning("Have you run _get_raw_metadata() to link the TX data files?")
        return None
    
def get_tx_metadata(
    tx_files: pd.DataFrame,
    output_dir: str,
    proposal_ID: str,
    apid: str,
    overwrite: bool = False,
    superuser: bool = False
) -> pd.DataFrame:
    """
    Extract TX metadata using JAMO report select.

    Args:
        tx_files (pd.DataFrame): DataFrame of TX file information.
        output_dir (str): TX data directory.
        proposal_ID (str): Proposal ID.
        apid (str): Analysis project ID.
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: TX metadata DataFrame.
    """

    if os.path.exists(f"{output_dir}/portal_metadata.csv"):
        log.info("TX metadata already pulled from source.")
        tx_metadata = pd.read_csv(f"{output_dir}/portal_metadata.csv")
        return tx_metadata
    else:
        if superuser:
            log.info("You are a superuser. Proceeding with pulling TX metadata from source...")
        else:
            log.info(f"Source file {output_dir}/portal_metadata.csv does not exist.")
            raise ValueError("You are not currently authorized to download transcriptomics metadata from source. Please contact your JGI project manager for access.")

    myfields = [
        "metadata.proposal_id",
        "metadata.library_name",
        "file_name",
        "metadata.sequencing_project_id",
        "metadata.sequencing_project.sequencing_project_name",
        "metadata.sequencing_project.sequencing_product_name",
        "metadata.final_deliv_project_id",
        "metadata.sow_segment.sample_name",
        "metadata.sow_segment.sample_isolated_from",
        "metadata.sow_segment.collection_isolation_site_or_growth_conditions",
        "metadata.sow_segment.ncbi_tax_id",
        "metadata.sow_segment.species",
        "metadata.sow_segment.strain",
        "metadata.physical_run.dt_sequencing_end",
        "metadata.physical_run.instrument_type",
        "metadata.input_read_count",
        "metadata.filter_reads_count",
        "metadata.filter_reads_count_pct"
    ]

    def fetch_sample_info(lib_IDs):

        if len(lib_IDs) < 50:
            lib_group = [1] * len(lib_IDs)
        else:
            lib_group = pd.cut(range(len(lib_IDs)), bins=round(len(lib_IDs) / 50) + 1, labels=False)

        df_list = []
        for gg in set(lib_group):
            selected_libs = [lib_IDs[i] for i in range(len(lib_IDs)) if lib_group[i] == gg]
            myCmd = (
                f"module load jamo; jamo report select {','.join(myfields)} "
                f"where metadata.fastq_type=filtered, metadata.library_name in \\({','.join(selected_libs)}\\) "
                f"as txt 2>/dev/null | sed 's/^{proposal_ID}/!{proposal_ID}/g' | tr -d '\\n' | tr '!{proposal_ID}' '\\n{proposal_ID}'"
            )
            result = subprocess.run(myCmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                df = pd.read_csv(io.StringIO(result.stdout), sep='\t', names=[field.replace('metadata.', '').replace('sow_segment.', '') for field in myfields])
                df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    tx_metadata_list = []
    for _, row in tx_files.iterrows():
        lib_IDs = row['libIDs']
        if isinstance(lib_IDs, str) and ',' in lib_IDs:
            lib_IDs = lib_IDs.split(',')
        df = fetch_sample_info(lib_IDs)
        df = df[df['library_name'].isin(lib_IDs)]
        df['APID'] = row['APID']
        tx_metadata_list.append(df)

    tx_metadata = pd.concat(tx_metadata_list, ignore_index=True)
    tx_metadata['ix'] = range(1, len(tx_metadata) + 1)
    tx_metadata = tx_metadata[['ix'] + [col for col in tx_metadata.columns if col != 'ix']]
    tx_metadata = tx_metadata[tx_metadata['APID'].astype(str) == str(apid)]

    log.info("Saving TX metadata...")
    write_integration_file(data=tx_metadata, output_dir=output_dir, filename="portal_metadata", indexing=False)
    #display(tx_metadata.head())
    return tx_metadata

# ====================================
# Feature annotation functions
# ====================================

def generate_tx_annotation_table(
    raw_data: pd.DataFrame,
    raw_data_dir: str,
    genome_type: str,
    output_dir: str,
    output_filename: str,
) -> pd.DataFrame:
    """
    Generate a merged gene annotation table from multiple annotation files.
    
    Args:
        raw_data (pd.DataFrame): Raw transcriptomics data with gene IDs as index
        raw_data_dir (str): Directory containing annotation table files
        genome_type (str): Type of genome - "microbe", "algal", "metagenome", or "plant"
        output_dir (str): Output directory for saving the merged annotation table
        output_filename (str): Filename for the output annotation table
    
    Returns:
        pd.DataFrame: Merged annotation table with transcriptome_id and annotation columns
    """
    
    if genome_type == "microbe":
        annotation_df = _process_microbe_annotations(raw_data_dir, output_dir, output_filename)
    elif genome_type == "algal":
        annotation_df = _process_algal_annotations(raw_data_dir, output_dir, output_filename)
    elif genome_type == "metagenome":
        log.info(f"Annotation processing for '{genome_type}' genome type is not yet implemented.")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df
    elif genome_type == "plant":
        log.info(f"Annotation processing for '{genome_type}' genome type is not yet implemented.")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df
    else:
        raise ValueError(f"Invalid genome_type '{genome_type}'. Must be one of: 'microbe', 'algal', 'metagenome', 'plant'")

    if not raw_data.empty and not annotation_df.empty:
        _validate_annotation_gene_ids(annotation_df, raw_data)
    else:
        raise ValueError("Either input raw_data or computed annotation_df is empty. Cannot validate gene IDs.")

    annotation_df = annotation_df.set_index('transcriptome_id')
    annotation_df = annotation_df.map(lambda x: str(x).replace('|', ';;') if isinstance(x, str) else x)
    write_integration_file(annotation_df, output_dir, output_filename, indexing=True)

    return annotation_df

def _validate_annotation_gene_ids(annotation_df: pd.DataFrame, raw_data: pd.DataFrame) -> None:
    """
    Validate that gene IDs in annotation table match those in raw data.
    
    Args:
        annotation_df (pd.DataFrame): Annotation table with transcriptome_id column
        raw_data (pd.DataFrame): Raw transcriptomics data with gene IDs as GeneID column
    """
    
    # If "tx_" prefix is used in raw data, ensure annotation gene IDs also have it
    if all(str(gid).startswith('tx_') for gid in raw_data['GeneID']):
        if not all(str(gid).startswith('tx_') for gid in annotation_df['transcriptome_id']):
            annotation_df['transcriptome_id'] = annotation_df['transcriptome_id'].apply(lambda x: f'tx_{x}')

    # Get gene IDs from both datasets
    raw_data_genes = set(raw_data['GeneID'].tolist())
    annotation_genes = set(annotation_df['transcriptome_id'].tolist())
    
    # Calculate overlap statistics
    common_genes = raw_data_genes.intersection(annotation_genes)
    raw_only = raw_data_genes - annotation_genes
    annotation_only = annotation_genes - raw_data_genes
    
    # Log validation results
    log.info(f"Gene ID Validation Results:")
    log.info(f"  Raw data genes: {len(raw_data_genes)}")
    log.info(f"  Annotation genes: {len(annotation_genes)}")
    log.info(f"  Common genes: {len(common_genes)} ({len(common_genes)/len(raw_data_genes)*100:.1f}% of raw data)")
    log.info(f"  Raw data only: {len(raw_only)}")
    log.info(f"  Annotation only: {len(annotation_only)}")
    
    # Warn if low overlap
    overlap_pct = len(common_genes) / len(raw_data_genes) * 100
    if overlap_pct < 50:
        log.warning(f"Low gene ID overlap ({overlap_pct:.1f}%) between raw data and annotations")
    if overlap_pct == 0:
        log.info("Raw data GeneIDs:")
        log.info(raw_data['GeneID'].tolist()[:10])
        log.info("Annotation transcriptome_ids:")
        log.info(annotation_df['transcriptome_id'].tolist()[:10])
        raise ValueError("No matching gene IDs found between raw data and annotations, something is wrong.")

def _process_microbe_annotations(
    raw_data_dir: str,
    output_dir: str,
    output_filename: str
) -> pd.DataFrame:
    """
    Process microbe annotation files and merge into a single table.
    
    Args:
        raw_data_dir (str): Directory containing annotation files
        output_dir (str): Output directory
        output_filename (str): Output filename
        
    Returns:
        pd.DataFrame: Merged annotation table
    """
    
    log.info(f"Processing microbe annotations from {raw_data_dir}")
    
    # Find all annotation table files
    annotation_files = glob.glob(os.path.join(raw_data_dir, "*_annotation_table.tsv"))
    
    if not annotation_files:
        log.warning(f"No *_annotation_table.tsv files found in {raw_data_dir}")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        return empty_df
    
    log.info(f"Found {len(annotation_files)} annotation files:")
    for file in annotation_files:
        log.info(f"  {os.path.basename(file)}")
    
    # Process each annotation file
    processed_dfs = []
    for file_path in annotation_files:
        df = _read_and_select_microbe_annotations(file_path)
        if df is not None and not df.empty:
            agg_df = _aggregate_gene_annotations(df)
            processed_dfs.append(agg_df)
            log.info(f"  Processed {os.path.basename(file_path)}: {len(agg_df)} genes")
        else:
            log.warning(f"  Skipped {os.path.basename(file_path)}: no valid data")
    
    if not processed_dfs:
        log.warning("No valid annotation data found in any files")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        return empty_df
    
    # Merge all annotation dataframes
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='transcriptome_id', how='outer'), 
        processed_dfs
    )
    
    # Fill NaN values with empty strings for consistency
    merged_df = merged_df.fillna('')
    
    # Add protein ID mapping from GFF3 file if available
    merged_df = _add_protein_id_mapping(merged_df, raw_data_dir, "microbe")
    
    # Compress the dataframe to ensure one gene per row with semicolon-separated annotations
    final_df = _compress_annotation_table(merged_df)
    
    log.info(f"Final annotation table: {len(final_df)} genes with {len(final_df.columns)} annotation columns")    
    
    return final_df

def _aggregate_gene_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate microbe annotations by transcriptome_id, handling multiple annotations per gene.
    
    Args:
        df (pd.DataFrame): Annotation dataframe with transcriptome_id column
        
    Returns:
        pd.DataFrame: Aggregated dataframe with one row per transcriptome_id
    """
    
    def join_unique_values(series):
        """Join unique non-empty values with double semicolon"""
        unique_vals = sorted({
            str(x).strip() for x in series.dropna() 
            if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', '']
        })
        return ';;'.join(unique_vals) if unique_vals else ''
    
    # Group by transcriptome_id and aggregate all other columns
    annotation_columns = [col for col in df.columns if col != 'transcriptome_id']
    
    # Group by transcriptome_id and aggregate
    aggregated_df = df.groupby('transcriptome_id')[annotation_columns].agg(join_unique_values).reset_index()
    
    return aggregated_df

def _process_algal_annotations(
    raw_data_dir: str,
    output_dir: str,
    output_filename: str
) -> pd.DataFrame:
    """
    Process algal annotation files and merge into a single table.
    
    Args:
        raw_data_dir (str): Directory containing annotation files
        output_dir (str): Output directory
        output_filename (str): Output filename
        
    Returns:
        pd.DataFrame: Merged annotation table
    """
    
    log.info(f"Processing algal annotations from {raw_data_dir}")
    
    # Find all annotation table files
    annotation_files = glob.glob(os.path.join(raw_data_dir, "*_annotation_table.tsv"))
    
    if not annotation_files:
        log.warning(f"No *_annotation_table.tsv files found in {raw_data_dir}")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df
    
    log.info(f"Found {len(annotation_files)} annotation files:")
    for file in annotation_files:
        log.info(f"  {os.path.basename(file)}")
    
    # Process each annotation file
    processed_dfs = []
    for file_path in annotation_files:
        df = _read_and_select_algal_annotations(file_path)
        if df is not None and not df.empty:
            agg_df = _aggregate_algal_annotations(df)
            processed_dfs.append(agg_df)
            log.info(f"  Processed {os.path.basename(file_path)}: {len(agg_df)} proteins")
        else:
            log.warning(f"  Skipped {os.path.basename(file_path)}: no valid data")
    
    if not processed_dfs:
        log.warning("No valid annotation data found in any files")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df
    
    # Merge all annotation dataframes based on protein_id
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on='protein_id', how='outer'), 
        processed_dfs
    )
    
    # Fill NaN values with empty strings for consistency
    merged_df = merged_df.fillna('')
    
    # Add transcriptome_id mapping from GFF3 file (protein_id -> transcriptome_id)
    merged_df = _add_protein_id_mapping(merged_df, raw_data_dir, "algal")
    
    # Compress the dataframe to ensure one gene per row with semicolon-separated annotations
    final_df = _compress_annotation_table(merged_df)
    
    log.info(f"Final annotation table: {len(final_df)} genes with {len(final_df.columns)} annotation columns")
    
    # Save the merged annotation table
    write_integration_file(final_df, output_dir, output_filename, indexing=False)
    
    return final_df

def _read_and_select_microbe_annotations(file_path: str) -> pd.DataFrame:
    """
    Read and select relevant columns from microbe annotation files.
    
    Args:
        file_path (str): Path to annotation file
        
    Returns:
        pd.DataFrame: Processed annotation dataframe
    """
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Determine annotation type based on filename
        filename = os.path.basename(file_path)
        
        if 'cog_annotation_table' in filename:
            selected_cols = ['gene_oid', 'cog_id', 'cog_name']
            df = df[selected_cols].copy()
            df.rename(columns={'gene_oid': 'transcriptome_id', 'cog_id': 'cog_acc', 'cog_name': 'cog_desc'}, inplace=True)
            
        elif 'ipr_annotation_table' in filename:
            selected_cols = ['gene_oid', 'iprid', 'iprdesc', 'go_info']
            df = df[selected_cols].copy()
            df.rename(columns={'gene_oid': 'transcriptome_id', 'iprid': 'ipr_acc', 'iprdesc': 'ipr_desc', 'go_info': 'go_acc'}, inplace=True)
            
        elif 'kegg_annotation_table' in filename:
            selected_cols = ['gene_oid', 'ko_id', 'ko_name']
            df = df[selected_cols].copy()
            df.rename(columns={'gene_oid': 'transcriptome_id', 'ko_id': 'kegg_acc', 'ko_name': 'kegg_desc'}, inplace=True)
            
        elif 'pfam_annotation_table' in filename:
            selected_cols = ['gene_oid', 'pfam_id', 'pfam_name']
            df = df[selected_cols].copy()
            df.rename(columns={'gene_oid': 'transcriptome_id', 'pfam_id': 'pfam_acc', 'pfam_name': 'pfam_desc'}, inplace=True)
            
        elif 'tigrfam_annotation_table' in filename:
            selected_cols = ['gene_oid', 'tigrfam_id', 'tigrfam_name']
            df = df[selected_cols].copy()
            df.rename(columns={'gene_oid': 'transcriptome_id', 'tigrfam_id': 'tigrfam_acc', 'tigrfam_name': 'tigrfam_desc'}, inplace=True)
            
        else:
            log.warning(f"Unknown annotation file type: {filename}")
            return None
            
        return df
        
    except Exception as e:
        log.warning(f"Error reading file {file_path}: {e}")
        return None

def _read_and_select_algal_annotations(file_path: str) -> pd.DataFrame:
    """
    Read and select relevant columns from algal annotation files.
    
    Args:
        file_path (str): Path to annotation file
        
    Returns:
        pd.DataFrame: Processed annotation dataframe
    """
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Determine annotation type based on filename
        filename = os.path.basename(file_path)
        
        if 'go_annotation_table' in filename:
            # Select proteinId and GO annotation columns
            selected_cols = ['#proteinId', 'gotermId', 'goName', 'gotermType', 'goAcc']
            df = df[selected_cols].copy()
            df.rename(columns={
                '#proteinId': 'protein_id',
                'gotermId': 'go_id',
                'goName': 'go_name',
                'gotermType': 'go_type',
                'goAcc': 'go_acc'
            }, inplace=True)
            
        elif 'ipr_annotation_table' in filename:
            # Select proteinId and InterPro annotation columns
            selected_cols = ['#proteinId', 'iprId', 'iprDesc']
            df = df[selected_cols].copy()
            df.rename(columns={
                '#proteinId': 'protein_id',
                'iprId': 'ipr_acc',
                'iprDesc': 'ipr_desc',
                'goAcc': 'go_acc'
            }, inplace=True)
            
        elif 'kegg_annotation_table' in filename:
            # Select proteinId and KEGG annotation columns
            selected_cols = ['#proteinId', 'ecNum', 'definition']
            df = df[selected_cols].copy()
            df.rename(columns={
                '#proteinId': 'protein_id',
                'ecNum': 'kegg_acc',
                'definition': 'kegg_desc'
            }, inplace=True)
            
        elif 'kog_annotation_table' in filename:
            # Select proteinId and KOG annotation columns
            selected_cols = ['proteinId', 'kogid', 'kogdefline']
            df = df[selected_cols].copy()
            df.rename(columns={
                'proteinId': 'protein_id',
                'kogid': 'kog_acc',
                'kogdefline': 'kog_desc'
            }, inplace=True)
            
        else:
            log.warning(f"Unknown annotation file type: {filename}")
            return None
            
        return df
        
    except Exception as e:
        log.warning(f"Error reading file {file_path}: {e}")
        return None

def _aggregate_algal_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate algal annotations by protein_id, handling multiple annotations per protein.
    
    Args:
        df (pd.DataFrame): Annotation dataframe with protein_id column
        
    Returns:
        pd.DataFrame: Aggregated dataframe with one row per protein_id
    """
    
    def join_unique_values(series):
        """Join unique non-empty values with double semicolon"""
        unique_vals = sorted({
            str(x).strip() for x in series.dropna() 
            if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', '']
        })
        return ';;'.join(unique_vals) if unique_vals else ''
    
    # Group by protein_id and aggregate all other columns
    annotation_columns = [col for col in df.columns if col != 'protein_id']
    
    # Group by protein_id and aggregate
    aggregated_df = df.groupby('protein_id')[annotation_columns].agg(join_unique_values).reset_index()
    
    return aggregated_df

def _add_gene_id_mapping(
    merged_df: pd.DataFrame,
    raw_data_dir: str,
    genome_type: str
) -> pd.DataFrame:
    """
    Add transcriptome_id mapping from GFF3 file to the merged annotation table for algal genomes.
    Maps protein_id to transcriptome_id and adds display_name from product field.
    
    Args:
        merged_df (pd.DataFrame): Merged annotation dataframe with protein_id
        raw_data_dir (str): Directory containing GFF3 file
        genome_type (str): Genome type
        
    Returns:
        pd.DataFrame: Annotation dataframe with transcriptome_id and display_name columns added
    """
    
    # Look for GFF3 file in the raw data directory
    gff_files = glob.glob(os.path.join(raw_data_dir, "*.gff3"))
    if not gff_files:
        gff_files = glob.glob(os.path.join(raw_data_dir, "genes.gff3"))
    
    if not gff_files:
        log.warning("No GFF3 file found for gene ID mapping")
        # Create transcriptome_id from protein_id if no GFF3 found
        merged_df['transcriptome_id'] = merged_df['protein_id']
        merged_df['display_name'] = ''
        return merged_df
    
    gff_file = gff_files[0]  # Use first GFF3 file found
    log.info(f"Adding gene ID mapping from {os.path.basename(gff_file)}")
    
    # Parse GFF3 file to create protein_id -> gene_id mapping and product mapping
    protein_to_gene = {}
    protein_to_product = {}
    
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
                
            fields = line.strip().split('\t')
            if len(fields) >= 9 and fields[2] == 'gene':
                attributes = fields[8]
                
                # Extract gene ID from ID= field
                gene_match = re.search(r'ID=([^;]+)', attributes)
                if not gene_match:
                    continue
                gene_id = gene_match.group(1)
                
                # Extract protein ID from proteinId= field
                protein_match = re.search(r'proteinId=([^;]+)', attributes)
                if protein_match:
                    protein_id = protein_match.group(1)
                    protein_to_gene[str(protein_id)] = str(gene_id)
                    
                    # Extract product information from product= field
                    product_match = re.search(r'product=([^;]+)', attributes)
                    if product_match:
                        product = product_match.group(1).strip()
                        protein_to_product[str(protein_id)] = product
                    else:
                        protein_to_product[str(protein_id)] = ''
    
    log.info(f"Found {len(protein_to_gene)} protein-to-gene mappings")
    log.info(f"Found {len(protein_to_product)} protein-to-product mappings")
    
    # Add extracted columns to merged_df
    merged_df['protein_id'] = merged_df['protein_id'].astype(str)
    merged_df['transcriptome_id'] = merged_df['protein_id'].map(protein_to_gene).fillna(merged_df['protein_id'])
    merged_df['display_name'] = merged_df['protein_id'].map(protein_to_product).fillna('')

    return merged_df

def _add_protein_id_mapping(
    merged_df: pd.DataFrame,
    raw_data_dir: str,
    genome_type: str
) -> pd.DataFrame:
    """
    Add protein ID mapping and display name from GFF3 file to the merged annotation table.
    
    Args:
        merged_df (pd.DataFrame): Merged annotation dataframe
        raw_data_dir (str): Directory containing GFF3 file
        genome_type (str): Genome type ("microbe", "algal", etc.)
        
    Returns:
        pd.DataFrame: Annotation dataframe with protein ID and display_name columns added
    """
    
    # Look for GFF3 file in the raw data directory
    gff_files = glob.glob(os.path.join(raw_data_dir, "*.gff3"))
    if not gff_files:
        gff_files = glob.glob(os.path.join(raw_data_dir, "genes.gff3"))
    
    if not gff_files:
        log.warning("No GFF3 file found for protein ID mapping")
        merged_df['protein_id'] = ''
        merged_df['transcriptome_id'] = ''
        merged_df['display_name'] = ''
        return merged_df
    
    gff_file = gff_files[0]  # Use first GFF3 file found
    log.info(f"Adding protein ID mapping and display names from {os.path.basename(gff_file)}")
    
    # Parse GFF3 file to create gene_id -> protein_id mapping and product mapping
    gene_to_protein = {}
    gene_to_product = {}
    
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            # Set feature type, protein attribute, and display attribute based on genome type
            if genome_type == "algal":
                feature_type = 'gene'
                protein_attribute = 'proteinId'
                display_attribute = 'product_name'
            elif genome_type == "microbe":
                feature_type = 'CDS'
                protein_attribute = 'locus_tag'
                display_attribute = 'product'
            else:
                # For other genome types, try gene first, then CDS
                feature_type = 'gene'
                protein_attribute = 'proteinId'
                display_attribute = 'product'
            
            if fields[2] == feature_type:
                attributes = fields[8]
                
                # Extract gene ID
                gene_match = re.search(r'ID=([^;]+)', attributes)
                if not gene_match:
                    continue
                gene_id = gene_match.group(1)
                
                # Extract protein ID using the appropriate attribute
                protein_id = None
                pattern = rf"{re.escape(protein_attribute)}=([^;]+)"
                protein_match = re.search(pattern, attributes)
                if protein_match:
                    protein_id = protein_match.group(1)
                
                # Extract product information from the appropriate display attribute
                product = ''
                display_pattern = rf"{re.escape(display_attribute)}=([^;]+)"
                product_match = re.search(display_pattern, attributes)
                if product_match:
                    product = product_match.group(1).strip()
                
                if protein_id:
                    gene_to_protein[gene_id] = protein_id
                
                # Store product information regardless of whether protein_id exists
                if product:
                    gene_to_product[gene_id] = product
    
    log.info(f"Found {len(gene_to_protein)} gene-to-protein mappings")
    log.info(f"Found {len(gene_to_product)} gene-to-product mappings")

    merged_df['transcriptome_id'] = merged_df['transcriptome_id'].astype(str)
    merged_df['protein_id'] = merged_df['transcriptome_id'].map(gene_to_protein).fillna('')
    merged_df['display_name'] = merged_df['transcriptome_id'].map(gene_to_product).fillna('')

    return merged_df

def _compress_annotation_table(
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compress annotation table to ensure one gene per row with semicolon-separated annotations.
    
    Args:
        merged_df (pd.DataFrame): Merged annotation dataframe potentially with duplicate genes
        
    Returns:
        pd.DataFrame: Compressed dataframe with one row per gene
    """
    
    def join_unique_values(series):
        """Join unique non-empty values with double semicolon"""
        unique_vals = sorted({
            str(x).strip() for x in series.dropna() 
            if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', '']
        })
        return ';;'.join(unique_vals) if unique_vals else ''
    
    # Group by transcriptome_id and aggregate all other columns
    annotation_columns = [col for col in merged_df.columns if col != 'transcriptome_id']
    
    # Group by transcriptome_id and aggregate
    compressed_df = merged_df.groupby('transcriptome_id')[annotation_columns].agg(join_unique_values).reset_index()
    
    log.info(f"Compressed annotations from {len(merged_df)} rows to {len(compressed_df)} unique genes")
    
    return compressed_df

def generate_mx_annotation_table(
    raw_data: pd.DataFrame, 
    dataset_raw_dir: str, 
    polarity: str, 
    output_dir: str, 
    output_filename: str,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Generate metabolite ID to annotation mapping table for metabolomics data.
    
    Args:
        raw_data: Raw metabolomics data DataFrame
        dataset_raw_dir: Directory containing raw dataset files
        polarity: Polarity setting ('positive', 'negative', 'multipolarity')
        output_dir: Output directory for saving annotation map
        output_filename: Filename for saving the annotation map
        overwrite: Overwrite existing output if True
    
    Returns:
        pd.DataFrame: annotation_table DataFrame
    """
    
    # Get metabolite IDs from raw data
    if not raw_data.empty:
        if 'CompoundID' in raw_data.columns:
            metabolite_ids = set(raw_data['CompoundID'].tolist())
        else:
            metabolite_ids = set(raw_data.index.tolist())
    else:
        metabolite_ids = set()
    
    log.info(f"Found {len(metabolite_ids)} metabolites in raw data")
    
    # Find and process FBMN library-results files
    log.info(f"Looking for FBMN library-results files in {dataset_raw_dir}")
    
    fbmn_data = pd.DataFrame()
    try:
        if polarity == "multipolarity":
            compound_files = glob.glob(os.path.expanduser(f"{dataset_raw_dir}/*/*library-results.tsv"))
            if len(compound_files) > 1:
                # Process multiple files
                all_fbmn_data = []
                for file_path in compound_files:
                    file_polarity = ("positive" if "positive" in file_path 
                                else "negative" if "negative" in file_path 
                                else "unknown")
                    df = pd.read_csv(file_path, sep='\t')
                    df['metabolite_id'] = 'mx_' + df['#Scan#'].astype(str) + '_' + file_polarity
                    all_fbmn_data.append(df)
                fbmn_data = pd.concat(all_fbmn_data, axis=0, ignore_index=True)
            elif len(compound_files) == 1:
                log.warning("Only single compound file found with multipolarity setting.")
                fbmn_data = pd.read_csv(compound_files[0], sep='\t')
                fbmn_data['metabolite_id'] = 'mx_' + fbmn_data['#Scan#'].astype(str) + '_unknown'
                
        elif polarity in ["positive", "negative"]:
            compound_files = glob.glob(os.path.expanduser(f"{dataset_raw_dir}/*/*{polarity}*library-results.tsv"))
            if len(compound_files) == 1:
                fbmn_data = pd.read_csv(compound_files[0], sep='\t')
                fbmn_data['metabolite_id'] = 'mx_' + fbmn_data['#Scan#'].astype(str) + '_' + polarity
                log.info(f"Using compound file: {compound_files[0]}")
            elif len(compound_files) > 1:
                log.warning(f"Multiple compound files found: {compound_files}. Using first one.")
                fbmn_data = pd.read_csv(compound_files[0], sep='\t')
                fbmn_data['metabolite_id'] = 'mx_' + fbmn_data['#Scan#'].astype(str) + '_' + polarity
            else:
                log.warning(f"No compound files found for {polarity} polarity.")
        else:
            log.warning(f"Unknown polarity: {polarity}")
                
    except Exception as e:
        log.warning(f"Error reading compound files: {e}")
    
    # Create annotation mapping for all metabolites
    log.info("Creating metabolite annotation mapping...")
    mapping_data = []
    
    for met_id in tqdm(sorted(metabolite_ids), desc="Processing metabolites", unit="metabolite"):
        if not fbmn_data.empty:
            # Find ALL matching annotations for this metabolite ID
            matches = fbmn_data[fbmn_data['metabolite_id'] == met_id]
            if len(matches) > 0:
                # Helper function to concatenate unique values
                def concat_unique_values(series):
                    """Concatenate unique non-null values with ;;"""
                    unique_vals = sorted({
                        str(val).strip() for val in series.dropna() 
                        if str(val).strip() and str(val).strip().lower() not in ['nan', 'none', '']
                    })
                    return ';;'.join(unique_vals) if unique_vals else None
                
                # Handle InChiKey variations and concatenate all values
                inchikey_cols = ['InChiKey', 'InChIKey', 'inchikey', 'INCHIKEY']
                inchikey_values = []
                for col in inchikey_cols:
                    if col in matches.columns:
                        inchikey_values.extend(matches[col].dropna().tolist())
                
                mapping_data.append({
                    'metabolite_id': met_id,
                    'molecular_formula': concat_unique_values(matches.get('molecular_formula', pd.Series())),
                    'Compound_Name': concat_unique_values(matches.get('Compound_Name', pd.Series())),
                    'INCHI': concat_unique_values(matches.get('INCHI', pd.Series())),
                    'InChiKey': concat_unique_values(pd.Series(inchikey_values)),
                    'superclass': concat_unique_values(matches.get('superclass', pd.Series())),
                    'class': concat_unique_values(matches.get('class', pd.Series())),
                    'subclass': concat_unique_values(matches.get('subclass', pd.Series())),
                    'npclassifier_superclass': concat_unique_values(matches.get('npclassifier_superclass', pd.Series())),
                    'npclassifier_class': concat_unique_values(matches.get('npclassifier_class', pd.Series())),
                    'npclassifier_pathway': concat_unique_values(matches.get('npclassifier_pathway', pd.Series())),
                    'library_usi': concat_unique_values(matches.get('library_usi', pd.Series()))
                })
                continue
        
        # No annotation found
        mapping_data.append({
            'metabolite_id': met_id,
            'molecular_formula': None,
            'Compound_Name': None,
            'INCHI': None,
            'InChiKey': None,
            'superclass': None,
            'class': None,
            'subclass': None,
            'npclassifier_superclass': None,
            'npclassifier_class': None,
            'npclassifier_pathway': None,
            'library_usi': None
        })
    
    # Create and save annotation DataFrame
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.rename(columns={'metabolite_id': 'metabolome_id'}, inplace=True)
    mapping_df = mapping_df.set_index('metabolome_id')
    mapping_df = mapping_df.map(lambda x: str(x).replace('|', ';;') if isinstance(x, str) else x)
    mapping_df['display_name'] = mapping_df['Compound_Name']
    
    # Validate annotation IDs against raw data before saving
    if not raw_data.empty and not mapping_df.empty:
        _validate_annotation_metabolite_ids(mapping_df, raw_data)
    else:
        if raw_data.empty:
            log.warning("Raw data is empty, skipping validation")
        if mapping_df.empty:
            log.warning("Annotation table is empty, skipping validation")
    
    os.makedirs(output_dir, exist_ok=True)
    write_integration_file(data=mapping_df, output_dir=output_dir, filename=output_filename, indexing=True)
    
    # Summary statistics for annotation
    total_rows = len(mapping_df)
    annotated_count = mapping_df.dropna(subset=['INCHI', 'InChiKey'], how='all').shape[0]
    
    # Count metabolites with multiple annotations
    multi_annotation_count = 0
    for col in ['Compound_Name', 'superclass', 'class', 'subclass']:
        if col in mapping_df.columns:
            multi_count = mapping_df[col].str.contains(';;', na=False).sum()
            if multi_count > 0:
                #log.info(f"Metabolites with multiple {col} annotations: {multi_count}")
                multi_annotation_count = max(multi_annotation_count, multi_count)
    
    log.info(f"Created annotation mapping with {total_rows} rows")
    log.info(f"Metabolites with annotations: {annotated_count}")
    log.info(f"Metabolites without annotations: {total_rows - annotated_count}")
    log.info(f"Metabolites with multiple annotations: {multi_annotation_count}")
    
    return mapping_df

def _validate_annotation_metabolite_ids(annotation_df: pd.DataFrame, raw_data: pd.DataFrame) -> None:
    """
    Validate that metabolite IDs in annotation table match those in raw data.
    
    Args:
        annotation_df (pd.DataFrame): Annotation table with metabolome_id as index
        raw_data (pd.DataFrame): Raw metabolomics data with CompoundID column or metabolite IDs as index
    """
    
    # Get metabolite IDs from raw data
    if not raw_data.empty:
        if 'CompoundID' in raw_data.columns:
            raw_data_metabolites = set(raw_data['CompoundID'].tolist())
        else:
            raw_data_metabolites = set(raw_data.index.tolist())
    else:
        raw_data_metabolites = set()
    
    # Get metabolite IDs from annotation table (should be in index as metabolome_id)
    if hasattr(annotation_df.index, 'name') and annotation_df.index.name == 'metabolome_id':
        annotation_metabolites = set(annotation_df.index.tolist())
    elif 'metabolome_id' in annotation_df.columns:
        annotation_metabolites = set(annotation_df['metabolome_id'].tolist())
    else:
        # Fallback to index if no metabolome_id column
        annotation_metabolites = set(annotation_df.index.tolist())
    
    # Calculate overlap statistics
    common_metabolites = raw_data_metabolites.intersection(annotation_metabolites)
    raw_only = raw_data_metabolites - annotation_metabolites
    annotation_only = annotation_metabolites - raw_data_metabolites
    
    # Log validation results
    log.info(f"Metabolite ID Validation Results:")
    log.info(f"  Raw data metabolites: {len(raw_data_metabolites)}")
    log.info(f"  Annotation metabolites: {len(annotation_metabolites)}")
    log.info(f"  Common metabolites: {len(common_metabolites)} ({len(common_metabolites)/len(raw_data_metabolites)*100:.1f}% of raw data)")
    log.info(f"  Raw data only: {len(raw_only)}")
    log.info(f"  Annotation only: {len(annotation_only)}")
    
    # Warn if low overlap
    overlap_pct = len(common_metabolites) / len(raw_data_metabolites) * 100 if len(raw_data_metabolites) > 0 else 0
    if overlap_pct < 50:
        log.warning(f"Low metabolite ID overlap ({overlap_pct:.1f}%) between raw data and annotations")
    if overlap_pct == 0:
        log.info("Raw data metabolite IDs (first 10):")
        log.info(list(raw_data_metabolites)[:10])
        log.info("Annotation metabolite IDs (first 10):")
        log.info(list(annotation_metabolites)[:10])
        raise ValueError("No matching metabolite IDs found between raw data and annotations, something is wrong.")

def annotate_integrated_features(
    integrated_data: pd.DataFrame,
    datasets: List = None,
    output_dir: str = None,
    output_filename: str = None
) -> pd.DataFrame:
    """
    Build a combined annotation dataframe for integrated features from multiple datasets.
    
    Parameters
    ----------
    integrated_data : pd.DataFrame
        Integrated feature matrix (features x samples)
    datasets : List, optional
        List of dataset objects with annotation_table attributes
    output_dir : str, optional
        Directory to save the annotation results
    output_filename : str, default "integrated_features_annotated"
        Filename for the output annotation table
        
    Returns
    -------
    pd.DataFrame
        Combined annotation dataframe with columns for each annotation type
        and one row per feature from integrated_data
    """
    
    # Get all unique features from integrated data
    all_features = integrated_data.index.tolist()
    log.info(f"Annotating {len(all_features)} features from integrated data")
    
    # Start with a base dataframe containing all features
    final_annotation_df = pd.DataFrame(index=all_features)
    final_annotation_df.index.name = 'feature_id'
    
    # Track dataset-specific statistics
    dataset_stats = {}
    
    # Build combined annotation dataframe from datasets if provided
    if datasets is not None:
        for dataset in datasets:
            if hasattr(dataset, 'annotation_table') and dataset.annotation_table is not None:
                log.info(f"Processing annotation table for {dataset.dataset_name}")
                
                ann_table = dataset.annotation_table.copy()
                
                # Find potential ID columns in the annotation table
                potential_id_cols = []
                for col in ann_table.columns:
                    if 'id' in col.lower() or col == ann_table.index.name:
                        potential_id_cols.append(col)
                
                # Add dataset prefix to annotation columns to avoid conflicts
                ann_table.columns = [f"{dataset.dataset_name}_{col}" for col in ann_table.columns]
                
                # Count matches for this dataset
                matches = ann_table.index.intersection(all_features)
                
                # Count features from this dataset in integrated data
                dataset_features = [f for f in all_features if f.startswith(f"{dataset.dataset_name}_")]
                
                # Store dataset statistics
                dataset_stats[dataset.dataset_name] = {
                    'total_features_in_integrated': len(dataset_features),
                    'total_annotations_available': len(ann_table),
                    'features_with_annotations': len(matches),
                    'annotation_columns': len(ann_table.columns)
                }
                
                # Merge with the main annotation dataframe
                # Use left join to keep all features from integrated_data
                final_annotation_df = final_annotation_df.join(ann_table, how='left')
                
                log.info(f"  Dataset features in integrated data: {len(dataset_features)}")
                log.info(f"  Annotation records available: {len(ann_table)}")
                log.info(f"  Features with annotations: {len(matches)}")
                log.info(f"  Annotation columns added: {len(ann_table.columns)}")
    
    # Fill NaN values with 'Unassigned'
    final_annotation_df = final_annotation_df.fillna('Unassigned')
    
    # Reset index to make feature_id a column for easier handling
    final_annotation_df = final_annotation_df.reset_index()
    
    # Overall summary statistics
    n_features = len(final_annotation_df)
    n_annotation_cols = len(final_annotation_df.columns) - 1  # Exclude feature_id column
    
    # Count features with at least one non-"Unassigned" annotation
    non_unassigned_mask = (final_annotation_df.iloc[:, 1:] != 'Unassigned').any(axis=1)
    n_annotated_features = non_unassigned_mask.sum()
    
    log.info(f"Overall annotation summary:")
    log.info(f"  Total features: {n_features}")
    log.info(f"  Features with annotations: {n_annotated_features}")
    log.info(f"  Features without annotations: {n_features - n_annotated_features}")
    log.info(f"  Total annotation columns: {n_annotation_cols}")
    
    # Dataset-specific annotation summaries
    if dataset_stats:
        log.info(f"Dataset-specific annotation summaries:")
        for dataset_name, stats in dataset_stats.items():
            # Count features from this dataset that have annotations
            dataset_features = [f for f in all_features if f.startswith(f"{dataset_name}_")]
            if dataset_features:
                dataset_feature_indices = final_annotation_df[
                    final_annotation_df['feature_id'].isin(dataset_features)
                ].index
                
                # Check which of these features have annotations from any source
                dataset_annotation_cols = [col for col in final_annotation_df.columns 
                                         if col.startswith(f"{dataset_name}_")]
                
                if dataset_annotation_cols:
                    # Count features with dataset-specific annotations
                    dataset_annotated_mask = (
                        final_annotation_df.loc[dataset_feature_indices, dataset_annotation_cols] != 'Unassigned'
                    ).any(axis=1)
                    dataset_annotated_count = dataset_annotated_mask.sum()
                else:
                    dataset_annotated_count = 0
                
                # Count features with any annotations (from any dataset)
                any_annotation_mask = (
                    final_annotation_df.loc[dataset_feature_indices, 
                                          final_annotation_df.columns[1:]] != 'Unassigned'
                ).any(axis=1)
                any_annotated_count = any_annotation_mask.sum()
                
                annotation_rate = (dataset_annotated_count / len(dataset_features)) * 100 if dataset_features else 0
                
                log.info(f"  {dataset_name}:")
                log.info(f"    Features in integrated data: {len(dataset_features)}")
                log.info(f"    Features with {dataset_name} annotations: {dataset_annotated_count} ({annotation_rate:.1f}%)")
                log.info(f"    Features with any annotations: {any_annotated_count}")
                log.info(f"    Annotation columns for {dataset_name}: {len(dataset_annotation_cols)}")
    
    # Save results if output directory provided
    if output_dir:
        write_integration_file(
            data=final_annotation_df,
            output_dir=output_dir,
            filename=output_filename,
            indexing=False
        )
    
    return final_annotation_df

# def parse_gff3_proteinid_to_id(gff3_path: str) -> dict:
#     """
#     Parse a GFF3 file to map protein IDs to gene IDs.

#     Args:
#         gff3_path (str): Path to GFF3 file.

#     Returns:
#         dict: Mapping from proteinId to geneId.
#     """

#     open_func = gzip.open if gff3_path.endswith('.gz') else open
#     proteinid_to_id = {}
#     with open_func(gff3_path, 'rt') as f:
#         for line in f:
#             if line.startswith('#'):
#                 continue
#             fields = line.strip().split('\t')
#             if len(fields) < 9 or fields[2] != 'gene':
#                 continue
#             attrs = {kv.split('=')[0]: kv.split('=')[1] for kv in fields[8].split(';') if '=' in kv}
#             protein_id = attrs.get('proteinId')
#             gene_id = attrs.get('ID')
#             if protein_id and gene_id:
#                 proteinid_to_id[protein_id.strip()] = gene_id.strip()
#     return proteinid_to_id

# def parse_annotation_file(annotation_path: str, tx_id_col: str, tx_annotation_col: str) -> dict:
#     """
#     Parse an annotation file to map IDs to annotation values.

#     Args:
#         annotation_path (str): Path to annotation file.
#         tx_id_col (str): Column name for IDs.
#         tx_annotation_col (str): Column name for annotation.

#     Returns:
#         dict: Mapping from ID to annotation.
#     """
    
#     df = pd.read_csv(annotation_path)
#     if tx_id_col not in df.columns or tx_annotation_col not in df.columns:
#         raise ValueError(f"Required columns {tx_id_col} or {tx_annotation_col} not found in annotation file.")
#     df = df.dropna(subset=[tx_id_col, tx_annotation_col])
#     annotation_table = dict(zip(df[tx_id_col].astype(str).str.strip(), df[tx_annotation_col].astype(int).astype(str).str.strip()))
#     return annotation_table

# def map_ids_to_annotations(
#     gff3_path: str,
#     annotation_path: str,
#     tx_id_col: str,
#     tx_annotation_col: str
# ) -> dict:
#     """
#     Map gene IDs from a GFF3 file to annotation values using an annotation file.

#     Args:
#         gff3_path (str): Path to GFF3 file.
#         annotation_path (str): Path to annotation file.
#         tx_id_col (str): ID column name.
#         tx_annotation_col (str): Annotation column name.

#     Returns:
#         dict: Mapping from gene ID to annotation.
#     """
    
#     proteinid_to_id = parse_gff3_proteinid_to_id(gff3_path)
#     annotation_table = parse_annotation_file(annotation_path, tx_id_col, tx_annotation_col)
#     id_to_annotation = {}
#     for protein_id, gene_id in proteinid_to_id.items():
#         for annot_key in annotation_table:
#             if protein_id in annot_key:
#                 id_to_annotation[gene_id] = annotation_table[annot_key]
#                 break
#     return id_to_annotation

# def annotate_genes_to_rhea(
#     gff3_path: str,
#     annotation_path: str,
#     tx_id_col: str,
#     tx_annotation_col: str
# ) -> pd.DataFrame:
#     """
#     Annotate gene IDs with Rhea IDs using GFF3 and annotation files.

#     Args:
#         gff3_path (str): Path to GFF3 file.
#         annotation_path (str): Path to annotation file.
#         tx_id_col (str): ID column name.
#         tx_annotation_col (str): Annotation column name.

#     Returns:
#         pd.DataFrame: DataFrame mapping gene IDs to Rhea IDs.
#     """

#     id_to_annotation = map_ids_to_annotations(gff3_path, annotation_path, tx_id_col, tx_annotation_col)
#     id_to_annotation = pd.DataFrame.from_dict(id_to_annotation, orient='index')
#     id_to_annotation.index.name = tx_id_col
#     id_to_annotation.rename(columns={0: tx_annotation_col}, inplace=True)
#     id_to_annotation[tx_id_col] = id_to_annotation.index
#     return id_to_annotation

# def check_annotation_ids(
#     annotated_data: pd.DataFrame,
#     dtype: str,
#     id_col: str,
#     method: str
# ) -> pd.DataFrame:
#     """
#     Ensure annotation DataFrame has correct index format and matches integrated data.

#     Args:
#         annotated_data (pd.DataFrame): Annotation DataFrame.
#         dtype (str): Data type prefix ('tx' or 'mx').
#         id_col (str): ID column name.
#         method (str): Annotation method.

#     Returns:
#         pd.DataFrame: Checked annotation DataFrame.
#     """

#     if dtype+"_" in annotated_data[id_col].astype(str).values[0]:
#         annotated_data.index = annotated_data[id_col].astype(str)
#     else:
#         annotated_data.index = dtype+"_" + annotated_data[id_col].astype(str)
#     if len(annotated_data.index.intersection(annotated_data.index)) == 0:
#         log.info(f"Warning! No matching indexes between integrated_data and annotation done with {method}. Please check the annotation file.")
#         return None
#     return annotated_data

# def annotate_integrated_data(
#     integrated_data: pd.DataFrame,
#     output_dir: str,
#     project_name: str,
#     tx_dir: str,
#     mx_dir: str,
#     magi2_dir: str,
#     tx_annotation_file: str,
#     mx_annotation_file: str,
#     tx_annotation_method: str,
#     mx_annotation_method: str,
#     tx_id_col: str,
#     mx_id_col: str,
#     tx_annotation_col: str,
#     mx_annotation_col: str,
#     overwrite: bool = False
# ) -> pd.DataFrame:
#     """
#     Annotate integrated data features with functional information from transcriptomics and metabolomics.

#     Args:
#         integrated_data (pd.DataFrame): Integrated feature matrix.
#         output_dir (str): Output directory.
#         project_name (str): Project name.
#         tx_dir (str): Transcriptomics data directory.
#         mx_dir (str): Metabolomics data directory.
#         magi2_dir (str): MAGI2 results directory.
#         tx_annotation_file (str): Transcriptomics annotation file.
#         mx_annotation_file (str): Metabolomics annotation file.
#         tx_annotation_method (str): Annotation method for transcriptomics.
#         mx_annotation_method (str): Annotation method for metabolomics.
#         tx_id_col (str): Transcriptomics ID column.
#         mx_id_col (str): Metabolomics ID column.
#         tx_annotation_col (str): Transcriptomics annotation column.
#         mx_annotation_col (str): Metabolomics annotation column.
#         overwrite (bool): Overwrite existing results.

#     Returns:
#         pd.DataFrame: DataFrame with feature annotations.
#     """
    
#     annotated_data_filename = f"integrated_data_annotated"
#     if os.path.exists(f"{output_dir}/{annotated_data_filename}.csv") and overwrite is False:
#         log.info(f"Integrated data already annotated: {output_dir}/{annotated_data_filename}.csv")
#         annotated_data = pd.read_csv(f"{output_dir}/{annotated_data_filename}.csv", index_col=0)
#         return annotated_data

#     if tx_annotation_method == "custom":
#         if tx_id_col is None or tx_annotation_col is None:
#             raise ValueError("Please provide tx_id_col and tx_annotation_col for custom annotation.")
#         sep = "\t" if tx_annotation_file.endswith((".txt", ".tsv", ".tab")) else ","
#         tx_annotation = pd.read_csv(glob.glob(os.path.expanduser(tx_annotation_file))[0], sep=sep)
#         tx_annotation = tx_annotation.drop_duplicates(subset=[tx_id_col])
#         tx_annotation[tx_annotation_col] = tx_annotation[tx_annotation_col].astype(str)
#         tx_annotation = check_annotation_ids(tx_annotation, "tx", tx_id_col, tx_annotation_method)

#     elif tx_annotation_method == "jgi":
#         log.info("JGI annotation method not yet implemented!!")
#         return None

#     elif tx_annotation_method == "magi2":
#         tx_id_col = "gene_ID"
#         tx_annotation_col = "rhea_ID_g2r"

#         gff_filename = glob.glob(f"{tx_dir}/*gff3*")
#         if len(gff_filename) == 0:
#             raise ValueError(f"No GFF3 file found in {tx_dir}.")
#         elif len(gff_filename) > 1:
#             raise ValueError(f"Multiple GFF3 files found in the specified directory: {gff_filename}. \nPlease specify one.")
#         elif len(gff_filename) == 1:
#             gff_filename = gff_filename[0]

#         magi_genes = f"{magi2_dir}/{project_name}/output_{project_name}/magi_gene_results.csv"
        
#         tx_annotation = annotate_genes_to_rhea(gff_filename, magi_genes, tx_id_col, tx_annotation_col)
#         tx_annotation = check_annotation_ids(tx_annotation, "tx", tx_id_col, tx_annotation_method)

#     else:
#         raise ValueError("Please select a valid tx_annotation_method: custom, jgi, or magi2.")

#     if mx_annotation_method == "fbmn":
#         mx_id_col = "#Scan#"
#         mx_annotation_col = "Compound_Name"
#         mx_annotation = pd.read_csv(f"{mx_dir}/fbmn_compounds.csv")
#         mx_annotation = mx_annotation.drop_duplicates(subset=[mx_id_col], keep='first')
#         mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)

#     elif mx_annotation_method == "custom":
#         if mx_id_col is None or mx_annotation_col is None:
#             raise ValueError("Please provide mx_id_col and mx_annotation_col for custom annotation.")
#         sep = "\t" if mx_annotation_file.endswith((".txt", ".tsv", ".tab")) else ","
#         mx_annotation = pd.read_csv(glob.glob(os.path.expanduser(mx_annotation_file))[0], sep=sep)
#         mx_annotation = mx_annotation.drop_duplicates(subset=[mx_id_col])
#         mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)
        
#     elif mx_annotation_method == "magi2":
#         mx_id_col = "#Scan#"
#         mx_annotation_col = "rhea_ID_r2g"
#         magi_compounds = pd.read_csv(f"{magi2_dir}/{project_name}/output_{project_name}/magi_compound_results.csv")
#         fbmn_compounds = pd.read_csv(f"{mx_dir}/fbmn_compounds.csv")
#         mx_annotation = fbmn_compounds.merge(magi_compounds[['original_compound', mx_annotation_col, 'MAGI_score']],on='original_compound',how='left')
#         mx_annotation = mx_annotation.sort_values(by='MAGI_score', ascending=False).drop_duplicates(subset=[mx_id_col], keep='first')
#         mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)

#     else:
#         raise ValueError("Please select a valid mx_annotation_method: fbmn, custom, or magi2.")

#     log.info("Annotating features in integrated data...")
#     tx_annotation = tx_annotation.dropna(subset=[tx_id_col])
#     mx_annotation = mx_annotation.dropna(subset=[mx_id_col])
#     tx_annotation[tx_annotation_col] = tx_annotation[tx_annotation_col].replace("<NA>","Unassigned").fillna("Unassigned")
#     mx_annotation[mx_annotation_col] = mx_annotation[mx_annotation_col].replace("<NA>","Unassigned").fillna("Unassigned")

#     annotated_data = integrated_data[[]]
#     annotated_data['tx_annotation'] = integrated_data.index.map(tx_annotation[tx_annotation_col])
#     annotated_data['mx_annotation'] = integrated_data.index.map(mx_annotation[mx_annotation_col])
    
#     annotated_data['annotation'] = annotated_data['tx_annotation'].combine_first(annotated_data['mx_annotation'])
    
#     result = annotated_data[['annotation']]
#     log.info("Writing annotated data to file...")
#     write_integration_file(data=result, output_dir=output_dir, filename=annotated_data_filename, indexing=True)
#     return result

# ====================================
# Data linking and integration functions
# ====================================

def link_metadata_with_custom_script(
    datasets: list,
    custom_script_path: str,
) -> dict:
    """
    Use external custom script to linked metadata tables across datasets.

    Args:
        datasets (list): List of dataset objects with dataset_info attribute.
        custom_script_path (str): Path to custom metadata linking script.

    Returns:
        dict: Dictionary mapping dataset names to linked metadata DataFrames.
    """
    
    # Load and execute custom script
    spec = importlib.util.spec_from_file_location("custom_link", custom_script_path)
    custom_link = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_link)

    # Setup dataset info for external function
    dataset_info = {
        ds.dataset_name: {"outdir": ds.output_dir, "apid": getattr(ds, "apid", None), "raw": ds._raw_metadata_filename}
        for ds in datasets
    }
    
    # Call the custom linked function
    linked_metadata = custom_link.link_metadata_tables(dataset_info)
    for dataset_name, linked_metadata_df in linked_metadata.items():
        if linked_metadata_df.empty or linked_metadata_df is None:
            raise ValueError(f"Custom linking script did not return valid linked metadata for {dataset_name}.")
        if linked_metadata_df.shape[0] < 2:
            raise ValueError(f"Custom linking script returned insufficient samples for {dataset_name}.")

    # Save results if output directory provided
    for ds in datasets:
        log.info(f"Saving linked metadata for {ds.dataset_name}...")
        write_integration_file(linked_metadata[ds.dataset_name], ds.output_dir, ds._linked_metadata_filename, indexing=True)

    return linked_metadata

def _data_colnames_to_replace(metadata, data):
    """Find the metadata column that matches data column names."""
    data_columns = data.columns.tolist()
    for column in metadata.columns:
        metadata_columns = set(metadata[column])
        if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
            return column
    return None

def link_data_across_datasets(
    datasets: list,
    overlap_only: bool = True
) -> dict:
    """
    Integrate multiple omics datasets by matching sample names using metadata mapping.

    Args:
        datasets (list): List of dataset objects with linked_metadata and raw_data attributes.
        overlap_only (bool): If True, restrict to overlapping samples.

    Returns:
        dict: Dictionary mapping dataset names to integrated data DataFrames.
    """

    unified_data = {}
    sample_sets = {}
    # Process each dataset
    for ds in datasets:
        log.info(f"Processing {ds.dataset_name} metadata and data...")

        # Find the column that maps data columns to unified sample names
        unifying_col = 'unique_group'
        sample_col = _data_colnames_to_replace(ds.linked_metadata, ds.raw_data)
        if sample_col is None:
            raise ValueError(f"Could not find matching column between metadata and data for {ds.dataset_name}")
        
        # Get library names and create data subset
        library_names = ds.linked_metadata[sample_col].tolist()
        data_subset = ds.raw_data[[ds.raw_data.columns[0]] + [col for col in ds.raw_data.columns if col in library_names]].copy()
        
        # Create mapping from library names to unified group names
        mapping = dict(zip(ds.linked_metadata[sample_col], ds.linked_metadata[unifying_col]))
        data_subset.columns = [data_subset.columns[0]] + [mapping.get(col, col) for col in data_subset.columns[1:]]
        
        # Handle duplicate columns (occurs in metabolomics datasets with multiple polarities)
        if data_subset.columns.duplicated().any():
            data_subset = data_subset.T.groupby(data_subset.columns).sum().T
        
        unified_data[ds.dataset_name] = data_subset
        sample_sets[ds.dataset_name] = set(data_subset.columns)

    # Restrict to overlapping samples if requested
    if overlap_only and len(unified_data) > 1:
        log.info("\tRestricting matching samples to only those present in all datasets...")
        overlapping_columns = set.intersection(*sample_sets.values())
        
        if not overlapping_columns:
            raise ValueError("No overlapping samples found across datasets.")
        
        # Preserve the first column (feature IDs) and overlapping sample columns
        first_cols = ["GeneID", "CompoundID"]  # Potential feature ID column names
        overlapping_columns = first_cols + list(overlapping_columns)
        
        for name, data_subset in unified_data.items():
            cols = [col for col in overlapping_columns if col in data_subset.columns]
            unified_data[name] = data_subset[cols]

    # Save results if output directory provided
    for ds in datasets:
        log.info(f"Saving linked data for {ds.dataset_name}...")
        df = unified_data[ds.dataset_name]
        df = df.set_index(df.columns[0])
        write_integration_file(df, ds.output_dir, ds._linked_data_filename, indexing=True)
        unified_data[ds.dataset_name] = df

    return unified_data


def integrate_metadata(
    datasets: list,
    metadata_vars: List[str] = [],
    unifying_col: str = 'unique_group',
    output_filename: str = "integrated_metadata",
    output_dir: str = None
) -> pd.DataFrame:
    """
    Integrate multiple metadata tables into a single DataFrame using a unifying column.

    Args:
        datasets (list): List of dataset objects with linked_metadata attributes.
        metadata_vars (list of str): Metadata columns to include.
        unifying_col (str): Column name to join on.
        output_dir (str, optional): Output directory to save integrated metadata.

    Returns:
        pd.DataFrame: Integrated metadata DataFrame.
    """

    log.info("Creating a single integrated (shared) metadata table across datasets...")

    metadata_tables = [ds.linked_metadata for ds in datasets if hasattr(ds, "linked_metadata")]

    subset_cols = metadata_vars + [unifying_col]
    integrated_metadata = metadata_tables[0][subset_cols].copy()

    # Merge all subsequent tables
    for i, table in enumerate(metadata_tables[1:], start=2):
        integrated_metadata = integrated_metadata.merge(
            table[subset_cols],
            on=unifying_col,
            suffixes=(None, f'_{i}'),
            how='outer'
        )
        # Collapse columns if they match
        for column in metadata_vars:
            col1 = column
            col2 = f"{column}_{i}"
            if col2 in integrated_metadata.columns:
                integrated_metadata[column] = integrated_metadata[col1].combine_first(integrated_metadata[col2])
                integrated_metadata.drop(columns=[col2], inplace=True)

    integrated_metadata.rename(columns={unifying_col: 'sample'}, inplace=True)
    integrated_metadata.sort_values('sample', inplace=True)
    integrated_metadata.drop_duplicates(inplace=True)
    integrated_metadata.set_index('sample', inplace=True)

    log.info("Writing integrated metadata table...")
    write_integration_file(data=integrated_metadata, output_dir=output_dir, filename=output_filename)
    
    return integrated_metadata

def integrate_data(
    datasets: list,
    overlap_only: bool = True,
    output_filename: str = "integrated_data",
    output_dir: str = None
) -> pd.DataFrame:
    """
    Integrate multiple normalized datasets into a single feature matrix by concatenating features.

    Args:
        datasets (list): List of dataset objects with normalized_data attributes.
        overlap_only (bool): If True, restrict to overlapping samples across datasets.
        output_dir (str, optional): Output directory to save integrated data.

    Returns:
        pd.DataFrame: Integrated feature matrix with all datasets combined.
    """
    
    log.info("Creating a single integrated feature matrix across datasets...")
    
    # Collect normalized data from all datasets
    dataset_data = {}
    sample_sets = {}
    
    for ds in datasets:
        if hasattr(ds, 'normalized_data') and not ds.normalized_data.empty:
            # Add dataset prefix to feature names to avoid conflicts
            data_copy = ds.normalized_data.copy()
            if not data_copy.index.str.startswith(f"{ds.dataset_name}_").any():
                data_copy.index = [f"{ds.dataset_name}_{idx}" for idx in data_copy.index]
            
            dataset_data[ds.dataset_name] = data_copy
            sample_sets[ds.dataset_name] = set(data_copy.columns)
            log.info(f"\tAdding {data_copy.shape[0]} features from {ds.dataset_name}")
        else:
            raise ValueError(f"Dataset {ds.dataset_name} missing normalized_data. Run processing pipeline first.")
    
    # Handle overlapping samples if requested
    if overlap_only and len(dataset_data) > 1:
        log.info("\tRestricting to overlapping samples across all datasets...")
        overlapping_samples = set.intersection(*sample_sets.values())
        
        if not overlapping_samples:
            raise ValueError("No overlapping samples found across datasets.")
        
        # Filter each dataset to only include overlapping samples
        for ds_name, data in dataset_data.items():
            overlapping_cols = [col for col in overlapping_samples if col in data.columns]
            dataset_data[ds_name] = data[overlapping_cols]
            log.info(f"\t{ds_name}: {len(overlapping_cols)} overlapping samples")
    
    # Combine all datasets vertically (concatenate features)
    integrated_data = pd.concat(dataset_data.values(), axis=0)
    integrated_data.index.name = 'features'
    integrated_data = integrated_data.fillna(0)
    
    log.info(f"Final integrated dataset: {integrated_data.shape[0]} features x {integrated_data.shape[1]} samples")
    
    # Save result if output directory provided
    if output_dir:
        log.info("Writing integrated data table...")
        write_integration_file(integrated_data, output_dir, output_filename, indexing=True)
    
    return integrated_data

# ====================================
# Data processing functions
# ====================================

def remove_uniform_percent_low_variance_features(data: pd.DataFrame, filter_percent: float) -> pd.DataFrame | None:
    """
    Remove a uniform percentage of features with the lowest variance.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        filter_percent (float): Percentage of features to remove.

    Returns:
        pd.DataFrame or None: Filtered feature matrix, or None if variance is too low.
    """

    # Calculate variance for each row
    row_variances = data.var(axis=1)
    
    # Check if the variance of variances is very low
    if np.var(row_variances) < 0.001:
        log.info("Low variance detected. Is data already autoscaled?")
        return None
    
    # Determine the variance threshold
    var_thresh = np.quantile(row_variances, filter_percent / 100)
    
    # Filter rows with variance below the threshold
    feat_keep = row_variances >= var_thresh
    filtered_df = data[feat_keep]

    log.info(f"Started with {data.shape[0]} features; filtered out {filter_percent}% ({data.shape[0] - filtered_df.shape[0]}) to keep {filtered_df.shape[0]}.")

    return filtered_df


def filter_data(
    data: pd.DataFrame,
    dataset_name: str,
    data_type: str,
    output_dir: str,
    output_filename: str,
    filter_method: str,
    filter_value: float,
) -> pd.DataFrame:
    """
    Filter features from a dataset based on a minimum average obs value or proportion of samples with feature.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        dataset_name (str): Name of the dataset (used for output).
        data_type (str): Data type ('counts', 'abundance', etc.).
        output_dir (str): Output directory.
        filter_method (str): Filtering method ('minimum', 'proportion', or 'none').
        filter_value (float): Threshold value for filtering.

    Returns:
        pd.DataFrame: Filtered feature matrix.
    """

    # Filter data based on the specified method
    if filter_method == "minimum":
        row_means = data.mean(axis=1)
        log.info(f"Filtering out features with {filter_method} method in {dataset_name} that have an average {data_type} value less than {filter_value} across samples...")
        filtered_data = data[row_means >= filter_value]
        log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
    elif filter_method == "proportion":
        log.info(f"Filtering out features with {filter_method} method in {dataset_name} that were observed in fewer than {filter_value}% samples...")
        global_min = data.values.min()
        min_count = (data <= global_min).sum(axis=1)
        min_proportion = min_count / data.shape[1]
        filtered_data = data[min_proportion < filter_value / 100]
        log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
    elif filter_method == "none":
        log.info("Not filtering any features.")
        log.info(f"Keeping all {data.shape[0]} features.")
        filtered_data = data
    else:
        log.info(f"Invalid filter method '{filter_method}'. Please choose 'minimum', 'proportion', or 'none'.")
        return None

    log.info(f"Saving filtered data for {dataset_name}...")
    write_integration_file(filtered_data, output_dir, output_filename, indexing=True)
    return filtered_data

def devariance_data(
    data: pd.DataFrame,
    filter_value: float,
    dataset_name: str,
    output_filename: str,
    output_dir: str,
    devariance_mode: str = "none",
) -> pd.DataFrame | None:
    """
    Remove low-variance features from a dataset using various strategies.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        filter_value (float): Uniform percent of features to remove.
        dataset_name (str): Name for output file.
        output_dir (str): Output directory.
        devariance_mode (str): Devariance method ('percent', 'none').

    Returns:
        pd.DataFrame or None: Filtered feature matrix, or None if variance is too low.
    """

    if devariance_mode == "percent":
        log.info(f"Removing {filter_value}% of features with the lowest variance in {dataset_name}...")
        data_filtered = remove_uniform_percent_low_variance_features(data=data, filter_percent=filter_value)
    elif devariance_mode == "none":
        log.info(f"\tNot removing any features based on variance in {dataset_name}. Retaining all {data.shape[0]} features.")
        log.info(f"Saving devarianced data for {dataset_name}...")
        data_filtered = data
    else:
        log.info(f"Invalid devariance mode '{devariance_mode}'. Please choose 'percent' or 'none'.")
        return None
    
    log.info(f"Saving devarianced data for {dataset_name}...")
    write_integration_file(data_filtered, output_dir, output_filename, indexing=True)
    return data_filtered

def scale_data(
    df: pd.DataFrame,
    output_filename: str = None,
    output_dir: str = None,
    dataset_name: str = None,
    log2: bool = True,
    norm_method: str = "modified_zscore"
) -> pd.DataFrame:
    """
    Normalize and scale feature matrix with multiple methods.
    
    New norm_methods:
    - 'quantile': Force identical distributions across all features
    - 'rank_normal': Rank-based inverse normal transformation
    - 'vst': Variance stabilizing transformation + z-score
    - 'vsn': Variance Stabilizing Normalization
    """
    
    if norm_method == "none":
        log.info("Not scaling data.")
        return df
    
    # Convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Quantile normalization (forces identical distribution)
    if norm_method == "quantile":
        log.info(f"Applying quantile normalization to {dataset_name}...")
        
        # Get reference distribution (sorted values of all data)
        reference = np.sort(df.values.flatten())
        
        # Apply to each column
        scaled_values = np.zeros_like(df.values)
        for j in range(df.shape[1]):
            ranks = rankdata(df.iloc[:, j], method='average')
            for i in range(df.shape[0]):
                idx = int(ranks[i]) - 1
                scaled_values[i, j] = reference[idx]
        
        scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    
    # Rank-based inverse normal transformation
    elif norm_method == "rank_normal":
        log.info(f"Applying rank-based inverse normal transformation to {dataset_name}...")
        
        def transform_col(col):
            ranks = rankdata(col, method='average')
            quantiles = ranks / (len(ranks) + 1)
            return norm.ppf(quantiles)
        
        scaled_df = df.apply(transform_col, axis=0)
    
    # VST + z-score (good for count data)
    elif norm_method == "vst":
        log.info(f"Applying VST transformation to {dataset_name}...")
        vst_df = np.arcsinh(np.sqrt(df))
        scaled_df = vst_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    # VSN - Variance Stabilizing Normalization
    elif norm_method == "vsn":
        log.info(f"Applying VSN transformation to {dataset_name}...")
        
        # Convert to numpy array
        X = df.values.astype(float)
        
        # VSN transformation function
        def _vsn_transform(x: np.ndarray, lam: float) -> np.ndarray:
            """Element-wise VSN transform: g_λ(x) = log2[(x + sqrt(x² + λ)) / 2]"""
            x = np.where(x < 0, 0.0, x)
            return np.log2((x + np.sqrt(x * x + lam)) / 2.0)
        
        # Estimate lambda by minimizing variance of transformed data
        def _objective(lam_candidate: float) -> float:
            if lam_candidate <= 0:
                return np.inf
            Y = _vsn_transform(X, lam_candidate)
            return np.nanvar(Y)
        
        # Use scipy's bounded minimizer to find optimal lambda
        res = minimize_scalar(
            _objective,
            bounds=(1e-6, 1e6),
            method='bounded',
            options={'xatol': 1e-8}
        )
        
        if not res.success:
            raise RuntimeError(
                f"λ estimation failed for VSN: {res.message}. "
                "Try a different normalization method."
            )
        
        lam = res.x
        log.info(f"  Estimated λ = {lam:.6f} for VSN transformation")
        
        # Apply transformation with estimated lambda
        Y = _vsn_transform(X, lam)
        scaled_df = pd.DataFrame(Y, index=df.index, columns=df.columns)
    
    elif norm_method in ["zscore", "modified_zscore", "logfc_mean", "logfc_median", "logfc_geometric_mean"]:
        if log2 and norm_method not in ["logfc_mean", "logfc_median", "logfc_geometric_mean"]:
            df = np.log2(df + 1)
        if norm_method == "zscore":
            scaled_df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        elif norm_method == "modified_zscore":
            scaled_df = df.apply(lambda x: ((x - x.median()) * 0.6745) / (x - x.median()).abs().median(), axis=0)
        elif norm_method == "logfc_mean":
            log.info(f"Scaling {dataset_name} data using log2 fold-change relative to mean...")
            df_pseudocount = df + 1
            log_values = np.log2(df_pseudocount)
            mean_log = log_values.mean(axis=1, skipna=True)
            scaled_df = log_values.subtract(mean_log, axis=0)
        elif norm_method == "logfc_median":
            log.info(f"Scaling {dataset_name} data using log2 fold-change relative to median...")
            df_pseudocount = df + 1
            log_values = np.log2(df_pseudocount)
            median_log = log_values.median(axis=1, skipna=True)
            scaled_df = log_values.subtract(median_log, axis=0)
        elif norm_method == "logfc_geometric_mean":
            log.info(f"Scaling {dataset_name} data using log2 fold-change relative to geometric mean...")
            df_pseudocount = df + 1
            geometric_means = df_pseudocount.apply(lambda row: gmean(row.dropna()), axis=1)
            scaled_df = np.log2(df_pseudocount.divide(geometric_means, axis=0))
        else:
            raise ValueError("Please select a valid norm_method: 'zscore', 'modified_zscore', 'logfc_mean', 'logfc_median', or 'logfc_geometric_mean'.")
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")

    # Ensure output is float, and replace NA/inf/-inf with 0
    scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    if output_dir:
        log.info(f"Saving scaled data for {dataset_name}...")
        write_integration_file(scaled_df, output_dir, output_filename, indexing=True)

    return scaled_df

def remove_low_replicable_features(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_name: str,
    output_filename: str,
    output_dir: str,
    method: str = "variance",
    group_col: str = "group",
    threshold: float = 0.6,
):
    """
    Remove features (rows) from `data` with high within-group variability.
    Highly optimized version using vectorized pandas operations.
    """

    if method == "none":
        log.info(f"\tNot removing any features based on replicability in {dataset_name}. Retaining all {data.shape[0]} features.")
        replicable_data = data
    elif method == "variance":
        if group_col not in metadata.columns:
            raise ValueError(f"Column '{group_col}' not found in metadata.")
        
        log.info(f"Removing features with high within-group variability (threshold: {threshold})...")
        
        # Get groups and prepare sample mapping
        groups = metadata[group_col].unique()
        group_sample_map = {
            group: [s for s in metadata[metadata[group_col] == group]['unique_group'].tolist() 
                   if s in data.columns]
            for group in groups
        }
        
        # Filter to groups with at least 2 samples
        valid_groups = {k: v for k, v in group_sample_map.items() if len(v) >= 2}
        
        if not valid_groups:
            log.info("No groups with sufficient replicates (>=2) found. Keeping all features.")
            replicable_data = data
        else:
            # Vectorized approach: calculate all group variances at once
            keep_mask = pd.Series(True, index=data.index)
            
            for group, samples in valid_groups.items():
                # Calculate standard deviation for all features in this group
                group_variances = data[samples].std(axis=1, skipna=True)
                
                # Mark features that exceed threshold in this group for removal
                high_var_in_group = group_variances > threshold
                keep_mask = keep_mask & (~high_var_in_group)
            
            replicable_data = data.loc[keep_mask]
        
        log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - replicable_data.shape[0]} to keep {replicable_data.shape[0]}.")
    
    else:
        raise ValueError("Currently only 'variance' or 'none' methods are supported for removing low replicable features.")

    log.info(f"Saving replicable data for {dataset_name}...")
    write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
    return replicable_data

# ====================================
# Plotting functions
# ====================================

def _draw_pca(ax, pca_df: pd.DataFrame, hue_col: str,
              title: str, alpha: float = 0.75) -> None:
    """Add KDE + scatter to *ax* for the supplied PCA DataFrame."""
    sns.kdeplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue=hue_col,
        fill=True,
        alpha=alpha,
        palette="viridis",
        bw_adjust=2,
        ax=ax,
        warn_singular=False
    )
    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue=hue_col,
        palette="viridis",
        alpha=alpha,
        s=50,
        edgecolor="w",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_title(title, fontsize=10)
    ax.legend(title=hue_col, loc="best", fontsize=8)

def plot_simple_pca(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_variables: List[str],
    title: str = "PCA Plot"):
    """
    Plot a simple PCA for a single dataset and metadata variable.
    """
    df_samples = set(df.columns)
    if 'unique_group' not in metadata.columns:
        metadata['unique_group'] = metadata.index
    meta_samples = set(metadata["unique_group"])
    common = list(df_samples & meta_samples)

    # build PCA matrix
    X = (
        df.T.loc[common]
        .replace([np.inf, -np.inf], np.nan)
        .infer_objects(copy=False)
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    pca_df = pd.DataFrame(pcs, columns=["PCA1", "PCA2"], index=X.index)
    pca_df = pca_df.reset_index().rename(columns={"index": "unique_group"})

    # attach metadata columns (all of them - KD-plot can use any)
    pca_df = pca_df.merge(metadata, on="unique_group", how="left")

    # individual PDF plots
    for meta_var in metadata_variables:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_title = f"{title} - {meta_var}"
        _draw_pca(ax, pca_df, hue_col=meta_var, title=title, alpha=0.75)
        display(fig)
        plt.close(fig)

def plot_pca(
    data: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    metadata_variables: List[str],
    output_dir: str = None,
    output_filename: str = None,
    dataset_name: str = None,
    alpha: float = 0.75,
    show_plot: bool = False,
) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, pd.DataFrame]]:
    """
    For each DataFrame in *data* (e.g. “linked”, “normalized”)
    compute a 2-component PCA on the intersecting samples,
    draw a seaborn plot for every *metadata_variable*, and
    store the figure as a PDF.

    Returns
    -------
    plot_paths : dict[data_type][metadata_variable] → Path to PDF
    pca_frames : dict[data_type] → PCA-augmented DataFrame (used later for the grid)
    """

    plot_paths: Dict[str, Dict[str, Path]] = {}
    pca_frames: Dict[str, pd.DataFrame] = {}

    for d_type, df in data.items():
        # match samples
        df_samples = set(df.columns)
        meta_samples = set(metadata["unique_group"])
        common = list(df_samples & meta_samples)

        if not common:
            log.warning(f"No common samples for {d_type}; skipping.")
            continue

        # build PCA matrix
        X = (
            df.T.loc[common]
            .replace([np.inf, -np.inf], np.nan)
            .infer_objects(copy=False)
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
        )

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        pca_df = pd.DataFrame(pcs, columns=["PCA1", "PCA2"], index=X.index)
        pca_df = pca_df.reset_index().rename(columns={"index": "unique_group"})

        # attach metadata columns (all of them - KD-plot can use any)
        pca_df = pca_df.merge(metadata, on="unique_group", how="left")
        pca_frames[d_type] = pca_df

        # individual PDF plots
        plot_paths[d_type] = {}
        for meta_var in metadata_variables:
            fig, ax = plt.subplots(figsize=(6, 5))
            title = f"{dataset_name} - {d_type} - {meta_var}"
            _draw_pca(ax, pca_df, hue_col=meta_var, title=title, alpha=alpha)

            pdf_path = f"{output_dir}/pca_of_{dataset_name}_{d_type}_by_{meta_var}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)

            plot_paths[d_type][meta_var] = pdf_path

    plot_pdf_grids(
        pca_frames=pca_frames,
        metadata_variables=metadata_variables,
        output_dir=output_dir,
        output_filename=output_filename,
        alpha=alpha,
        show_plot=show_plot,
    )

    return

def plot_pdf_grids(
    pca_frames: Dict[str, pd.DataFrame],
    metadata_variables: List[str],
    output_dir: str,
    output_filename: str = None,
    alpha: float = 0.75,
    show_plot: bool = True,
) -> Path:
    """
    Build a Matplotlib grid (rows = data types, columns = metadata variables)
    from the PCA DataFrames created by :func:`plot_pca`.

    If *show_plot* is True (default) the figure is displayed inline (Jupyter).
    Returns the path to the grid PDF.
    """
    pdf_metadata_vars = metadata_variables.copy()
    data_types = list(pca_frames.keys())
    n_rows, n_cols = len(data_types), len(pdf_metadata_vars)

    # create a figure with enough space for all sub-plots
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 4, n_rows * 3.5),
        constrained_layout=True,
    )
    # ensure a 2-D array even if n_rows == 1 or n_cols == 1
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, d_type in enumerate(data_types):
        pca_df = pca_frames[d_type]
        for j, meta_var in enumerate(pdf_metadata_vars):
            ax = axes[i, j]
            if d_type == "linked":
                d_type = "non-normalized"
            title = f"{d_type} - {meta_var}"
            _draw_pca(ax, pca_df, hue_col=meta_var, title=title, alpha=alpha)

    grid_pdf = f"{output_dir}/{output_filename}"
    fig.savefig(grid_pdf, format="pdf", bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

    return

def plot_data_variance_indv_histogram(
    data: pd.DataFrame,
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
    dataset_name: str = None,
    output_dir: str = None
) -> None:
    """
    Plot a histogram of all values in a single dataset.

    Args:
        data (pd.DataFrame): Feature matrix.
        bins (int): Number of histogram bins.
        transparency (float): Alpha for bars.
        xlog (bool): Use log scale for x-axis.
        dataset_name (str, optional): Name for output file.
        output_dir (str, optional): Output directory for plots.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    sns.histplot(data.values.flatten(), bins=bins, kde=False, color=sns.color_palette("viridis", 1)[0], 
                    edgecolor="black", fill=True, alpha=transparency)
    
    if xlog:
        plt.xscale('log')
    plt.title(f'Histogram of {dataset_name}')
    plt.xlabel('Quantitative value')
    plt.ylabel('Frequency')
    
    # Save the plot if output_dir is specified
    if output_dir:
        output_subdir = f"{output_dir}/data_distributions"
        os.makedirs(output_subdir, exist_ok=True)
        filename = f"distribution_of_{dataset_name}.pdf"
        log.info(f"Saving plot to {output_subdir}/{filename}")
        plt.savefig(f"{output_subdir}/{filename}")
    
    plt.show()


def plot_simple_histogram(
    dataframe: pd.DataFrame,
    title: str,
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
) -> None:

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", 1)

    sns.histplot(
        dataframe.values.flatten(),
        bins=bins,
        kde=False,
        color=palette[0],
        element="step",
        edgecolor="black",
        fill=True,
        alpha=transparency
    )

    if xlog:
        plt.xscale('log')
    plt.title(title)
    plt.xlabel('Quantitative value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.close()


def plot_data_variance_histogram(
    dataframes: dict[str, pd.DataFrame],
    datatype: str,
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
    output_dir: str = None,
    show_plot: bool = True
) -> None:
    """
    Plot histograms of values for multiple datasets on the same plot.

    Args:
        dataframes (dict of str: pd.DataFrame): Dictionary mapping labels to feature matrices.
        datatype (str): Type of data being plotted.
        bins (int): Number of histogram bins.
        transparency (float): Alpha for bars.
        xlog (bool): Use log scale for x-axis.
        output_dir (str, optional): Output directory for plots.
        show_plot (bool): Whether to display the plot.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", len(dataframes))

    for i, (label, df) in enumerate(dataframes.items()):
        sns.histplot(
            df.values.flatten(),
            bins=bins,
            kde=False,
            color=palette[i],
            label=label,
            element="step",
            edgecolor="black",
            fill=True,
            alpha=transparency
        )

    if xlog:
        plt.xscale('log')
    plt.title('Histogram of DataFrames')
    plt.xlabel('Quantitative value')
    plt.ylabel('Frequency')
    plt.title('Data structure')
    plt.legend()

    # Save the plot if output_dir is specified
    filename = f"distribution_of_{datatype}_datasets.pdf"
    log.info(f"Saving plot to {output_dir}/{filename}...")
    plt.savefig(f"{output_dir}/{filename}")
    if show_plot is True:
        log.info("Datasets should follow similar distributions, while quantitative values can be slightly shifted:")
        plt.show()
    plt.close()

def plot_feature_abundance_by_metadata(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    feature: str,
    metadata_group: str | List[str],
    output_dir: str = None,
    save_plot: bool = False
) -> None:
    """
    Plot the abundance of a single feature across metadata groups.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        metadata (pd.DataFrame): Metadata DataFrame (samples x variables).
        feature (str): Feature name to plot.
        metadata_group (str or list of str): Metadata variable(s) to group by (can be one or two for color/shape).

    Returns:
        None

    Example:
        plot_feature_abundance_by_metadata(integrated_data, integrated_metadata, "tx_Pavir.8NG007100",  "location")
    """

    # Select the row data
    if feature not in data.index:
        raise ValueError(f"Feature '{feature}' not found in data.")
    row_data = data.loc[feature]
    
    # Merge row data with metadata
    linked_data = pd.merge(row_data.to_frame(name='abundance'), metadata, left_index=True, right_index=True)
    linked_data.sort_values(by=metadata_group if isinstance(metadata_group, str) else metadata_group[0], inplace=True)
    
    # Check if metadata_group is a list
    if isinstance(metadata_group, list) and len(metadata_group) == 2:
        color_group, shape_group = metadata_group
        linked_data['color_shape_group'] = linked_data[color_group].astype(str) + "_" + linked_data[shape_group].astype(str)
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='color_shape_group', y='abundance', data=linked_data, palette='viridis')
        sns.stripplot(x='color_shape_group', y='abundance', data=linked_data, color='k', alpha=0.5, jitter=True)
        plt.xlabel(f'{color_group} and {shape_group}')
        plt.ylabel('Z-scored abundance')
        plt.title(f'Abundance of {feature} by {color_group} and {shape_group}')
        plt.xticks(rotation=90)
        plt.show()
    else:
        # Plot the data
        plt.figure(figsize=(12, 8))
        sns.violinplot(x=metadata_group, y='abundance', data=linked_data, palette='viridis')
        sns.stripplot(x=metadata_group, y='abundance', data=linked_data, color='k', alpha=0.5, jitter=True)
        plt.xlabel(metadata_group)
        plt.ylabel('Z-scored abundance')
        plt.title(f'Abundance of {feature} by {metadata_group}')
        plt.xticks(rotation=90)
        plt.show()

    if save_plot and output_dir:
        output_subdir = f"{output_dir}/boxplots"
        os.makedirs(output_subdir, exist_ok=True)
        filename = f"abundance_of_{feature}_by_{metadata_group}.pdf"
        log.info(f"Saving plot to {output_subdir}/{filename}")
        plt.savefig(f"{output_subdir}/{filename}")

def plot_submodule_abundance_by_metadata(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    node_table: pd.DataFrame,
    submodule_name: str,
    metadata_group: str | List[str],
    output_dir: str = None,
    save_plot: bool = False
) -> None:
    """
    Plot the average abundance of all features in a network submodule across metadata groups.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        metadata (pd.DataFrame): Metadata DataFrame (samples x variables).
        node_table (pd.DataFrame): Network node table with 'submodule' column and feature names as index or in a column.
        submodule_name (str): Name of the submodule to extract features from (matches values in 'submodule' column).
        metadata_group (str or list of str): Metadata variable(s) to group by (can be one or two for color/shape).

    Returns:
        None

    Example:
        # Load network node table and plot a specific submodule
        node_table = pd.read_csv("network_node_table.csv", index_col=0)
        plot_submodule_abundance_by_metadata(
            integrated_data, 
            integrated_metadata, 
            node_table,
            "submodule_1", 
            "location"
        )
    """

    # Check if node_table has required 'submodule' column
    if 'submodule' not in node_table.columns:
        raise ValueError("Node table must contain a 'submodule' column")
    
    # Extract features for the specified submodule
    # Try to get feature names from index first, then from a potential node_id/feature column
    if node_table.index.name in ['node_id', 'feature_id', 'features'] or 'node_id' in str(node_table.index.name).lower():
        # Feature names are in the index
        submodule_mask = node_table['submodule'] == submodule_name
        submodule_features = node_table.index[submodule_mask].tolist()
    elif 'node_id' in node_table.columns:
        # Feature names are in a 'node_id' column
        submodule_mask = node_table['submodule'] == submodule_name
        submodule_features = node_table.loc[submodule_mask, 'node_id'].tolist()
    elif 'feature_id' in node_table.columns:
        # Feature names are in a 'feature_id' column
        submodule_mask = node_table['submodule'] == submodule_name
        submodule_features = node_table.loc[submodule_mask, 'feature_id'].tolist()
    else:
        # Default to using the index
        submodule_mask = node_table['submodule'] == submodule_name
        submodule_features = node_table.index[submodule_mask].tolist()
    
    # Check if submodule exists
    if submodule_name not in node_table['submodule'].values:
        available_submodules = node_table['submodule'].unique().tolist()
        raise ValueError(f"Submodule '{submodule_name}' not found in node table. Available submodules: {available_submodules}")
    
    # Check if any features were found
    if not submodule_features:
        raise ValueError(f"No features found for submodule '{submodule_name}' in node table")
    
    log.info(f"Found {len(submodule_features)} features in submodule '{submodule_name}'")
    
    # Filter features that are present in the data
    available_features = [f for f in submodule_features if f in data.index]
    
    if not available_features:
        raise ValueError(f"None of the submodule features found in data. Available features: {data.index.tolist()[:10]}...")
    
    if len(available_features) < len(submodule_features):
        missing_features = set(submodule_features) - set(available_features)
        log.info(f"Warning: {len(missing_features)} features from submodule not found in data")
    
    # Calculate mean abundance across all features in the submodule for each sample
    submodule_data = data.loc[available_features]
    mean_abundance = submodule_data.mean(axis=0)  # Mean across features for each sample
    
    # Merge mean abundance with metadata
    linked_data = pd.merge(
        mean_abundance.to_frame(name='mean_abundance'), 
        metadata, 
        left_index=True, 
        right_index=True
    )
    linked_data.sort_values(by=metadata_group if isinstance(metadata_group, str) else metadata_group[0], inplace=True)

    # Check if metadata_group is a list for two-variable grouping
    if isinstance(metadata_group, list) and len(metadata_group) == 2:
        color_group, shape_group = metadata_group
        linked_data['color_shape_group'] = (
            linked_data[color_group].astype(str) + "_" + linked_data[shape_group].astype(str)
        )
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(
            x='color_shape_group', 
            y='mean_abundance', 
            data=linked_data, 
            palette='viridis'
        )
        sns.stripplot(
            x='color_shape_group', 
            y='mean_abundance', 
            data=linked_data, 
            color='k', 
            alpha=0.5, 
            jitter=True
        )
        plt.xlabel(f'{color_group} and {shape_group}')
        plt.ylabel('Mean abundance (Z-scored)')
        plt.title(f'Mean abundance of {submodule_name} ({len(available_features)} features) by {color_group} and {shape_group}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        # Single metadata variable
        plt.figure(figsize=(12, 8))
        sns.violinplot(
            x=metadata_group, 
            y='mean_abundance', 
            data=linked_data, 
            palette='viridis'
        )
        sns.stripplot(
            x=metadata_group, 
            y='mean_abundance', 
            data=linked_data, 
            color='k', 
            alpha=0.5, 
            jitter=True
        )
        plt.xlabel(metadata_group)
        plt.ylabel('Mean abundance (Z-scored)')
        plt.title(f'Mean abundance of {submodule_name} ({len(available_features)} features) by {metadata_group}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    if save_plot and output_dir:
        output_subdir = f"{output_dir}/boxplots"
        os.makedirs(output_subdir, exist_ok=True)
        filename = f"avg_abundance_of_{submodule_name}_nodes_by_{metadata_group}.pdf"
        log.info(f"Saving plot to {output_subdir}/{filename}")
        plt.savefig(f"{output_subdir}/{filename}")

def plot_replicate_correlation(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    min_val: float = 0.9,
    max_val: float = 1.0,
    method: str = "spearman",
    output_dir: str = None,
    dataset_name: str = None,
    overwrite: bool = False
) -> None:
    """
    Plot replicate correlation heatmaps for each group in the metadata.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        metadata (pd.DataFrame): Metadata DataFrame.
        min_val (float): Minimum value for colorbar.
        max_val (float): Maximum value for colorbar.
        method (str): Correlation method ('spearman', 'pearson', etc.).
        output_dir (str, optional): Output directory for plots.
        dataset_name (str, optional): Name for output file.
        overwrite (bool): Overwrite existing plot if True.

    Returns:
        None
    """
    
    # Check if plot already exists
    filename = f"heatmap_of_replicate_correlation_for_{dataset_name}.pdf"
    output_subdir = f"{output_dir}/plots"
    os.makedirs(output_subdir, exist_ok=True)
    output_plot = f"{output_subdir}/{filename}"
    if os.path.exists(output_plot) and not overwrite:
        log.info(f"Replicate heatmap plot already exists: {output_plot}. Not overwriting.")
        return
    
    corr_matrix = data.corr(method=method)
    corr_matrix.index.name = 'sample'

    metadata_filtered = metadata[metadata.index.isin(corr_matrix.index)]
    metadata_filtered = metadata_filtered.loc[corr_matrix.index] # Align metadata to correlation matrix before merging

    merged_corr_data = corr_matrix.join(metadata_filtered, how='inner')
    group_column = 'group'
    
    g = sns.FacetGrid(merged_corr_data, col=group_column, col_wrap=3, height=4)
    cbar_ax = g.figure.add_axes([1, .3, .02, .4])
    vmin = min_val
    vmax = max_val

    def plot_heatmap(data, **kwargs):
        group = data[group_column].iloc[0]
        samples_in_group = metadata_filtered[metadata_filtered[group_column] == group].index
        subset_corr_matrix = corr_matrix.loc[samples_in_group, samples_in_group]
        ax = sns.heatmap(subset_corr_matrix, annot=True, cmap='coolwarm', cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_title(group)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    g.map_dataframe(plot_heatmap)
    g.figure.suptitle(f"{dataset_name} sample {method} replicate correlation", y=1.05)

    # Save the plot if output_dir is specified
    if output_dir:
        log.info(f"Saving plot to {output_plot}")
        g.figure.savefig(output_plot)
        plt.close(g.figure)
    else:
        log.info("Not saving plot to disk.")
    
    return

def plot_heatmap_with_dendrogram(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_var: str,
    corr_method: str = "pearson",
    figure_size: tuple = (12, 8),
    output_dir: str = None,
    dataset_name: str = None,
    overwrite: bool = False
) -> None:
    """
    Plot a clustered heatmap with dendrogram colored by a metadata variable.

    Args:
        data (pd.DataFrame): Feature matrix.
        metadata (pd.DataFrame): Metadata DataFrame.
        metadata_var (str): Metadata column to color by.
        corr_method (str): Correlation method.
        figure_size (tuple): Figure size.
        output_dir (str, optional): Output directory for plots.
        dataset_name (str, optional): Name for output file.
        overwrite (bool): Overwrite existing plot if True.

    Returns:
        None
    """

    # Check if plot already exists
    filename = f"heatmap_of_grouped_metadata_for_{dataset_name}.pdf"
    output_subdir = f"{output_dir}/plots"
    os.makedirs(output_subdir, exist_ok=True)
    output_plot = f"{output_subdir}/{filename}"
    if os.path.exists(output_plot) and not overwrite:
        log.info(f"Heatmap plot already exists: {output_plot}. Not overwriting.")
        return

    # Compute the correlation matrix
    corr_matrix = data.corr(method=corr_method)
    corr_matrix.index.name = 'sample'
    
    # Create a color palette for the specified metadata variable using viridis
    unique_values = metadata[metadata_var].unique()
    palette = sns.color_palette("viridis", len(unique_values))
    lut = dict(zip(unique_values, palette))
    col_colors = metadata[metadata_var].map(lut)
    
    # Create a clustermap without annotations and with column colors
    g = sns.clustermap(corr_matrix, method='average', cmap='coolwarm', annot=False, figsize=figure_size, col_colors=col_colors)
    
    # Add metadata labels to the x and y axis
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=7)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
    g.ax_heatmap.set_ylabel('')
    
    # Create a legend for the metadata variable
    for value in unique_values:
        g.ax_col_dendrogram.bar(0, 0, color=lut[value], label=f"{metadata_var}: {value}", linewidth=0)
    
    g.ax_col_dendrogram.legend(loc="center", ncol=3, bbox_to_anchor=(0.5, 1.1), bbox_transform=g.figure.transFigure)
    g.figure.suptitle(f"{corr_method} correlation of {dataset_name} by {metadata_var}", y=1)

    # Save the plot if output_dir is specified
    if output_dir:
        log.info(f"Saving plot to {output_plot}")
        g.savefig(output_plot)
        plt.close(g.figure)
    else:
        log.info("Not saving plot to disk.")

    plt.show()
    plt.close()

def plot_alluvial_for_submodules(
    submodules_dir: str,
    go_annotation_file: str,
    show_plot_in_notebook: bool = False,
    overwrite: bool = False
) -> None:
    """
    Plot and save alluvial (Sankey) diagrams for network submodules using GO annotation.

    Args:
        submodules_dir (str): Directory containing submodule node tables.
        go_annotation_file (str): Path to GO annotation file.
        show_plot_in_notebook (bool): Show plot in notebook if True.
        overwrite (bool): Overwrite existing plots if True.

    Returns:
        None
    """

    if os.path.exists(os.path.join(submodules_dir, "submodule_sankeys")) and overwrite is False:
        log.info(f"Submodule Sankey diagrams already exist in {os.path.join(submodules_dir, 'submodule_sankeys')}. Set overwrite=True to regenerate.")
        return
    elif os.path.exists(os.path.join(submodules_dir, "submodule_sankeys")) and overwrite is True:
        log.info(f"Overwriting existing submodule Sankey diagrams in {os.path.join(submodules_dir, 'submodule_sankeys')}.")
        shutil.rmtree(os.path.join(submodules_dir, "submodule_sankeys"))
        os.makedirs(os.path.join(submodules_dir, "submodule_sankeys"), exist_ok=True)
    elif not os.path.exists(os.path.join(submodules_dir, "submodule_sankeys")):
        os.makedirs(os.path.join(submodules_dir, "submodule_sankeys"), exist_ok=True)

    # Read the GO annotation file
    pathway_map = pd.read_csv(go_annotation_file, sep=None, engine='python')
    pathway_map = pathway_map.loc[pathway_map.groupby('geneID').head(1).index]
    pathway_map_classes = pathway_map[['geneID', 'goName', 'gotermType', 'goAcc']]

    # Loop through all *_node_table.csv files in the submodules directory
    node_table_files = glob.glob(os.path.join(submodules_dir, "*_node_table.csv"))
    for node_table in tqdm(node_table_files, desc="Processing submodules", unit="node file"):
        submodule = pd.read_csv(node_table)
        submodule_annotated = pd.merge(pathway_map_classes, submodule, left_on='geneID', right_on='node_id', how='right')
        submodule_annotated['goName'] = submodule_annotated['goName'].fillna('NA')
        submodule_annotated['goAcc'] = submodule_annotated['goAcc'].fillna('NA')
        submodule_annotated['gotermType'] = submodule_annotated['gotermType'].fillna('Unassigned_Type')
        submodule_annotated['goLabel'] = submodule_annotated['goName'] + ' (' + submodule_annotated['goAcc'] + ')'
        submodule_annotated['goLabel'] = submodule_annotated.groupby('goLabel')['goLabel'].transform(lambda x: x + ' (' + str(x.count()) + ')')

        # Collapse all Unassigned_Type and NA (NA) into a single row
        mask_unassigned = (submodule_annotated['gotermType'] == 'Unassigned_Type') | (submodule_annotated['goLabel'].str.startswith('NA (NA)'))
        n_unassigned = mask_unassigned.sum()
        # Remove all unassigned rows
        submodule_annotated_collapsed = submodule_annotated[~mask_unassigned].copy()
        # Add a single row for all unassigned, if any exist
        if n_unassigned > 0:
            collapsed_row = {
                'gotermType': 'Unassigned',
                'goLabel': f'Unassigned ({n_unassigned})'
            }
            # Add other columns as needed, or fill with NA
            for col in submodule_annotated_collapsed.columns:
                if col not in collapsed_row:
                    collapsed_row[col] = pd.NA
            submodule_annotated_collapsed = pd.concat([
                submodule_annotated_collapsed,
                pd.DataFrame([collapsed_row])
            ], ignore_index=True)

        submodule_annotated = submodule_annotated_collapsed

        order = submodule_annotated['goLabel'].value_counts().index
        submodule_annotated['sorter'] = submodule_annotated['goLabel'].map({k: i for i, k in enumerate(order)})
        submodule_annotated = submodule_annotated.sort_values('sorter').drop('sorter', axis=1)
        if submodule_annotated.empty:
            continue

        stages = ['gotermType', 'goLabel']
        unique_labels = pd.concat([submodule_annotated[stage] for stage in stages]).unique()
        n_labels = len(unique_labels)
        min_height = 200  # Minimum height for readability
        height_per_label = 20
        height = max(min_height, n_labels * height_per_label)
        width = 1000

        labels = pd.concat([submodule_annotated[stage] for stage in stages]).unique().tolist()
        label_to_index = {label: i for i, label in enumerate(labels)}
        sources = submodule_annotated['gotermType'].map(label_to_index)
        targets = submodule_annotated['goLabel'].map(label_to_index)
        values = [1] * len(submodule_annotated)

        goterm_counts = submodule_annotated.groupby('gotermType')['goLabel'].nunique().to_dict()
        updated_labels = []
        for label in labels:
            if label == 'Unassigned':
                updated_labels.append(f"Unassigned ({n_unassigned})")
            elif label in goterm_counts:
                updated_labels.append(f"{label} ({goterm_counts[label]})")
            else:
                updated_labels.append(label)

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=updated_labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            )
        ))

        submodule_name = os.path.splitext(os.path.basename(node_table))[0].replace("_network_node_table", "")
        fig.update_layout(
            title_text=f"Alluvial Diagram for {submodule_name}",
            font_size=10,
            width=width,
            height=height,
            showlegend=False
        )
        if show_plot_in_notebook:
            fig.show()
            
        output_pdf = os.path.join(submodules_dir, "submodule_sankeys", f"{submodule_name}_sankey_diagram.pdf")
        fig.write_image(output_pdf, width=width, height=height)

# ====================================
# Functional Enrichment
# ====================================

def perform_functional_enrichment(
    node_table: pd.DataFrame,
    annotation_column: str,
    p_value_threshold: float = 0.05,
    correction_method : str = "fdr_bh",
    min_annotation_count: int = 1,
    output_dir: str = None,
    output_filename: str = "submodule_enrichment_results"
) -> pd.DataFrame:
    """
    Perform functional enrichment analysis of annotation terms in network submodules 
    using Fisher's exact test, handling multiple annotations per node stored as 
    double semicolon-separated strings.
    """
    
    # Input validation
    if annotation_column not in node_table.columns:
        raise ValueError(f"Annotation column '{annotation_column}' not found in node table")
    
    if 'submodule' not in node_table.columns:
        raise ValueError("Node table must contain 'submodule' column")
    
    log.info(f"Performing enrichment analysis on annotation column: '{annotation_column}'")
    
    # Flatten annotations from semicolon-separated strings to individual rows
    annotation_rows = []
    for idx, row in node_table.iterrows():
        node_id = row.name if row.name is not None else idx
        submodule = row['submodule']
        annotations_str = row[annotation_column]
        
        # Handle case where annotations are stored as semicolon-separated strings
        if pd.isna(annotations_str) or annotations_str in ['Unassigned', 'NA', '']:
            annotation_list = []
        else:
            # Split on double semicolons and clean up
            annotation_list = [ann.strip() for ann in str(annotations_str).split(';;') 
                             if ann.strip() and ann.strip() not in ['Unassigned', 'NA', '']]
        
        # Create a row for each annotation
        if annotation_list:
            for annotation in annotation_list:
                annotation_rows.append({
                    'node_id': node_id,
                    'submodule': submodule,
                    'annotation': annotation
                })
        else:
            # Keep track of nodes without annotations for background calculation
            annotation_rows.append({
                'node_id': node_id,
                'submodule': submodule,
                'annotation': None
            })
    
    if not annotation_rows:
        log.warning("No annotation data found for enrichment analysis")
        return pd.DataFrame()
    
    # Convert to DataFrame for easier manipulation
    flat_annotations = pd.DataFrame(annotation_rows)
    
    # Remove rows with no annotations for the enrichment test
    valid_annotations = flat_annotations[flat_annotations['annotation'].notna()].copy()
    
    if valid_annotations.empty:
        log.warning("No valid annotations found after filtering")
        return pd.DataFrame()
    
    # Filter annotations by minimum count AND require > 1 total occurrence
    annotation_counts = valid_annotations['annotation'].value_counts()
    valid_annotation_terms = annotation_counts[
        (annotation_counts >= min_annotation_count) & (annotation_counts > 1)
    ].index.tolist()
    
    if not valid_annotation_terms:
        log.warning(f"No annotations meet minimum count threshold of {min_annotation_count} and >1 total occurrence")
        return pd.DataFrame()
    
    valid_annotations = valid_annotations[valid_annotations['annotation'].isin(valid_annotation_terms)]
    
    log.info(f"Testing {len(valid_annotation_terms)} unique annotation terms")
    log.info(f"Testing {len(valid_annotations['submodule'].unique())} submodules")
    
    # Get background: all nodes that have at least one valid annotation
    background_nodes = set(valid_annotations['node_id'].unique())
    total_nodes_in_network_with_annotations = len(background_nodes)
    
    # Also calculate total nodes in network (for reference)
    total_nodes_in_network = len(node_table)
    
    # Store results
    results = []
    
    # For each annotation term and each submodule, perform Fisher's exact test
    for annotation in valid_annotation_terms:
        # Get all nodes with this annotation (across all submodules)
        nodes_with_annotation = set(
            valid_annotations[valid_annotations['annotation'] == annotation]['node_id'].unique()
        )
        nodes_with_annotation_total = len(nodes_with_annotation)
        
        for submodule in valid_annotations['submodule'].unique():
            # Get all nodes in this submodule (that have any valid annotation)
            nodes_in_submodule_with_annotations = set(
                valid_annotations[valid_annotations['submodule'] == submodule]['node_id'].unique()
            )
            total_nodes_in_submodule_with_annotations = len(nodes_in_submodule_with_annotations)
            
            # Also calculate total nodes in submodule (for reference)
            total_nodes_in_submodule = len(node_table[node_table['submodule'] == submodule])
            
            # Create 2x2 contingency table
            # Rows: [in_submodule, not_in_submodule] 
            # Cols: [has_annotation, no_annotation]
            
            nodes_with_annotation_in_submodule = len(
                nodes_with_annotation & nodes_in_submodule_with_annotations
            )
            
            # Skip if this submodule has ≤1 nodes with this annotation
            if nodes_with_annotation_in_submodule <= 1:
                continue
            
            nodes_without_annotation_in_submodule = (
                total_nodes_in_submodule_with_annotations - nodes_with_annotation_in_submodule
            )
            
            nodes_with_annotation_not_in_submodule = (
                nodes_with_annotation_total - nodes_with_annotation_in_submodule  
            )
            
            nodes_without_annotation_not_in_submodule = (
                total_nodes_in_network_with_annotations - total_nodes_in_submodule_with_annotations - nodes_with_annotation_not_in_submodule
            )
            
            # Build contingency table
            contingency_table = np.array([
                [nodes_with_annotation_in_submodule, nodes_without_annotation_in_submodule],
                [nodes_with_annotation_not_in_submodule, nodes_without_annotation_not_in_submodule]
            ])
            
            # Calculate expected count under null hypothesis (using annotated nodes only)
            expected_count = (total_nodes_in_submodule_with_annotations * nodes_with_annotation_total) / total_nodes_in_network_with_annotations
            
            # Calculate fold enrichment
            if expected_count > 0:
                fold_enrichment = nodes_with_annotation_in_submodule / expected_count
            else:
                fold_enrichment = np.inf if nodes_with_annotation_in_submodule > 0 else 0
            
            # Perform Fisher's exact test (one-tailed, testing for enrichment)
            try:
                odds_ratio, p_value = fisher_exact(contingency_table, alternative='greater')
            except Exception as e:
                log.warning(f"Fisher's exact test failed for {annotation} in {submodule}: {e}")
                odds_ratio, p_value = np.nan, 1.0
            
            # Store results with both reference and actual test values
            results.append({
                'annotation_term': annotation,
                'submodule': submodule,
                'annotation_term_nodes_submodule': nodes_with_annotation_in_submodule,
                'annotation_term_nodes_total': nodes_with_annotation_total,
                # Reference values (total nodes including unannotated)
                'all_nodes_submodule': total_nodes_in_submodule,
                'all_nodes_total': total_nodes_in_network,
                # Actual test values (annotated nodes only)
                'annotated_nodes_submodule': total_nodes_in_submodule_with_annotations,
                'annotated_nodes_total': total_nodes_in_network_with_annotations,
                'expected_count': expected_count,
                'fold_enrichment': fold_enrichment,
                'odds_ratio': odds_ratio,
                'p_value': p_value
            })
    
    if not results:
        log.warning("No enrichment tests were performed after filtering")
        return pd.DataFrame()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Apply p-value correction
    valid_pvals = ~(results_df['p_value'].isna() | np.isinf(results_df['p_value']))
    
    if valid_pvals.sum() > 0:
        pval_results = multipletests(
            results_df.loc[valid_pvals, 'p_value'], 
            method=correction_method, 
            alpha=p_value_threshold
        )
        
        results_df['p_value_corr'] = np.nan
        results_df.loc[valid_pvals, 'p_value_corr'] = pval_results[1]
        results_df['significant'] = False
        results_df.loc[valid_pvals, 'significant'] = pval_results[0]
    else:
        results_df['p_value_corr'] = np.nan
        results_df['significant'] = False

    # Sort by corrected p-value, then by fold enrichment
    results_df = results_df.sort_values(['p_value_corr', 'fold_enrichment'], ascending=[True, False])
    sig_results_df = results_df[results_df['significant'] == True]

    # Summary statistics
    n_significant = results_df['significant'].sum()
    n_total_tests = len(results_df)
    
    log.info(f"Enrichment analysis complete:")
    log.info(f"  Total tests performed: {n_total_tests}")
    log.info(f"  Significant enrichments (< {p_value_threshold}): {n_significant}")
    log.info(f"  Background: {total_nodes_in_network_with_annotations}/{total_nodes_in_network} nodes with annotations")
    log.info(f"List of significant functional categories: ")
    if n_significant > 0:
        for idx, row in sig_results_df.iterrows():
            log.info(f"    Submodule: {row['submodule']}, Annotation: {row['annotation_term']}, Fold Enrichment: {row['fold_enrichment']:.2f}, Corrected p-value: {row['p_value_corr']:.4e}")
    else:
        log.info("    None")

    # Save results if output directory provided
    if output_dir:
        write_integration_file(
            data=results_df,
            output_dir=output_dir, 
            filename=output_filename,
            indexing=False
        )
    
    return results_df

# ====================================
# MOFA functions
# ====================================

# def run_full_mofa2_analysis(
#     integrated_data: pd.DataFrame,
#     mofa2_views: List[str],
#     metadata: pd.DataFrame,
#     output_dir: str,
#     output_filename: str,
#     num_factors: int = 5,
#     num_features: int = 10,
#     num_iterations: int = 100,
#     training_seed: int = 42,
#     overwrite: bool = False
# ) -> None:
#     """
#     Run a full MOFA2 analysis pipeline for multiple omics datasets and metadata groups.
#     https://biofam.github.io/MOFA2/

#     Args:
#         integrated_data (pd.DataFrame): Integreated omics dataframe (features x samples).
#         mofa2_views (list of str): Names for each omics view (e.g., ['tx', 'mx']).
#         metadata (pd.DataFrame): Metadata DataFrame (samples x variables).
#         output_dir (str): Output directory for results.
#         output_filename (str): Name for model h5 file.
#         num_factors (int): Number of MOFA factors to compute.
#         num_features (int): Number of top features to plot per factor.
#         num_iterations (int): Number of training iterations.
#         training_seed (int): Random seed for reproducibility.
#         overwrite (bool): Overwrite existing results if True.

#     Returns:
#         None
#     """

#     clear_directory(output_dir)

#     melted_dfs = []
#     for datatype in mofa2_views:
#         datatype_df = integrated_data.loc[integrated_data.index.str.contains(datatype)]
#         datatype_df = datatype_df.rename_axis('features').reset_index()
#         datatype_df_melted = datatype_df.melt(id_vars=['features'], var_name='sample', value_name='value')
#         datatype_df_melted['view'] = datatype
#         melted_dfs.append(datatype_df_melted)
#     mofa_data = pd.concat(melted_dfs, ignore_index=True)
    
#     # Merge with metadata
#     mofa_metadata = metadata.loc[metadata.index.isin(mofa_data['sample'].unique())]
#     mofa_data = mofa_data.merge(mofa_metadata, left_on='sample', right_index=True, how='left')
#     mofa_data = mofa_data[['sample', 'group', 'features', 'view', 'value']]
#     mofa_data = mofa_data.rename(columns={'features': 'feature'})

#     log.info("Converted omics data to mofa2 format:")

#     # Run the model and load to memory
#     model_file = run_mofa2_model(data=mofa_data, output_dir=output_dir, output_filename=output_filename,
#                                 num_factors=num_factors, num_iterations=num_iterations, 
#                                 training_seed=training_seed)
    
#     model = load_mofa2_model(model_file)
#     model.metadata = mofa_metadata
#     model.metadata.index.name = 'sample'

#     # Print and save output stats and info
#     info_text = f"""\
#     Information for model:
#         Samples: {model.shape[0]}
#         Features: {model.shape[1]}
#         Groups: {', '.join(model.groups)}
#         Datasets: {', '.join(model.views)}
#     """
#     info_file_path = os.path.join(output_dir, f"mofa2_model_summary.txt")
#     with open(info_file_path, 'w') as f:
#         f.write(info_text)
#     log.info(info_text)

#     # Calculate and save model stats
#     calculate_mofa2_feature_weights_and_r2(model=model, output_dir=output_dir, num_factors=num_factors)        

#     # Plot and save generic output graphs
#     #plot_mofa2_factor_r2(model=model, output_dir=output_subdir, num_factors=num_factors)
#     plot_mofa2_feature_weights_linear(model=model, num_features=num_features, output_dir=output_dir)
#     for dataset in mofa2_views:
#         plot_mofa2_feature_weights_scatter(model=model, data_type=dataset, factorX="Factor0", factorY="Factor1",
#                                            num_features=num_features, output_dir=output_dir)
#     # for dataset in mofa2_views:
#     #     plot_mofa2_feature_importance_per_factor(model=model, num_features=num_features,
#     #                                              data_type=dataset, output_dir=output_dir)
#     return mofa_data

# def run_mofa2_model(
#     data: pd.DataFrame,
#     output_dir: str = None,
#     output_filename: str = "mofa2_model.hdf5",
#     num_factors: int = 1,
#     num_iterations: int = 100,
#     training_seed: int = 42
# ) -> str:
#     """
#     Run MOFA2 model training and save the model to disk.

#     Args:
#         data (pd.DataFrame): Melted MOFA2 input DataFrame.
#         output_dir (str): Output directory for model file.
#         num_factors (int): Number of factors to compute.
#         num_iterations (int): Number of training iterations.
#         training_seed (int): Random seed for reproducibility.

#     Returns:
#         str: Path to the saved MOFA2 model file.
#     """

#     outfile = os.path.join(output_dir, output_filename)
        
#     ent = entry_point()
#     ent.set_data_options(
#         scale_views = False
#     )

#     ent.set_data_df(data)

#     ent.set_model_options(
#         factors = num_factors, 
#         spikeslab_weights = True, 
#         ard_weights = True,
#         ard_factors = True
#     )

#     ent.set_train_options(
#         iter = num_iterations,
#         convergence_mode = "fast", 
#         dropR2 = 0.001, 
#         gpu_mode = False, 
#         seed = training_seed
#     )

#     ent.build()

#     ent.run()

#     log.info(f"Exporting model to disk as {outfile}...")
#     ent.save(outfile=outfile,save_data=True)
#     return outfile

# def load_mofa2_model(filename: str):
#     """
#     Load a MOFA2 model from file.

#     Args:
#         filename (str): Path to MOFA2 model file.

#     Returns:
#         MOFA2 model object.
#     """

#     model = mofax.mofa_model(filename)
#     return model

# def calculate_mofa2_feature_weights_and_r2(
#     model,
#     output_dir: str,
#     num_factors: int
# ) -> None:
#     """
#     Calculate and save MOFA2 feature weights and R2 tables for each factor.

#     Args:
#         model: MOFA2 model object.
#         output_dir (str): Output directory to save results.
#         num_factors (int): Number of MOFA factors.

#     Returns:
#         None
#     """

#     feature_weight_per_factor = model.get_weights(df=True)
#     feature_weight_per_factor.set_index(feature_weight_per_factor.columns[0], inplace=False)
#     feature_weight_per_factor['abs_sum'] = feature_weight_per_factor.abs().sum(axis=1)
#     feature_weight_per_factor.sort_values(by='abs_sum', ascending=False, inplace=True)

#     r2_table = model.get_r2(factors=list(range(num_factors))).sort_values("R2", ascending=False)

#     log.info(f"Saving mofa2 factor weights and R2 tables...")
#     write_integration_file(feature_weight_per_factor, output_dir, f"mofa2_feature_weight_per_factor", index_label='Feature')
#     write_integration_file(r2_table, output_dir, f"mofa2_r2_per_factor", indexing=False)

# def plot_mofa2_factor_r2(
#     model,
#     output_dir: str,
#     num_factors: int
# ) -> "plt.Figure":
#     """
#     Plot and save the R2 values for each MOFA2 factor.

#     Args:
#         model: MOFA2 model object.
#         output_dir (str): Output directory to save the plot.
#         num_factors (int): Number of MOFA factors to compute.

#     Returns:
#         plt.Figure: The matplotlib figure object for the R2 plot.
#     """

#     r2_plot = mofax.plot_r2(model, factors=list(range(num_factors)), cmap="Blues")

#     log.info(f"Printing and saving mofa2 factor R2 plot...")
#     r2_plot.figure.savefig(f'{output_dir}/mofa2_r2_per_factor.pdf')

#     return r2_plot

# def plot_mofa2_feature_weights_linear(
#     model,
#     output_dir: str,
#     num_features: int
# ) -> "plt.Figure":
#     """
#     Plot and save the linear feature weights for MOFA2 factors.

#     Args:
#         model: MOFA2 model object.
#         output_dir (str): Output directory to save the plot.
#         num_features (int): Number of top features to plot per factor.

#     Returns:
#         plt.Figure: The matplotlib figure object for the feature weights plot.
#     """

#     feature_plot = mofax.plot_weights(model, n_features=num_features, label_size=7)
#     feature_plot.figure.set_size_inches(16, 8)

#     log.info(f"Printing and saving mofa2 feature weights linear plot...")
#     feature_plot.figure.savefig(f'{output_dir}/mofa2_feature_weights_linear_plot_combined_data.pdf')

#     return feature_plot

# def plot_mofa2_feature_weights_scatter(
#     model,
#     data_type: str,
#     factorX: str,
#     factorY: str,
#     num_features: int,
#     output_dir: str
# ) -> "plt.Figure":
#     """
#     Plot and save a scatterplot of feature weights for two MOFA2 factors.

#     Args:
#         model: MOFA2 model object.
#         data_type (str): Omics view (e.g., 'tx', 'mx').
#         factorX (str): Name of the first factor (e.g., 'Factor0').
#         factorY (str): Name of the second factor (e.g., 'Factor1').
#         num_features (int): Number of top features to plot.
#         output_dir (str): Output directory to save the plot.

#     Returns:
#         plt.Figure: The matplotlib figure object for the scatter plot.
#     """

#     feature_plot = mofax.plot_weights_scatter(
#         model,
#         view=data_type,
#         x=factorX, 
#         y=factorY, 
#         hist=True,
#         label_size=8,
#         n_features=num_features, 
#         linewidth=0, 
#         alpha=0.5,
#         height=10  # For sns.jointplot
#     )
#     feature_plot.figure.set_size_inches(10, 10)

#     log.info(f"Printing and saving {data_type} mofa2 feature weights scatter plot...")
#     feature_plot.figure.savefig(f'{output_dir}/mofa2_feature_weights_scatter_plot_{data_type}_data.pdf')

#     return feature_plot

# def plot_mofa2_feature_importance_per_factor(
#     model,
#     num_features: int,
#     data_type: str,
#     output_dir: str
# ) -> "plt.Figure":
#     """
#     Plot and save a dotplot of feature importance per MOFA2 factor.

#     Args:
#         model: MOFA2 model object.
#         num_features (int): Number of top features to plot.
#         data_type (str): Omics view (e.g., 'tx', 'mx').
#         output_dir (str): Output directory to save the plot.

#     Returns:
#         plt.Figure: The matplotlib figure object for the dotplot.
#     """

#     plot = mofax.plot_weights_dotplot(model, 
#                                         n_features=num_features, 
#                                         view=data_type,
#                                         w_abs=True, 
#                                         yticklabels_size=8)

#     log.info(f"Printing and saving {data_type} mofa2 feature importance per factor plot...")
#     plot.figure.savefig(f'{output_dir}/mofa2_feature_importance_per_factor_for_{data_type}_data.pdf')

#     return plot

# # ====================================
# # MAGI functions
# # ====================================

# # def run_magi2(
# #     run_name: str,
# #     sequence_file: str,
# #     compound_file: str,
# #     output_dir: str,
# #     magi2_source_dir: str,
# #     overwrite: bool = False
# # ) -> tuple[str, str]:
# #     """
# #     Run MAGI2 analysis by submitting a SLURM job on NERSC.

# #     Args:
# #         run_name (str): Name for the MAGI2 run.
# #         sequence_file (str): Path to the protein FASTA file.
# #         compound_file (str): Path to the compound input file.
# #         output_dir (str): Output directory for MAGI2 results.
# #         magi2_source_dir (str): Directory containing MAGI2 scripts.
# #         overwrite (bool): Overwrite existing results if True.

# #     Returns:
# #         tuple: (compound_results_file, gene_results_file) paths to MAGI2 output files.
# #     """

# #     if os.path.exists(f"{output_dir}/{run_name}/output_{run_name}") and overwrite is False:
# #         log.info(f"MAGI2 results directory already exists: {output_dir}/{run_name}/output_{run_name}. \nNot queueing new job.")
# #         log.info("Returning path to existing results files.")
# #         compound_results_file = f"{output_dir}/{run_name}/output_{run_name}/magi2_compounds.csv"
# #         gene_results_file = f"{output_dir}/{run_name}/output_{run_name}/magi2_gene_results.csv"
# #         return compound_results_file, gene_results_file
# #     elif os.path.exists(f"{output_dir}/{run_name}/output_{run_name}/") and overwrite is True:
# #         log.info(f"Overwriting existing submodule Sankey diagrams in {os.path.join(submodules_dir, 'submodule_sankeys')}.")
# #         shutil.rmtree(os.path.join(output_dir, "magi2_sankeys"))
# #         os.makedirs(os.path.join(output_dir, "magi2_sankeys"), exist_ok=True)
# #     elif not os.path.exists(f"{output_dir}/{run_name}/output_{run_name}/"):
# #         os.makedirs(f"{output_dir}/{run_name}/output_{run_name}/", exist_ok=True)

# #     log.info("Queueing MAGI2 with sbatch...\n")

# #     SLURM_PERLMUTTER_HEADER = """#!/bin/bash

# # #SBATCH -N 1
# # #SBATCH --exclusive
# # #SBATCH --error="slurm.err"
# # #SBATCH --output="slurm.out"
# # #SBATCH -A m2650
# # #SBATCH -C cpu
# # #SBATCH --qos=regular
# # #SBATCH -t 6:00:00

# #     """

# #     magi2_sbatch_filename = f"{output_dir}/magi2_slurm.sbatch"
# #     magi2_runner_filename = f"{output_dir}/magi2_kickoff.sh"
# #     cmd = f"{magi2_source_dir}/run_magi2.sh {run_name} {sequence_file} {compound_file} {output_dir} {magi2_source_dir}\n\necho MAGI2 completed."

# #     with open(magi2_sbatch_filename,'w') as fid:
# #         fid.write(f"{SLURM_PERLMUTTER_HEADER.replace('slurm', f'{output_dir}/magi2_slurm')}\n{cmd}")
# #     with open(magi2_runner_filename,'w') as fid:
# #         fid.write(f"sbatch {magi2_sbatch_filename}")

# #     with open(magi2_runner_filename, 'r') as fid:
# #         cmd = fid.read()
# #         sbatch_output = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
# #         sbatch_output_str = sbatch_output.stdout.decode('utf-8').replace('\n', '')
# #         job_id = sbatch_output_str.split()[-1]

# #     user = os.getenv('USER')
# #     log.info(f"MAGI2 job submitted with ID: {job_id}. Check status with 'squeue -u {user}'.")
# #     return "",""

# # def summarize_fbmn_results(
# #     fbmn_results_dir: str,
# #     polarity: str,
# #     overwrite: bool = False
# # ) -> None:
# #     """
# #     Summarize FBMN results and export a cleaned compound table.

# #     Args:
# #         fbmn_results_dir (str): Directory with FBMN results.
# #         polarity (str): Polarity ('positive', 'negative', 'multipolarity').
# #         overwrite (bool): Overwrite existing output if True.

# #     Returns:
# #         None
# #     """

# #     fbmn_output_filename = f"{fbmn_results_dir}/fbmn_compounds.csv"
# #     if os.path.exists(fbmn_output_filename) and overwrite is False:
# #         log.info(f"FBMN output summary already exists: {fbmn_output_filename}. Not overwriting.")
# #         return
    
# #     log.info(f"Summarizing FBMN file {fbmn_output_filename}.")
# #     try:
# #         if polarity == "multipolarity":
# #             compound_files = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*library-results.tsv"))
# #             if compound_files:
# #                 if len(compound_files) > 1:
# #                     multipolarity_datasets = []
# #                     for mx_data_file in compound_files:
# #                         file_polarity = (
# #                             "positive" if "positive" in mx_data_file 
# #                             else "negative" if "negative" in mx_data_file 
# #                             else "NO_POLARITY"
# #                         )
# #                         mx_dataset = pd.read_csv(mx_data_file, sep='\t')
# #                         mx_data = mx_dataset.copy()
# #                         mx_data["#Scan#"] = "mx_" + mx_data["#Scan#"].astype(str) + "_" + file_polarity
# #                         multipolarity_datasets.append(mx_data)
# #                     fbmn_summary = pd.concat(multipolarity_datasets, axis=0)
# #                 elif len(compound_files) == 1:
# #                     log.info(f"Warning: Only single compound files found with polarity set to multipolarity.")
# #                     fbmn_summary = pd.read_csv(compound_files[0], sep='\t')
# #         elif polarity in ["positive", "negative"]:
# #             compound_file = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*{polarity}*library-results.tsv"))
# #             if len(compound_file) > 1:
# #                 log.info(f"Multiple compound files found: {compound_file}. \n\nPlease check {fbmn_results_dir} to verify and use the compound_filename argument.")
# #                 return None
# #             elif len(compound_file) == 0:
# #                 log.info(f"No compound files found for {polarity} polarity.")
# #                 return None
# #             elif len(compound_file) == 1:
# #                 fbmn_summary = pd.read_csv(compound_file[0], sep='\t')
# #                 fbmn_summary["#Scan#"] = "mx_" + fbmn_summary["#Scan#"].astype(str) + "_" + file_polarity
# #             log.info(f"\nUsing metabolomics compound file at {compound_file}")
# #     except Exception as e:
# #         log.info(f"Compound file could not be read due to {e}")
# #         return None

# #     log.info("\tConverting FBMN outputs to summary format.")
# #     fbmn_data = fbmn_summary.rename(columns={"Smiles": 'original_compound'})
# #     fbmn_data['original_compound'] = fbmn_data['original_compound'].str.strip().replace(r'^\s*$', None, regex=True)

# #     # Check that smiles is valid to get mol object with rdkit
# #     keep_smiles = []
# #     for smiles in list(fbmn_data['original_compound']):
# #         if pd.isna(smiles):
# #             continue
# #         smiles_str = str(smiles)
# #         mol = Chem.MolFromSmiles(smiles_str)
# #         if mol is not None:
# #             keep_smiles.append(smiles_str)
# #     fbmn_data = fbmn_data[fbmn_data['original_compound'].isin(keep_smiles)]

# #     fbmn_data.to_csv(fbmn_output_filename, index=False)
    
# #     return

# # def convert_nucleotides_to_amino_acids(
# #     input_fasta: str
# # ) -> list:
# #     """
# #     Convert nucleotide FASTA sequences to amino acid sequences.

# #     Args:
# #         input_fasta (str): Path to nucleotide FASTA file.

# #     Returns:
# #         list: List of SeqRecord objects with amino acid sequences.
# #     """

# #     amino_acid_sequences = []
    
# #     if input_fasta.endswith('.gz'):
# #         open_func = gzip.open
# #     else:
# #         open_func = open
    
# #     with open_func(input_fasta, 'rt') as in_handle:
# #         for record in SeqIO.parse(in_handle, "fasta"):
# #             amino_acid_seq = record.seq.translate()
# #             amino_acid_seq = amino_acid_seq.replace('*', '')
# #             new_record = record[:]
# #             new_record.seq = amino_acid_seq
# #             amino_acid_sequences.append(new_record)
    
# #     log.info(f"\tConverted nucleotide -> amino acid sequences.")
# #     return amino_acid_sequences

# # ====================================
# # Output functions
# # ====================================

# def create_excel_metadata_sheet(
#     tx_metadata: pd.DataFrame,
#     mx_metadata: pd.DataFrame,
#     output_dir: str,
#     project_name: str
# ) -> None:
#     """
#     Create an Excel metadata template for user integration of transcriptomics and metabolomics metadata.

#     Args:
#         tx_metadata (pd.DataFrame): Transcriptomics metadata DataFrame.
#         mx_metadata (pd.DataFrame): Metabolomics metadata DataFrame.
#         output_dir (str): Output directory for the Excel file.
#         project_name (str): Project name for the file.

#     Returns:
#         None
#     """

#     # Create the data for the "Instructions" tab
#     log.info("Creating metadata Excel file...")
#     instructions_text = (
#         "Placeholder\n\n"
#     )
#     critical_message = (
#         "Placeholder\n\n"
#     )

#     instructions_df = pd.DataFrame([instructions_text], columns=["Instructions"])

#     # Create the data for the "RNA" tab
#     transcript_df = tx_metadata[['library_name', 'sample_name', 'sample_isolated_from', 'collection_isolation_site_or_growth_conditions']]
#     transcript_df.columns = ['JGI_Library_Name', 'JGI_Metadata_SampleName', 'JGI_Metadata_IsolatedFrom', 'JGI_Metadata_Conditions']
#     transcript_df['JGI_DataType'] = "Transcriptomics"
#     transcript_df['JGI_SampleIndex'] = ['tx' + str(i) for i in range(1, len(transcript_df) + 1)]
#     transcript_example_df = pd.DataFrame({
#     'JGI_SampleIndex': ['Example', 'Example', 'Example', 'Example', 'Example', 'Example'],
#     'JGI_DataType': ['Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics'],
#     'JGI_Library_Name': ['GOYZZ', 'GOZAA', 'GOZBS', 'GOZBT', 'GOZAW', 'GOZAX'],
#     'JGI_Metadata_SampleName': ['Swg_Sum_GH_Q13_1', 'Swg_Sum_GH_Q13_2', 'Swg_Fall_GH_Q15_1', 'Swg_Fall_GH_Q15_2', 'Swg_Sum_FT_Q15_1', 'Swg_Sum_FT_Q15_2'],
#     'JGI_Metadata_IsolatedFrom': ['Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Field grown Switchgrass (UCDavis)', 'Field grown Switchgrass (UCDavis)'],
#     'JGI_Metadata_Conditions': ['Greenhouse 1st harvest, Summer_August2018', 'Greenhouse 1st harvest, Summer_August2018', 'Greenhouse 2nd harvest, Fall_Oct2018', 'Greenhouse 2nd harvest, Fall_Oct2018', 'Field trial 1st harvest, Summer_June2018', 'Field trial 1st harvest, Summer_June2018'],
#     'USER_MetadataVar_1': ['Summer', 'Summer', 'Fall', 'Fall', 'Summer', 'Summer'],
#     'USER_MetadataVar_2': ['Greenhouse', 'Greenhouse', 'Greenhouse', 'Greenhouse', 'Field', 'Field'],
#     'USER_MetadataVar_3': ['Q13', 'Q13', 'Q15', 'Q15', 'Q15', 'Q15'],
#     'USER_MetadataVar_4': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
#     'USER_MetadataVar_5': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
#     'USER_Replicate': ["1", "2", "1", "2", "1", "2"]
#     })
    

#     # Create the data for the "Metabolites" tab
#     metabolites_df = mx_metadata[['file', 'full_sample_metadata']]
#     metabolites_df = metabolites_df[~metabolites_df['file'].str.contains(r'_QC_|ctrl|CTRL|ExCtrl', regex=True)]
#     metabolites_df.columns = ['JGI_Library_Name', 'JGI_Metadata_SampleName']
#     metabolites_df['JGI_DataType'] = "Metabolomics"
#     metabolites_df['JGI_SampleIndex'] = ['mx' + str(i) for i in range(1, len(metabolites_df) + 1)]
#     metabolites_example_df = pd.DataFrame({
#     'JGI_SampleIndex': ['Example', 'Example', 'Example', 'Example', 'Example', 'Example'],
#     'JGI_DataType': ['Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics'],
#     'JGI_Library_Name': ['20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_121_Swg-SumHvst-GrnHGrwn-QsuB13_1_Rg90to1350-CE102040-30mg-S1_Run18', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_121_Swg-SumHvst-GrnHGrwn-QsuB13_2_Rg90to1350-CE102040-30mg-S1_Run19', \
#                         '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_97_Swg-SumHvst-FldGrwn-QsuB15_1_Rg90to1350-CE102040-30mg-S1_Run27', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_97_Swg-SumHvst-FldGrwn-QsuB15_2_Rg90to1350-CE102040-30mg-S1_Run28', \
#                         '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_82_Swg-SumHvst-FldGrwn-QsuB10_1_Rg90to1350-CE102040-30mg-S1_Run45', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_82_Swg-SumHvst-FldGrwn-QsuB10_2_Rg90to1350-CE102040-30mg-S1_Run46'],
#     'JGI_Metadata_SampleName': ['swg-sumhvst-grnhgrwn-qsub13', 'swg-sumhvst-grnhgrwn-qsub13', \
#                      'swg-sumhvst-fldgrwn-qsub15', 'swg-sumhvst-fldgrwn-qsub15', \
#                      'swg-sumhvst-fldgrwn-qsub10', 'swg-sumhvst-fldgrwn-qsub10'],
#     'USER_MetadataVar_1': ['Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer'],
#     'USER_MetadataVar_2': ['Greenhouse', 'Greenhouse', 'Field', 'Field', 'Field', 'Field'],
#     'USER_MetadataVar_3': ['Q13', 'Q13', 'Q15', 'Q15', 'Q10', 'Q10'],
#     'USER_MetadataVar_4': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
#     'USER_MetadataVar_5': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
#     'USER_Replicate': ["1", "2", "1", "2", "1", "2"]
#     })

#     linked_data = pd.concat([transcript_df, metabolites_df], ignore_index=True, join='outer')
#     linked_data = pd.concat([transcript_example_df, linked_data], ignore_index=True, join='outer')
#     linked_data = pd.concat([metabolites_example_df, linked_data], ignore_index=True, join='outer')
#     linked_data = linked_data.reindex(columns=transcript_df.columns)
#     linked_data.fillna("-", inplace=True)
#     linked_data['USER_MetadataVar_1'] = pd.NA
#     linked_data['USER_MetadataVar_2'] = pd.NA
#     linked_data['USER_MetadataVar_3'] = pd.NA
#     linked_data['USER_MetadataVar_4'] = pd.NA
#     linked_data['USER_MetadataVar_5'] = pd.NA
#     linked_data['USER_Replicate'] = pd.NA
#     linked_data['JGI_Group'] = pd.NA
#     linked_data['JGI_Unique_SampleName'] = pd.NA
#     linked_data = linked_data[['JGI_SampleIndex', 'JGI_DataType', 'JGI_Library_Name', 'JGI_Metadata_SampleName', 'JGI_Metadata_IsolatedFrom', 'JGI_Metadata_Conditions', 'USER_MetadataVar_1', 'USER_MetadataVar_2', 'USER_MetadataVar_3', 'USER_MetadataVar_4', 'USER_MetadataVar_5', 'USER_Replicate', 'JGI_Group', 'JGI_Unique_SampleName']]

#     # Write to Excel file
#     output_path = f'{output_dir}/{project_name}_metadata_integration_template.xlsx'
#     with pd.ExcelWriter(output_path) as writer:
#         instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
#         linked_data.to_excel(writer, sheet_name='linked_Datasets', index=False)
#         tx_metadata.to_excel(writer, sheet_name='Full_Transcript_Metadata_Ref', index=False)
#         mx_metadata.to_excel(writer, sheet_name='Full_Metabolite_Metadata_Ref', index=False)

#     # Load the workbook and set the wrap text format for the Instructions sheet
#     workbook = load_workbook(output_path)
#     worksheet = workbook['Instructions']
#     worksheet['B1'] = "CRITICAL:"
#     worksheet['B1'].font = Font(color='FF0000', bold=True)
#     worksheet['B2'] = critical_message
#     worksheet['B2'].font = Font(bold=True)
#     for row in worksheet.iter_rows():
#         for cell in row:
#             cell.alignment = Alignment(wrap_text=True, vertical='top')
#     # Set column widths for the Instructions sheet
#     worksheet.column_dimensions['A'].width = 130
#     worksheet.column_dimensions['B'].width = 130
#     worksheet.row_dimensions[2].height = 1070
    
#     # Set column widths for the RNA sheet
#     worksheet = workbook['linked_Datasets']
#     for col, width in zip('ABCDEFGHIJKLMN', [15,15,20,30,40,70,30,30,30,30,30,15,40,40]): worksheet.column_dimensions[col].width = width
#     # Add formula to combine metadata columns
#     for row in range(2, worksheet.max_row + 1):
#         group_formula = (
#             f'=IF(AND(LEN(G{row})=0, LEN(H{row})=0, LEN(I{row})=0, LEN(J{row})=0, LEN(K{row})=0),"Undefined Group", '
#             f'IF(LEN(G{row})>0,G{row},"")'
#             f'&IF(LEN(H{row})>0,"---"&H{row},"")'
#             f'&IF(LEN(I{row})>0,"---"&I{row},"")'
#             f'&IF(LEN(J{row})>0,"---"&J{row},"")'
#             f'&IF(LEN(K{row})>0,"---"&K{row},""))'
#         )
#         worksheet[f'M{row}'] = group_formula
#         unique_formula = f'=IF(LEN(L{row})>0,M{row}&"---"&L{row},"Missing Replicate Number")'
#         worksheet[f'N{row}'] = unique_formula
#     # Highlight cells for user to fill in
#     fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
#     thin_border = Border(left=Side(style='thin', color='000000'),
#                          right=Side(style='thin', color='000000'),
#                          top=Side(style='thin', color='000000'),
#                          bottom=Side(style='thin', color='000000'))    
#     for row in range(14, worksheet.max_row + 1):
#         for col in ['G', 'H', 'I', 'J', 'K', 'L']:
#             worksheet[f'{col}{row}'].fill = fill
#             worksheet[f'{col}{row}'].border = thin_border
#     for row in range(2, 14):
#         for col in range(1, worksheet.max_column + 1):
#             cell = worksheet.cell(row=row, column=col)
#             cell.font = Font(italic=True, color="D3D3D3")
#     worksheet.freeze_panes = worksheet['A14']

#     log.info(f"\tMetadata instructions Excel file exported to {output_path}")
#     # Save the workbook
#     workbook.save(output_path)

# def copy_data(
#     src_dir: str,
#     results_subdir: str,
#     file_map: dict,
#     plots: bool = False,
#     plot_pattern: str = "*.pdf"
# ) -> None:
#     """
#     Copy data and plot files to user output directories.

#     Args:
#         src_dir (str): Source directory.
#         results_subdir (str): Destination subdirectory.
#         file_map (dict): Mapping of source to destination filenames.
#         plots (bool): Copy plot files if True.
#         plot_pattern (str): Glob pattern for plot files.

#     Returns:
#         None
#     """
#     os.makedirs(results_subdir, exist_ok=True)
#     # Copy main files
#     for src, dst in file_map.items():
#         src_path = os.path.join(src_dir, src)
#         dst_path = os.path.join(results_subdir, dst)
#         if os.path.exists(src_path):
#             shutil.copy(src_path, dst_path)
#         else:
#             log.info(f"Warning: Source file not found: {src_path}")

#     if not plots:
#         return

#     # Helper to copy plot files
#     def copy_plot_files(pattern, subfolder):
#         plot_dir = os.path.join(src_dir, "plots")
#         dest_dir = os.path.join(results_subdir, subfolder)
#         os.makedirs(dest_dir, exist_ok=True)
#         for file_path in glob.glob(os.path.join(plot_dir, pattern)):
#             file_name = os.path.basename(file_path)
#             shutil.copy(file_path, os.path.join(dest_dir, file_name))

#     copy_plot_files(f"*_nonnormalized{plot_pattern}", "plots_of_non-normalized_data")
#     copy_plot_files(f"*_normalized{plot_pattern}", "plots_of_normalized_data")
#     copy_plot_files(f"*_grid{plot_pattern}", "plots_of_combined_data")

# def create_user_output_directory(
#     mx_dir: str,
#     tx_dir: str,
#     integrated_dir: str,
#     final_dir: str,
#     project_name: str
# ) -> None:
#     """
#     Organize and copy results to a user-facing output directory structure.

#     Args:
#         mx_dir (str): Metabolomics results directory.
#         tx_dir (str): Transcriptomics results directory.
#         integrated_dir (str): Integrated results directory.
#         final_dir (str): Final user output directory.
#         project_name (str): Name of the project.

#     Returns:
#         None
#     """

#     copy_data(
#         tx_dir,
#         f"{final_dir}/transcript_results",
#         {
#             #"tx_metadata.csv": "full_transcript_metadata.csv",
#             "integrated_metadata.csv": "transcript_metadata.csv",
#             "tx_data_nonnormalized.csv": "transcript_count_data_non-normalized.csv",
#             "tx_data_normalized.csv": "transcript_count_data_normalized.csv"
#         },
#         plots=True
#     )
#     copy_data(
#         mx_dir,
#         f"{final_dir}/metabolite_results",
#         {
#             #"mx_metadata.csv": "full_metabolite_metadata.csv",
#             "integrated_metadata.csv": "metabolite_metadata.csv",
#             "mx_data_nonnormalized.csv": "metabolite_count_data_non-normalized.csv",
#             "mx_data_normalized.csv": "metabolite_count_data_normalized.csv"
#         },
#         plots=True
#     )
#     copy_data(
#         integrated_dir,
#         f"{final_dir}/integrated_results",
#         {
#             "integrated_metadata.csv": "integrated_metadata.csv",
#             "integrated_data.csv": "integrated_data.csv",
#             "integrated_data_annotated.csv": "integrated_data_annotated.csv"
#         },
#         plots=False
#     )

#     # Integrated data and plots
#     integrated_results_subdir = f"{final_dir}/integrated_results"
#     subdirs = {
#         "data_distributions": f"{integrated_results_subdir}/data_distributions",
#         "network_analyses": f"{integrated_results_subdir}/network_analyses",
#         "mofa2_analyses": f"{integrated_results_subdir}/mofa2_analyses",
#         "magi2_analyses": f"{integrated_results_subdir}/magi2_analyses"
#     }

#     for subdir in subdirs.values():
#         os.makedirs(subdir, exist_ok=True)
#     for src, dst in subdirs.items():
#         shutil.copytree(f"{integrated_dir}/{src}", dst, dirs_exist_ok=True)
#         if "magi" in src:
#             for file in glob.glob(os.path.join(dst, "*.sh")) + \
#                         glob.glob(os.path.join(dst, project_name, "log*")) + \
#                         glob.glob(os.path.join(dst, project_name, "error*")) + \
#                         [os.path.join(dst, project_name, f"output_{project_name}", "magi_results.csv")]:
#                 os.remove(file)
#             intermediate_dir = os.path.join(dst, project_name, f"output_{project_name}", "intermediate_files")
#             shutil.rmtree(intermediate_dir)

#     log.info(f"User output directory structure created at {final_dir}")

#     return

# def upload_to_google_drive(
#     project_folder: str,
#     project_name: str,
#     overwrite: bool = True
# ) -> None:
#     """
#     Upload all contents of the user results directory to 
#     Google Drive JGI_MXTX_Integration_Projects/{project_name} using rclone.
#     """
#     dest_folder = f"JGI_MXTX_Integration_Projects:{project_name}/"
#     orig_folder = project_folder

#     if overwrite is True:
#         log.info("Warning! Overwriting existing files in Google Drive.")
#         upload_command = (
#             f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone sync '
#             f'"{orig_folder}/" "{dest_folder}"'
#         )
#     else:
#         log.info("Warning! Not overwriting existing files in Google Drive, may have previous files in the output.")
#         upload_command = (
#             f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone copy --ignore-existing '
#             f'"{orig_folder}/" "{dest_folder}"'
#         )
#     try:
#         log.info(f"Uploading to Google Drive with command:\t{upload_command}")
#         subprocess.check_output(upload_command, shell=True)
#     except Exception as e:
#         log.info(f"Warning! Google Drive upload failed with exception: {e}. Command: {upload_command}")
#         return

#     # Check that upload worked
#     check_upload_command = (
#         f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone ls "{dest_folder}" --max-depth 2'
#     )
#     try:
#         check_upload_out = subprocess.check_output(check_upload_command, shell=True)
#         if check_upload_out.decode('utf-8').strip():
#             log.info(f"Google Drive upload confirmed!")
#             return
#         else:
#             log.info(f"Warning! Google Drive upload check failed because no data was returned with command: {check_upload_command}. Upload may not have been successful.")
#             return
#     except Exception as e:
#         log.info(f"Warning! Google Drive upload failed on upload check with exception: {e}. Command: {check_upload_command}")
#         return