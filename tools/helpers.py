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
import requests
from functools import reduce, lru_cache
from rapidfuzz import fuzz, process

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
from matplotlib.colors import PowerNorm

# --- Machine learning & statistics ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.covariance import GraphicalLasso
from statsmodels.stats.multitest import multipletests
from itertools import product
from scipy.stats import hypergeom

# --- Bioinformatics ---
import gff3_parser
from Bio import SeqIO

# --- PDF and Excel handling ---
from openpyxl import load_workbook
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# --- Network analysis ---
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
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

# ====================================
# Analysis step functions
# ====================================

def variance_selection(
    data: pd.DataFrame,
    top_n: int = None,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Select the top ``max_features`` most variable features by row-wise variance.

    Parameters
    ----------
    data : pd.DataFrame
        Feature-by-sample (or feature-by-condition) matrix.
    top_n : int, optional
        Alias for ``max_features`` (kept for config compatibility).
    max_features : int
        Number of top-variance features to retain.

    Returns
    -------
    pd.DataFrame
        Subset of ``data`` containing only the top-variance features.
    """
    n_keep = top_n if top_n is not None else max_features
    var_series = data.var(axis=1).sort_values(ascending=False)
    top_idx = var_series.index[:n_keep]
    log.info(f"Variance selection: keeping top {n_keep} most variable features.")
    return data.loc[top_idx]


def feature_list_selection(
    data: pd.DataFrame,
    feature_list_file: Union[str, Path],
    max_features: int = 10000,
) -> pd.DataFrame:
    """
    Subset features by a user-provided list (one feature name per line).

    Parameters
    ----------
    data : pd.DataFrame
        Feature-by-sample (or feature-by-condition) matrix.
    feature_list_file : str or Path
        Path to a plain-text file with one feature ID per line.
    max_features : int
        Hard cap on the number of features returned.

    Returns
    -------
    pd.DataFrame
        Subset of ``data`` restricted to features in the list (up to ``max_features``).
    """
    path = Path(feature_list_file)
    if not path.is_file():
        raise ValueError(f"Feature list file not found: {path}")
    feature_list = pd.read_csv(path, header=None, sep=r"\s+", engine="python")[0].astype(str).tolist()
    intersect = list(set(data.index).intersection(feature_list))
    if not intersect:
        log.warning("No features from the list were present in the data matrix.")
        return pd.DataFrame()
    ordered = [f for f in feature_list if f in intersect][:max_features]
    log.info(f"Feature list selection: {len(ordered)} features retained from {feature_list_file}.")
    return data.loc[ordered]


def perform_feature_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: str = None,
    output_filename: str = None,
) -> pd.DataFrame:
    """
    Apply feature selection to the integrated matrix.

    Supported methods (set ``analysis.feature_selection.method`` in ``analysis.yml``):

    * ``"variance"``     — keep the top-N most variable features (row-wise variance).
    * ``"feature_list"`` — keep only features present in a user-supplied text file.

    Parameters
    ----------
    data : pd.DataFrame
        Integrated feature matrix (features × samples or features × conditions).
    metadata : pd.DataFrame
        Integrated metadata (not used for selection, kept for API consistency).
    config : dict
        The ``feature_selection`` sub-dict from ``analysis.yml``.
    output_dir : str
        Directory to write the selected feature matrix.
    output_filename : str
        Output CSV filename (without extension).

    Returns
    -------
    pd.DataFrame
        Subset of ``data`` after feature selection.
    """
    method = config.get("method") or config.get("selected_method")
    if method is None:
        log.info("No feature_selection method specified — returning all features.")
        return data
    method = method.lower().strip()

    params_block = config.get("params", config.get(method, {}))
    max_features = params_block.get("max_features", 5000)

    valid_methods = {"variance", "feature_list"}
    if method not in valid_methods:
        raise ValueError(
            f"Unsupported feature-selection method '{method}'. "
            f"Valid options: {sorted(valid_methods)}"
        )

    log.info(f"Performing feature selection using method '{method}'.")

    if method == "variance":
        top_n = params_block.get("top_n", max_features)
        subset = variance_selection(data, top_n=top_n, max_features=max_features)
    else:  # feature_list
        feature_list_file = params_block.get("feature_list_file")
        if not feature_list_file:
            raise ValueError(
                "feature_selection.params.feature_list_file must be set when method='feature_list'."
            )
        subset = feature_list_selection(data, feature_list_file=feature_list_file, max_features=max_features)

    if subset.empty:
        raise ValueError(
            f"Feature selection (method='{method}') returned an empty matrix. "
            "Adjust your parameters."
        )

    log.info(f"Feature selection complete: {subset.shape[0]} features retained out of {data.shape[0]}.")
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

def plot_feature_pair_correlation(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_1: str,
    feature_2: str,
    color_by: str = None,
    output_dir: str = None,
    show_plot: bool = True,
    figsize: tuple = (8, 6),
    alpha: float = 0.7,
    s: int = 50
) -> dict:
    """
    Create a scatter plot showing the correlation between two features,
    optionally colored by a metadata category.
    
    Args:
        data (pd.DataFrame): Feature matrix (features x samples)
        metadata (pd.DataFrame): Metadata DataFrame (samples x variables)
        feature_1 (str): Name of first feature (x-axis)
        feature_2 (str): Name of second feature (y-axis)
        color_by (str, optional): Metadata column to use for point colors
        output_dir (str, optional): Directory to save plot
        show_plot (bool): Whether to display plot inline
        figsize (tuple): Figure size in inches
        alpha (float): Point transparency (0-1)
        s (int): Point size

    """
    
    # Validate features exist in data
    if feature_1 not in data.index:
        raise ValueError(f"Feature '{feature_1}' not found in data")
    if feature_2 not in data.index:
        raise ValueError(f"Feature '{feature_2}' not found in data")
    
    # Extract feature vectors
    x = data.loc[feature_1].values
    y = data.loc[feature_2].values
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': x,
        'y': y,
        'sample': data.columns
    })
    
    # Merge with metadata if color_by specified
    if color_by is not None:
        if color_by not in metadata.columns:
            raise ValueError(f"Metadata column '{color_by}' not found")
        
        # Merge on sample names
        plot_df = plot_df.merge(
            metadata[[color_by]], 
            left_on='sample', 
            right_index=True, 
            how='left'
        )
    
    # Remove NaN values for correlation calculation
    mask = ~(np.isnan(plot_df['x']) | np.isnan(plot_df['y']))
    plot_df_clean = plot_df[mask].copy()
    
    if len(plot_df_clean) < 3:
        log.warning(f"Only {len(plot_df_clean)} valid samples for correlation")
        return {
            'feature_1': feature_1,
            'feature_2': feature_2,
            'r': np.nan,
            'p_value': np.nan,
            'n_samples': len(plot_df_clean)
        }
    
    # Calculate correlation statistics
    from scipy.stats import pearsonr
    r, p_value = pearsonr(plot_df_clean['x'], plot_df_clean['y'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    if color_by is not None and color_by in plot_df_clean.columns:
        # Get unique groups and sort them
        unique_groups = plot_df_clean[color_by].unique()
        
        # Sort groups - try numeric first, then alphabetic
        try:
            # Try to convert to numeric and sort
            unique_groups_sorted = sorted(unique_groups, key=lambda x: float(x))
        except (ValueError, TypeError):
            # If conversion fails, sort alphabetically
            unique_groups_sorted = sorted(unique_groups, key=str)
        
        # Plot with color grouping using viridis palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_groups_sorted)))
        
        for idx, group in enumerate(unique_groups_sorted):
            group_data = plot_df_clean[plot_df_clean[color_by] == group]
            ax.scatter(
                group_data['x'], 
                group_data['y'],
                label=group,
                alpha=alpha,
                s=s,
                color=colors[idx],
                edgecolors='black',
                linewidth=0.5
            )
        ax.legend(title=color_by, loc='best', framealpha=0.9)
    else:
        # Plot without color grouping (use viridis purple)
        ax.scatter(
            plot_df_clean['x'],
            plot_df_clean['y'],
            alpha=alpha,
            s=s,
            c='#440154',  # Viridis dark purple
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add regression line
    x_line = np.linspace(plot_df_clean['x'].min(), plot_df_clean['x'].max(), 100)
    slope, intercept = np.polyfit(plot_df_clean['x'], plot_df_clean['y'], 1)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7, label='Linear fit')
    
    # Formatting
    ax.set_xlabel(feature_1, fontsize=12)
    ax.set_ylabel(feature_2, fontsize=12)
    ax.set_title(
        f'Correlation: {feature_1} vs {feature_2}\n' +
        f'r = {r:.3f}, p = {p_value:.2e}, n = {len(plot_df_clean)}',
        fontsize=12,
        pad=15
    )
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output directory specified
    output_filename = f"correlation_{feature_1}_vs_{feature_2}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename}.pdf")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved plot to {output_path}")
    
    # Display plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Log statistics
    log.info(f"Correlation statistics:")
    log.info(f"  Features: {feature_1} vs {feature_2}")
    log.info(f"  Pearson r: {r:.4f}")
    log.info(f"  P-value: {p_value:.2e}")
    log.info(f"  N samples: {len(plot_df_clean)}")
    
    return

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
    calculate_r2: bool = False,
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
    calculate_r2 : bool, default False
        If True, also compute R-squared values by fitting a linear regression
        line to each feature pair. Adds an 'r_squared' column to output.

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
            max_iter=500
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
    
    # Calculate R-squared if requested
    if calculate_r2:
        log.info("Computing R-squared values for feature pairs...")
        r2_values = []
        
        for idx, row in df.iterrows():
            feat1 = row['feature_1']
            feat2 = row['feature_2']
            
            # Get feature vectors
            x = data.loc[feat1].values
            y = data.loc[feat2].values
            
            # Remove NaN values
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 3:
                # Not enough points for meaningful regression
                r2_values.append(np.nan)
                continue
            
            # Fit linear regression: y = mx + b
            # Using least squares: y = X @ beta, where X = [ones, x]
            X_design = np.vstack([np.ones(len(x_clean)), x_clean]).T
            beta, residuals, rank, s = np.linalg.lstsq(X_design, y_clean, rcond=None)
            
            # Calculate R-squared
            y_pred = X_design @ beta
            ss_res = np.sum((y_clean - y_pred) ** 2)  # Residual sum of squares
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)  # Total sum of squares
            
            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
            else:
                r2 = 0.0  # Constant y values
            
            r2_values.append(r2)
        
        df['r_squared'] = r2_values
        log.info(f"R-squared computation complete. Mean R²: {np.nanmean(r2_values):.4f}")
    
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

def _recursively_split_large_modules(
    modules: List[Tuple[str, nx.Graph]],
    method: str,
    max_module_size: int,
    max_depth: int = 5,
    _depth: int = 0,
    **kwargs,
) -> List[Tuple[str, nx.Graph]]:
    """Recursively re-cluster any module exceeding `max_module_size`.

    Repeatedly applies `_detect_submodules(method, **kwargs)` to oversized
    modules until every resulting module is at or below `max_module_size`,
    the module can no longer be split (a single re-clustering pass returns
    only 1 community, i.e. no further structure to find), or `max_depth`
    recursive splits have been attempted (safety valve against pathological
    cases, e.g. a dense clique that never subdivides).

    Parameters
    ----------
    modules : list of (name, subgraph)
        Output of `_detect_submodules`.
    method : str
        Same `_detect_submodules` method to use for re-splitting oversized
        modules. Typically "louvain" or "leiden" (methods with a tunable
        resolution/granularity knob); other methods will simply be
        re-invoked with the same fixed behavior each time, which may not
        produce a different split on retry.
    max_module_size : int
        Hard cap on the number of nodes allowed per final module.
    max_depth : int
        Maximum recursion depth per oversized module, to guarantee termination.
    **kwargs
        Forwarded to `_detect_submodules` on each re-clustering attempt
        (e.g. `resolution=2.0` for louvain/leiden).

    Returns
    -------
    list of (name, subgraph)
        Final module list, re-numbered sequentially as "submodule_1", "submodule_2", ...
    """
    final_modules: List[Tuple[str, nx.Graph]] = []

    for name, subgraph in modules:
        if subgraph.number_of_nodes() <= max_module_size or _depth >= max_depth:
            final_modules.append((name, subgraph))
            continue

        sub_result = _detect_submodules(subgraph, method=method, **kwargs)

        if len(sub_result) <= 1:
            # Method found no further structure to split on -- stop recursing
            # on this branch even though it's still oversized.
            log.warning(
                f"Module '{name}' has {subgraph.number_of_nodes()} nodes "
                f"(exceeds max_module_size={max_module_size}) but could not be "
                f"split further by method='{method}' with the given kwargs."
            )
            final_modules.append((name, subgraph))
            continue

        # Recurse in case any of the newly split pieces are STILL too large
        final_modules.extend(
            _recursively_split_large_modules(
                sub_result, method=method, max_module_size=max_module_size,
                max_depth=max_depth, _depth=_depth + 1, **kwargs,
            )
        )

    # Re-number sequentially for clean, final submodule names
    return [(f"submodule_{i+1}", g) for i, (_, g) in enumerate(final_modules)]

def _detect_submodules(
    G: nx.Graph,
    method: str,
    max_module_size: int | None = None,
    split_max_depth: int = 5,
    **kwargs,
) -> List[Tuple[str, nx.Graph]]:
    """
    Returns a list of (module_name, subgraph) tuples.
    Supported ``method`` values:
        * "subgraphs" - simple connected components
        * "louvain" - python-louvain (supports `resolution` kwarg; >1.0 = more, smaller modules)
        * "leiden" - leidenalg (requires igraph; supports `resolution` kwarg via
          RBConfigurationVertexPartition by default, or pass `partition_type` explicitly)
        * "k_clique" - k-clique communities
        * "greedy_modularity" - greedy modularity maximization
        * "label_propagation" - asynchronous label propagation
        * "girvan_newman" - Girvan-Newman method

    ``kwargs`` are forwarded to the specific implementation.

    Parameters
    ----------
    max_module_size : int, optional
        If set, any module larger than this is recursively re-clustered
        (using the same `method`/`kwargs`) until every module is at or
        below this size or can no longer be subdivided. Use this to enforce
        a hard granularity cap when resolution tuning alone isn't enough
        (e.g. due to modularity's resolution-limit on very large modules).
    split_max_depth : int
        Max recursion depth per oversized module when `max_module_size` is set.
    """
    if method == "subgraphs":
        comps = nx.connected_components(G)
        result = [(f"submodule_{i+1}", G.subgraph(c).copy())
                  for i, c in enumerate(comps)]

    elif method == "louvain":
        resolution = kwargs.get("resolution", 1.0)
        G_abs = G.copy()
        for u, v, data in G_abs.edges(data=True):
            if 'weight' in data:
                data['weight'] = abs(data['weight'])
        partition = community_louvain.best_partition(G_abs, weight="weight", resolution=resolution)
        modules: Dict[int, List[str]] = {}
        for node, comm in partition.items():
            modules.setdefault(comm, []).append(node)
        result = [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                  for i, (comm, nodes) in enumerate(sorted(modules.items()))]

    elif method == "leiden":
        resolution = kwargs.get("resolution", 1.0)
        ig_g = ig.Graph.from_networkx(G)
        node_names = list(G.nodes())
        ig_g.vs["name"] = node_names
        partition_type = kwargs.get("partition_type", leidenalg.RBConfigurationVertexPartition)
        partition = leidenalg.find_partition(
            ig_g, partition_type, weights="weight", resolution_parameter=resolution
        )
        result = []
        for i, community in enumerate(partition):
            nodes = [ig_g.vs[idx]["name"] for idx in community]
            result.append((f"submodule_{i+1}", G.subgraph(nodes).copy()))

    elif method == "k_clique":
        k: int = kwargs.get("k", 3)
        communities = nx.community.k_clique_communities(G, k)
        result = [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                  for i, nodes in enumerate(communities)]

    elif method == "greedy_modularity":
        weight = kwargs.get("weight", "weight")
        resolution = kwargs.get("resolution", 1)
        cutoff = kwargs.get("cutoff", 1)
        best_n = kwargs.get("best_n", None)
        communities = nx.community.greedy_modularity_communities(
            G, weight=weight, resolution=resolution, cutoff=cutoff, best_n=best_n
        )
        result = [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                  for i, nodes in enumerate(communities)]

    elif method == "label_propagation":
        weight = kwargs.get("weight", "weight")
        seed = kwargs.get("seed", None)
        communities = nx.community.asyn_lpa_communities(G, weight=weight, seed=seed)
        result = [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                  for i, nodes in enumerate(communities)]

    elif method == "girvan_newman":
        level: int = kwargs.get("level", 0)
        most_valuable_edge = kwargs.get("most_valuable_edge", None)
        communities_generator = nx.community.girvan_newman(G, most_valuable_edge=most_valuable_edge)
        for i, communities in enumerate(communities_generator):
            if i == level:
                break
        result = [(f"submodule_{i+1}", G.subgraph(nodes).copy())
                  for i, nodes in enumerate(communities)]

    else:
        valid_methods = [
            "subgraphs", "louvain", "leiden",
            "k_clique", "greedy_modularity",
            "label_propagation", "girvan_newman"
        ]
        raise ValueError(
            f"Invalid submodule method '{method}'. "
            f"Choose from: {', '.join(valid_methods)}"
        )

    if max_module_size is not None:
        result = _recursively_split_large_modules(
            result, method=method, max_module_size=max_module_size,
            max_depth=split_max_depth, **kwargs,
        )

    return result

def plot_correlation_network(
    corr_table: pd.DataFrame,
    integrated_data: pd.DataFrame,
    integrated_metadata: pd.DataFrame,
    output_dir: str,
    output_filenames: Dict[str, str],
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    submodule_mode: str = "louvain",
    module_resolution: float = 1.0,
    max_module_size: int = 100,
    network_layout: str = None,
) -> None:
    """
    Build a correlation graph from a long-format table, optionally
    detect submodules (connected components, Louvain, Leiden)
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
            resolution=module_resolution,
            max_module_size=max_module_size
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
    log.info("Rendering interactive network in notebook…")
    color_attr = "submodule_color" if submodule_mode != "none" else "datatype_color"
    log.info("Pre-computing network layout...")
    widget = _nx_to_plotly_widget(
        G,
        node_color_attr=color_attr,
        node_size_attr="node_size",
        layout=network_layout,
        seed=1111,
    )
    display(widget)

    # Add unified 'group' column as alias for 'submodule' for downstream compatibility
    if 'submodule' in node_table.columns:
        node_table['group'] = node_table['submodule']
        write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")

    return node_table, edge_table


def group_features_hierarchical(
    data: pd.DataFrame,
    distance_metric: str = "correlation",
    linkage_method: str = "average",
    height_cutoff: float = 0.3,
    output_dir: str = None,
    output_filenames: Dict[str, str] = None,
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    integrated_data: pd.DataFrame = None,
    integrated_metadata: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group features by hierarchical clustering of their LFC vectors.

    Parameters
    ----------
    data : pd.DataFrame
        LFC matrix with features as rows and comparisons as columns.
    distance_metric : str
        scipy pdist metric, default "correlation" (= 1 - Pearson).
    linkage_method : str
        scipy linkage method, default "average".
    height_cutoff : float
        fcluster distance threshold; features within this distance cluster
        together. Default 0.3.
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys "node_table" and "edge_table".
    datasets : List
        List of dataset objects with a .dataset_name attribute (used for
        node coloring).
    annotation_df : pd.DataFrame, optional
        Feature annotation table; merged on feature ID if provided.
    integrated_data : pd.DataFrame
        Not used for clustering; accepted for API consistency.
    integrated_metadata : pd.DataFrame
        Not used for clustering; accepted for API consistency.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (node_table, edge_table) where node_table is indexed by feature ID
        with a 'group' column, and edge_table is empty.
    """
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster

    log.info("Grouping features by hierarchical clustering...")

    # Drop non-numeric columns, handle NaN
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.loc[~numeric_data.isna().all(axis=1)]
    numeric_data = numeric_data.fillna(0)

    # Compute pairwise distances and run linkage
    dist = pdist(numeric_data.values, metric=distance_metric)
    Z = linkage(dist, method=linkage_method)
    labels = fcluster(Z, t=height_cutoff, criterion='distance')

    # Build group series
    group_series = pd.Series(
        [f"group_{lbl}" for lbl in labels],
        index=numeric_data.index,
        name="group",
    )

    # Determine prefix color/shape maps
    feature_ids = numeric_data.index.tolist()
    all_prefixes = [ds.dataset_name + "_" for ds in (datasets or [])]
    present_prefixes = [p for p in all_prefixes if any(f.startswith(p) for f in feature_ids)]
    color_map, shape_map = _make_prefix_maps(present_prefixes)

    # Assign color and shape per feature
    def _get_color(fid):
        for p in present_prefixes:
            if fid.startswith(p):
                return color_map.get(p, "gray")
        return "gray"

    def _get_shape(fid):
        for p in present_prefixes:
            if fid.startswith(p):
                return shape_map.get(p, "circle")
        return "circle"

    node_table = pd.DataFrame({
        "group": group_series,
        "datatype_color": [_get_color(f) for f in feature_ids],
        "datatype_shape": [_get_shape(f) for f in feature_ids],
    }, index=feature_ids)
    node_table.index.name = "node_id"

    # Merge annotation columns if provided
    if annotation_df is not None and not annotation_df.empty:
        ann = annotation_df.copy()
        if 'feature_id' in ann.columns:
            ann = ann.set_index('feature_id')
        ann_cols = [c for c in ann.columns if c not in node_table.columns]
        node_table = node_table.join(ann[ann_cols], how='left')

    n_groups = node_table['group'].nunique()
    log.info(f"\tHierarchical clustering found {n_groups} groups across {len(node_table)} features.")

    # Write outputs
    write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
    empty_edge_table = pd.DataFrame(columns=['source', 'target', 'weight'])
    write_integration_file(data=empty_edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")

    return node_table, empty_edge_table


def group_features_hdbscan(
    data: pd.DataFrame,
    metric: str = "euclidean",
    min_cluster_size: int = 5,
    min_samples: int = 3,
    output_dir: str = None,
    output_filenames: Dict[str, str] = None,
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    integrated_data: pd.DataFrame = None,
    integrated_metadata: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group features using HDBSCAN density-based clustering on their LFC vectors.

    Parameters
    ----------
    data : pd.DataFrame
        LFC matrix with features as rows and comparisons as columns.
    metric : str
        HDBSCAN distance metric, default "euclidean". If "correlation" is
        passed, a correlation distance matrix is precomputed and
        metric="precomputed" is used internally.
    min_cluster_size : int
        Minimum group size, default 5.
    min_samples : int
        Core point threshold, default 3.
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys "node_table" and "edge_table".
    datasets : List
        List of dataset objects with a .dataset_name attribute.
    annotation_df : pd.DataFrame, optional
        Feature annotation table; merged on feature ID if provided.
    integrated_data : pd.DataFrame
        Not used for clustering; accepted for API consistency.
    integrated_metadata : pd.DataFrame
        Not used for clustering; accepted for API consistency.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (node_table, edge_table) where node_table is indexed by feature ID
        with a 'group' column, and edge_table is empty.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "The 'hdbscan' package is required for group_features_hdbscan(). "
            "Install it with: pip install hdbscan"
        )

    from scipy.spatial.distance import pdist, squareform

    log.info("Grouping features by HDBSCAN clustering...")

    # Drop non-numeric columns, handle NaN
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.fillna(0)

    # Handle correlation metric (not natively supported by HDBSCAN)
    if metric == "correlation":
        X = squareform(pdist(numeric_data.values, metric='correlation'))
        hdbscan_metric = "precomputed"
    else:
        X = numeric_data.values
        hdbscan_metric = metric

    # Run HDBSCAN
    raw_labels = hdbscan.HDBSCAN(
        metric=hdbscan_metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    ).fit_predict(X)

    # Map labels: -1 → "noise", others → "group_{label+1}"
    group_labels = [
        "noise" if lbl == -1 else f"group_{lbl + 1}"
        for lbl in raw_labels
    ]

    feature_ids = numeric_data.index.tolist()

    # Determine prefix color/shape maps
    all_prefixes = [ds.dataset_name + "_" for ds in (datasets or [])]
    present_prefixes = [p for p in all_prefixes if any(f.startswith(p) for f in feature_ids)]
    color_map, shape_map = _make_prefix_maps(present_prefixes)

    def _get_color(fid):
        for p in present_prefixes:
            if fid.startswith(p):
                return color_map.get(p, "gray")
        return "gray"

    def _get_shape(fid):
        for p in present_prefixes:
            if fid.startswith(p):
                return shape_map.get(p, "circle")
        return "circle"

    node_table = pd.DataFrame({
        "group": group_labels,
        "datatype_color": [_get_color(f) for f in feature_ids],
        "datatype_shape": [_get_shape(f) for f in feature_ids],
    }, index=feature_ids)
    node_table.index.name = "node_id"

    # Merge annotation columns if provided
    if annotation_df is not None and not annotation_df.empty:
        ann = annotation_df.copy()
        if 'feature_id' in ann.columns:
            ann = ann.set_index('feature_id')
        ann_cols = [c for c in ann.columns if c not in node_table.columns]
        node_table = node_table.join(ann[ann_cols], how='left')

    n_groups = sum(1 for g in group_labels if g != "noise")
    n_noise = sum(1 for g in group_labels if g == "noise")
    unique_groups = len(set(g for g in group_labels if g != "noise"))
    log.info(f"\tHDBSCAN found {unique_groups} groups across {n_groups} features; {n_noise} features assigned to noise.")

    # Write outputs
    write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
    empty_edge_table = pd.DataFrame(columns=['source', 'target', 'weight'])
    write_integration_file(data=empty_edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")

    return node_table, empty_edge_table


def group_features(
    data: pd.DataFrame,
    method: str,
    method_params: Dict,
    output_dir: str,
    output_filenames: Dict[str, str],
    datasets: List,
    annotation_df: Optional[pd.DataFrame],
    integrated_data: pd.DataFrame,
    integrated_metadata: pd.DataFrame,
    feature_correlation_table: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified dispatcher that routes to the correct feature-grouping function.

    Parameters
    ----------
    data : pd.DataFrame
        LFC matrix with features as rows and comparisons as columns.
    method : str
        One of "network_modules", "hierarchical_clustering", "hdbscan".
    method_params : Dict
        Sub-block from config for the selected method (e.g. all keys under
        ``network_modules:`` or ``hierarchical_clustering:``).
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys "node_table" and "edge_table".
    datasets : List
        List of dataset objects with a .dataset_name attribute.
    annotation_df : pd.DataFrame, optional
        Feature annotation table.
    integrated_data : pd.DataFrame
        Full integrated data matrix (passed through for API consistency).
    integrated_metadata : pd.DataFrame
        Integrated metadata (passed through for API consistency).
    feature_correlation_table : pd.DataFrame, optional
        Pre-computed correlation table required when method="network_modules".

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (node_table, edge_table) produced by the selected grouping method.

    Raises
    ------
    ValueError
        If an unknown method name is supplied.
    """
    if method == "network_modules":
        node_table, edge_table = plot_correlation_network(
            corr_table=feature_correlation_table,
            integrated_data=integrated_data,
            integrated_metadata=integrated_metadata,
            output_dir=output_dir,
            output_filenames=output_filenames,
            datasets=datasets,
            annotation_df=annotation_df,
            submodule_mode=method_params.get('submodule_mode', 'louvain'),
            network_layout=method_params.get('network_layout', None),
        )
        # Add 'group' column as alias for 'submodule' (plot_correlation_network
        # already does this, but guard here for safety)
        if 'submodule' in node_table.columns and 'group' not in node_table.columns:
            node_table['group'] = node_table['submodule']
        return node_table, edge_table

    elif method == "hierarchical_clustering":
        return group_features_hierarchical(
            data=data,
            distance_metric=method_params.get('distance_metric', 'correlation'),
            linkage_method=method_params.get('linkage_method', 'average'),
            height_cutoff=method_params.get('height_cutoff', 0.3),
            output_dir=output_dir,
            output_filenames=output_filenames,
            datasets=datasets,
            annotation_df=annotation_df,
            integrated_data=integrated_data,
            integrated_metadata=integrated_metadata,
        )

    elif method == "hdbscan":
        return group_features_hdbscan(
            data=data,
            metric=method_params.get('metric', 'euclidean'),
            min_cluster_size=method_params.get('min_cluster_size', 5),
            min_samples=method_params.get('min_samples', 3),
            output_dir=output_dir,
            output_filenames=output_filenames,
            datasets=datasets,
            annotation_df=annotation_df,
            integrated_data=integrated_data,
            integrated_metadata=integrated_metadata,
        )

    else:
        raise ValueError(
            f"Unknown feature_grouping method '{method}'. "
            "Choose from: network_modules, hierarchical_clustering, hdbscan"
        )


def display_existing_network(
    graph_file: str,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    network_layout: str = None
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
    network_layout : str, optional
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
            layout=network_layout,
            seed=1111,
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
    seed=1111,
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

    # Build edge trace WITH HOVER TEXT INCLUDING CORRELATION
    log.info("Building edge traces...")
    edge_x, edge_y = [], []
    edge_hover_text = []
    
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        
        # Get correlation weight from edge data
        edge_data = G.get_edge_data(u, v)
        correlation = edge_data.get('weight', 0.0) if edge_data else 0.0
        
        # Create hover text with correlation value
        edge_hover_text.append(f"{u} : {v} ({correlation:.3f})")
        # Add two more None entries to match the x/y coordinate structure
        edge_hover_text.extend([None, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="text",
        hovertext=edge_hover_text,
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


###################
# Add MAGI2 evidence
###################

def parse_magi_table(
    annotation_table: pd.DataFrame,
    magi_raw_dir: str,
    tx_raw_dir: str,
    score_cutoff: float,
    output_dir: str,
    output_filename: str
) -> pd.DataFrame:
    """
    Parse MAGI2 output files and return a DataFrame with metabolite and transcript IDs (names)
    that match existing feature IDs in the integrated dataset.
    
    Handles multiple annotations per feature stored as ;;-separated strings.
    Uses GFF file to map protein IDs to gene IDs for transcripts.
    """

    magi_df = pd.read_csv(os.path.join(magi_raw_dir, "magi_combined_results.csv"))
    magi_df = magi_df[['original_compound', 'gene_ID', 'rhea_ID_r2g', 'rhea_ID_g2r', 'MAGI_score']]
    
    # Ensure all columns are strings except MAGI_score
    for col in magi_df.columns:
        if col != 'MAGI_score':
            magi_df[col] = magi_df[col].astype(str)
    
    # Ensure MAGI_score is numeric
    magi_df['MAGI_score'] = pd.to_numeric(magi_df['MAGI_score'], errors='coerce')
    
    # Filter by score cutoff
    log.info(f"MAGI entries before score cutoff ({score_cutoff}): {len(magi_df)}")
    magi_df = magi_df[magi_df['MAGI_score'] >= score_cutoff].reset_index(drop=True)
    log.info(f"MAGI entries after score cutoff ({score_cutoff}): {len(magi_df)}")
    
    # Parse gene_ID to extract protein_id (3rd pipe-separated element)
    log.info("Parsing gene_ID to extract protein IDs...")
    magi_df['parsed_protein_id'] = magi_df['gene_ID'].str.split('|').str[2]
    
    # Find GFF file in the same directory structure as annotation table
    gff_files = []
    for pattern in ['*.gff3', '*.gff3.gz', '*.gff', '*.gff.gz']:
        gff_files.extend(glob.glob(os.path.join(tx_raw_dir, '**', pattern), recursive=True))
    if not gff_files:
        raise ValueError(f"No GFF3 file found in {tx_raw_dir} or subdirectories. "
                        "GFF file is required for mapping protein IDs to gene IDs.")
    gff_file = gff_files[0]  # Use first GFF file found
    log.info(f"Using GFF file: {os.path.basename(gff_file)}")
    
    # Parse GFF3 file to create protein_id -> gene_id mapping
    log.info("Parsing GFF file to create protein ID to gene ID mappings...")
    protein_to_gene = {}
    
    open_func = gzip.open if gff_file.endswith('.gz') else open
    with open_func(gff_file, 'rt') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            # Only process 'gene' features
            if fields[2] != 'gene':
                continue
            
            attributes = fields[8]
            
            # Extract gene ID (the feature ID we want)
            gene_match = re.search(r'ID=([^;]+)', attributes)
            if not gene_match:
                continue
            gene_id = gene_match.group(1)
            
            # Extract protein ID
            protein_match = re.search(r'proteinId=([^;]+)', attributes)
            if not protein_match:
                continue
            protein_id = protein_match.group(1)
            
            # Store mapping
            protein_to_gene[protein_id] = gene_id
    
    log.info(f"Found {len(protein_to_gene)} protein-to-gene mappings in GFF file")
    
    # Map protein IDs to gene IDs (feature IDs)
    def map_protein_to_gene(protein_id):
        if protein_id in protein_to_gene:
            gene_id = protein_to_gene[protein_id]
            # Add 'tx_' prefix if not already present
            if not gene_id.startswith('tx_'):
                return f'tx_{gene_id}'
            return gene_id
        return None
    
    log.info("Mapping MAGI protein IDs to transcript feature IDs...")
    magi_df['gene_feature_id'] = magi_df['parsed_protein_id'].apply(map_protein_to_gene)
    
    # Now handle metabolite mapping using annotation table
    log.info("Creating SMILES to feature ID mappings from annotation table...")
    smiles_to_features = {}
    
    # Ensure annotation_table columns are strings
    for col in annotation_table.columns:
        if col not in ['feature_id']:
            annotation_table[col] = annotation_table[col].astype(str)
    
    for idx, row in annotation_table.iterrows():
        feature_id = str(row['feature_id'])
        smiles_str = str(row.get('mx_Smiles', ''))
        
        # Skip if no SMILES data
        if smiles_str in ['nan', 'Unassigned', 'NA', '', 'None']:
            continue
        
        # Split on ;; and add each SMILES to the mapping
        for smiles in smiles_str.split(';;'):
            smiles = smiles.strip()
            if smiles and smiles not in ['nan', 'Unassigned', 'NA', '']:
                if smiles not in smiles_to_features:
                    smiles_to_features[smiles] = []
                smiles_to_features[smiles].append(feature_id)
    
    log.info(f"Created {len(smiles_to_features)} unique SMILES mappings")
    
    # Map SMILES to feature IDs (with ;;-separated values)
    def map_smiles_to_features(smiles):
        if smiles in smiles_to_features:
            return ';;'.join(smiles_to_features[smiles])
        return None
    
    log.info("Mapping MAGI SMILES to metabolite feature IDs...")
    magi_df['compound_feature_id'] = magi_df['original_compound'].apply(map_smiles_to_features)
    
    # Drop the intermediate column
    magi_df = magi_df.drop(columns=['parsed_protein_id'])
    
    # Reorder columns for clarity
    cols = ['compound_feature_id', 'gene_feature_id', 'original_compound', 'gene_ID', 
            'rhea_ID_r2g', 'rhea_ID_g2r', 'MAGI_score']
    magi_df = magi_df[cols]
    
    # Final type check: ensure all columns are strings except MAGI_score
    for col in magi_df.columns:
        if col != 'MAGI_score':
            magi_df[col] = magi_df[col].astype(str)
    
    # Log summary statistics
    n_total = len(magi_df)
    n_mapped_genes = magi_df['gene_feature_id'].notna().sum()
    n_mapped_compounds = magi_df['compound_feature_id'].notna().sum()
    n_both_mapped = ((magi_df['gene_feature_id'].notna()) & 
                     (magi_df['compound_feature_id'].notna())).sum()
    
    # Count multiple mappings
    n_multi_genes = magi_df['gene_feature_id'].str.contains(';;', na=False).sum()
    n_multi_compounds = magi_df['compound_feature_id'].str.contains(';;', na=False).sum()
    
    log.info(f"MAGI mapping summary:")
    log.info(f"  Total MAGI predictions: {n_total}")
    log.info(f"  Genes mapped to features: {n_mapped_genes} ({n_mapped_genes/n_total*100:.1f}%)")
    log.info(f"    With multiple mappings: {n_multi_genes}")
    log.info(f"  Compounds mapped to features: {n_mapped_compounds} ({n_mapped_compounds/n_total*100:.1f}%)")
    log.info(f"    With multiple mappings: {n_multi_compounds}")
    log.info(f"  Both mapped: {n_both_mapped} ({n_both_mapped/n_total*100:.1f}%)")
    
    # Save the mapped table
    write_integration_file(
        data=magi_df,
        output_dir=output_dir,
        filename=output_filename,
        indexing=True
    )
    
    return magi_df

def add_magi_scores_to_correlation_table(
    correlation_table: pd.DataFrame,
    scatter_correlation: str,
    magi_df: pd.DataFrame,
    output_dir: str = None,
    output_filename: str = "correlation_table_with_magi_scores"
) -> pd.DataFrame:
    """
    Add MAGI scores to a feature correlation table based on gene-compound pairs.
    Only considers bipartite correlations (between different dataset types).
    
    For each correlated feature pair from different datasets, checks if they appear 
    in the MAGI predictions and adds the MAGI_score if found. Handles multiple 
    feature IDs stored as ;;-separated strings.
    
    Args:
        correlation_table: DataFrame with columns ['feature_1', 'feature_2', 'correlation']
        magi_df: DataFrame from parse_magi_table() with MAGI predictions and feature mappings
        output_dir: Directory to save the output table
        output_filename: Filename for the output table
        
    Returns:
        DataFrame with only bipartite correlations and added 'MAGI_score' column
    """
    
    log.info("Adding MAGI scores to correlation table...")
    log.info(f"  Total correlation pairs: {len(correlation_table):,}")
    log.info(f"  MAGI predictions: {len(magi_df):,}")
    
    # Create a copy to avoid modifying the original
    result_df = correlation_table.copy()
    
    # Ensure correlation_table columns are strings except correlation
    for col in result_df.columns:
        if col not in ['correlation', 'MAGI_score']:
            result_df[col] = result_df[col].astype(str)
    
    # Ensure correlation column is numeric
    result_df['correlation'] = pd.to_numeric(result_df['correlation'], errors='coerce')
    
    # Filter for bipartite correlations only (different dataset prefixes)
    # Extract dataset prefix (everything before the first underscore)
    result_df['dataset_1'] = result_df['feature_1'].str.split('_').str[0]
    result_df['dataset_2'] = result_df['feature_2'].str.split('_').str[0]
    
    # Keep only pairs where datasets differ
    bipartite_mask = result_df['dataset_1'] != result_df['dataset_2']
    result_df = result_df[bipartite_mask].copy()
    
    # Drop the temporary dataset columns
    result_df = result_df.drop(columns=['dataset_1', 'dataset_2'])
    
    log.info(f"  Bipartite correlation pairs (different datasets): {len(result_df):,}")
    
    # Initialize MAGI_score column with NaN
    result_df['MAGI_score'] = np.nan

    # Ensure magi_df columns are strings except MAGI_score
    for col in magi_df.columns:
        if col != 'MAGI_score':
            magi_df[col] = magi_df[col].astype(str)
    
    # Ensure MAGI_score is numeric
    magi_df['MAGI_score'] = pd.to_numeric(magi_df['MAGI_score'], errors='coerce')
    
    # Build lookup dictionaries for efficient matching
    # Key: (gene_feature_id, compound_feature_id) -> MAGI_score
    magi_lookup = {}
    
    for idx, row in magi_df.iterrows():
        gene_ids = row['gene_feature_id']
        compound_ids = row['compound_feature_id']
        magi_score = row['MAGI_score']
        
        # Skip if either mapping is missing
        if pd.isna(gene_ids) or pd.isna(compound_ids) or gene_ids == 'nan' or compound_ids == 'nan':
            continue
        
        # Handle ;;-separated multiple IDs
        gene_id_list = [g.strip() for g in str(gene_ids).split(';;') if g.strip() and g.strip() != 'nan']
        compound_id_list = [c.strip() for c in str(compound_ids).split(';;') if c.strip() and c.strip() != 'nan']
        
        # Create all possible gene-compound pairs
        for gene_id in gene_id_list:
            for compound_id in compound_id_list:
                # Store both orderings since we don't know which feature is which
                magi_lookup[(gene_id, compound_id)] = magi_score
                magi_lookup[(compound_id, gene_id)] = magi_score
    
    log.info(f"  Built MAGI lookup with {len(magi_lookup):,} gene-compound pairs")
    # Find only valid non-None pairs
    valid_magi_lookup = {k: v for k, v in magi_lookup.items() if k[0] != 'None' and k[1] != 'None'}
    log.info(f"  Valid MAGI lookup pairs (excluding 'None'): {len(valid_magi_lookup):,}")
    magi_lookup = valid_magi_lookup
    
    # Match correlation pairs to MAGI predictions
    matches_found = 0
    
    for idx, row in result_df.iterrows():
        feature_1 = row['feature_1']
        feature_2 = row['feature_2']
        
        # Check if pair exists in MAGI lookup (either ordering)
        pair_key = (feature_1, feature_2)
        
        if pair_key in magi_lookup:
            result_df.at[idx, 'MAGI_score'] = magi_lookup[pair_key]
            matches_found += 1
    
    # Summary statistics
    n_with_magi = result_df['MAGI_score'].notna().sum()
    pct_with_magi = (n_with_magi / len(result_df) * 100) if len(result_df) > 0 else 0
    
    log.info(f"MAGI score matching summary (bipartite pairs only):")
    log.info(f"  Bipartite correlation pairs with MAGI support: {n_with_magi:,} ({pct_with_magi:.1f}%)")
    
    if n_with_magi > 0:
        # Show distribution of MAGI scores for matched pairs
        magi_scores = result_df['MAGI_score'].dropna()
        log.info(f"  MAGI score range: {magi_scores.min():.3f} - {magi_scores.max():.3f}")
        log.info(f"  MAGI score mean: {magi_scores.mean():.3f}")
        log.info(f"  MAGI score median: {magi_scores.median():.3f}")
    
    # Save results if output directory provided
    if output_dir:
        write_integration_file(
            data=result_df,
            output_dir=output_dir,
            filename=output_filename,
            indexing=True
        )
    
    # Create scatter plot if requested
    if n_with_magi > 0:
        log.info("Creating MAGI vs. Correlation scatter plot...")
        plot_magi_vs_correlation(
            correlation_table=result_df,
            corr_values=scatter_correlation,
            output_dir=output_dir,
            output_filename=f"{output_filename}_scatter",
            show_plot=True
        )
    
    return result_df

def plot_magi_vs_correlation(
    correlation_table: pd.DataFrame,
    corr_values: str = "correlation",
    output_dir: str = None,
    output_filename: str = "magi_vs_correlation_scatter",
    show_plot: bool = True
) -> Dict[str, float]:
    """
    Plot MAGI scores vs correlation scores with a fitted regression line.
    
    Args:
        correlation_table: DataFrame with 'correlation' and 'MAGI_score' columns
        output_dir: Directory to save the plot
        output_filename: Filename for the plot (without extension)
        show_plot: Whether to display the plot inline
        
    Returns:
        Dictionary with regression statistics (slope, intercept, r_squared, p_value)
    """
    
    # Filter to only rows with MAGI scores
    data_with_magi = correlation_table.dropna(subset=['MAGI_score'])
    
    if data_with_magi.empty:
        log.warning("No MAGI scores found for plotting")
        return {}
    
    log.info(f"Plotting {len(data_with_magi):,} correlations with MAGI scores")
    
    # Extract data
    x = data_with_magi[corr_values].values
    x = x.astype(float)
    y = data_with_magi['MAGI_score'].values
    y = y.astype(float)
    
    # Fit linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5, s=20, c='#440154', edgecolors='none')
    
    # Fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, 
            label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_squared:.3f}, p = {p_value:.2e}')
    
    # Labels and styling
    ax.set_xlabel('Correlation Score', fontsize=12)
    ax.set_ylabel('MAGI Score', fontsize=12)
    ax.set_title('MAGI Score vs. Correlation Score', fontsize=14, pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add density information using hexbin overlay
    ax2 = ax.twinx()
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_path = f"{output_dir}/{output_filename}.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        log.info(f"Saved plot to {output_path}")
    
    # Display plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # Log statistics
    log.info(f"Regression statistics:")
    log.info(f"  Slope: {slope:.4f}")
    log.info(f"  Intercept: {intercept:.4f}")
    log.info(f"  R²: {r_squared:.4f}")
    log.info(f"  P-value: {p_value:.2e}")
    log.info(f"  Std Error: {std_err:.4f}")
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': len(data_with_magi)
    }

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
    filtered_mx: bool = True,
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
    filtered_mx: bool = True,
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
    filtered_mx: bool = True,
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
    tx_data_files_sep = "\t"
    if not tx_data_files:
        tx_data_pattern = f"{input_dir}/{type}.csv"
        tx_data_files = glob.glob(os.path.expanduser(tx_data_pattern))
        tx_data_files_sep = ","
    
    if tx_data_files:
        if len(tx_data_files) > 1:
            log.info(f"Multiple TX data files found matching pattern {tx_data_pattern}.")
            log.info("Please specify the correct file.")
            return None
        tx_data_filename = tx_data_files[0]  # Assuming you want the first (and only) match
        tx_data = pd.read_csv(tx_data_filename, sep=tx_data_files_sep)
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

def _process_plant_annotations(
    raw_data_dir: str,
    output_dir: str,
    output_filename: str
) -> pd.DataFrame:
    """
    Process plant (Phytozome) annotation files and merge into a single table.

    Plant data has a single annotation file called ``kegg_annotation_table.tsv``
    whose columns are::

        #pacId  locusName  transcriptName  peptideName  Pfam  Panther  KOG
        KEGG/ec  KO  GO  Best-hit-arabi-name  arabi-symbol  arabi-defline

    The ``transcriptName`` column (e.g. ``LOC_Os01g04030.1``) is used as the
    primary identifier and is mapped to ``transcriptome_id`` via the GFF3
    ``mRNA`` feature ``Name`` attribute.  ``pacId`` is stored as ``protein_id``
    (it is the numeric Phytozome protein/pacid identifier).

    Args:
        raw_data_dir (str): Directory containing ``kegg_annotation_table.tsv``
            and optionally a ``genes.gff3`` file.
        output_dir (str): Output directory for the merged annotation table.
        output_filename (str): Filename for the output annotation table.

    Returns:
        pd.DataFrame: Merged annotation table with ``transcriptome_id`` and
            annotation columns.
    """

    log.info(f"Processing plant annotations from {raw_data_dir}")

    # Plant has exactly one annotation file
    annotation_file = os.path.join(raw_data_dir, "kegg_annotation_table.tsv")

    if not os.path.isfile(annotation_file):
        log.warning(f"kegg_annotation_table.tsv not found in {raw_data_dir}")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df

    log.info(f"Found annotation file: {os.path.basename(annotation_file)}")

    df = _read_and_select_plant_annotations(annotation_file)
    if df is None or df.empty:
        log.warning(f"No valid data in {os.path.basename(annotation_file)}")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df

    # Aggregate to one row per locus (handles any duplicate locusName rows)
    agg_df = _aggregate_plant_annotations(df)
    log.info(f"  Processed {os.path.basename(annotation_file)}: {len(agg_df)} loci")

    # Fill NaN values with empty strings for consistency
    agg_df = agg_df.fillna('')

    # Map locus_name -> GFF3 gene ID= to set transcriptome_id matching raw data
    merged_df = _add_protein_id_mapping(agg_df, raw_data_dir, "plant")

    # Compress to one gene per row with semicolon-separated annotations
    final_df = _compress_annotation_table(merged_df)

    log.info(f"Final annotation table: {len(final_df)} genes with {len(final_df.columns)} annotation columns")

    # Save the merged annotation table
    write_integration_file(final_df, output_dir, output_filename, indexing=False)

    return final_df


def _read_and_select_plant_annotations(file_path: str) -> pd.DataFrame:
    """
    Read and select relevant columns from the plant Phytozome
    ``kegg_annotation_table.tsv`` annotation file.

    The file uses a ``#``-prefixed header line and tab separation.  The
    following source columns are extracted and renamed:

    ==================  =================
    Source column       Output column
    ==================  =================
    ``#pacId``          ``protein_id``
    ``transcriptName``  ``transcriptome_id``
    ``Pfam``            ``pfam_acc``
    ``Panther``         ``panther_acc``
    ``KOG``             ``kog_acc``
    ``KEGG/ec``         ``kegg_ec``
    ``KO``              ``ko_acc``
    ``GO``              ``go_acc``
    ``Best-hit-arabi-name`` ``arabi_hit``
    ``arabi-symbol``    ``arabi_symbol``
    ``arabi-defline``   ``arabi_defline``
    ==================  =================

    Args:
        file_path (str): Path to ``kegg_annotation_table.tsv``.

    Returns:
        pd.DataFrame or None: Processed annotation dataframe, or ``None`` on
            error.
    """

    try:
        df = pd.read_csv(file_path, sep='\t', comment=None, low_memory=False)

        # The header line starts with '#pacId'; strip leading '#' from column names
        df.columns = [c.lstrip('#') for c in df.columns]

        required_cols = ['pacId', 'transcriptName']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            log.warning(f"Missing required columns {missing} in {os.path.basename(file_path)}")
            return None

        # Map source columns to standardised output names; only keep columns
        # that are actually present in the file.
        col_map = {
            'pacId':               'protein_id',
            'locusName':           'locus_name',
            'transcriptName':      'transcript_name',
            'Pfam':                'pfam_acc',
            'Panther':             'panther_acc',
            'KOG':                 'kog_acc',
            'KEGG/ec':             'kegg_ec',
            'KO':                  'ko_acc',
            'GO':                  'go_acc',
            'Best-hit-arabi-name': 'arabi_hit',
            'arabi-symbol':        'arabi_symbol',
            'arabi-defline':       'arabi_defline',
        }

        available_map = {src: dst for src, dst in col_map.items() if src in df.columns}
        df = df[list(available_map.keys())].copy()
        df.rename(columns=available_map, inplace=True)

        # Ensure protein_id is stored as string (pacId is numeric in the file)
        df['protein_id'] = df['protein_id'].astype(str)

        # Drop rows where the locus name is missing (primary join key)
        df = df.dropna(subset=['locus_name'])
        df = df[df['locus_name'].astype(str).str.strip() != '']

        return df

    except Exception as e:
        log.warning(f"Error reading plant annotation file {file_path}: {e}")
        return None


def _aggregate_plant_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate plant annotations by ``locus_name``, collapsing any duplicate
    rows for the same locus into ``;;``-separated strings.

    ``locus_name`` (from the ``locusName`` column) is the gene-level identifier
    that matches the raw transcriptomics data GeneIDs after GFF3 ID mapping.

    Args:
        df (pd.DataFrame): Annotation dataframe with a ``locus_name`` column
            produced by :func:`_read_and_select_plant_annotations`.

    Returns:
        pd.DataFrame: Aggregated dataframe with one row per ``locus_name``.
    """

    def join_unique_values(series):
        """Join unique non-empty values with double semicolon."""
        unique_vals = sorted({
            str(x).strip() for x in series.dropna()
            if str(x).strip() and str(x).strip().lower() not in ['nan', 'none', '']
        })
        return ';;'.join(unique_vals) if unique_vals else ''

    annotation_columns = [col for col in df.columns if col != 'locus_name']
    aggregated_df = (
        df.groupby('locus_name')[annotation_columns]
        .agg(join_unique_values)
        .reset_index()
    )

    return aggregated_df


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
            elif genome_type == "plant":
                feature_type = 'mRNA'
                protein_attribute = 'pacid'
                display_attribute = 'Name'
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

    if genome_type == "plant":
        # For plant data the annotation table uses locusName (e.g. LOC_Os01g04030)
        # as the gene-level key, while the raw transcriptomics data uses the GFF3
        # gene feature ID= value (e.g. LOC_Os01g04030.MSUv7.0).
        # gene_to_protein here maps GFF3 gene ID= -> pacid (protein_attribute='pacid'
        # is set on mRNA features, so gene_to_protein is actually empty for plant
        # when feature_type='mRNA' was used above).
        #
        # Re-parse the GFF3 gene features to build locus_name (Name=) -> gene_id (ID=).
        locus_to_gene_id = {}
        gene_id_to_display = {}
        gff_files_plant = glob.glob(os.path.join(raw_data_dir, "*.gff3"))
        if not gff_files_plant:
            gff_files_plant = glob.glob(os.path.join(raw_data_dir, "genes.gff3"))
        if gff_files_plant:
            with open(gff_files_plant[0], 'r') as _gf:
                for _line in _gf:
                    if _line.startswith('#') or not _line.strip():
                        continue
                    _fields = _line.strip().split('\t')
                    if len(_fields) < 9 or _fields[2] != 'gene':
                        continue
                    _attrs = _fields[8]
                    _id_m = re.search(r'ID=([^;]+)', _attrs)
                    _name_m = re.search(r'Name=([^;]+)', _attrs)
                    if _id_m and _name_m:
                        _gene_id = _id_m.group(1)
                        _locus = _name_m.group(1)
                        locus_to_gene_id[_locus] = _gene_id
                        gene_id_to_display[_gene_id] = _locus

        merged_df['locus_name'] = merged_df['locus_name'].astype(str)
        # Map locus_name -> GFF3 gene ID= (the transcriptome_id used in raw data)
        merged_df['transcriptome_id'] = merged_df['locus_name'].map(locus_to_gene_id).fillna(
            merged_df['locus_name']
        )
        merged_df['display_name'] = merged_df['transcriptome_id'].map(gene_id_to_display).fillna(
            merged_df['locus_name']
        )
        # protein_id comes from pacId in the annotation file; keep as-is
        if 'protein_id' not in merged_df.columns:
            merged_df['protein_id'] = ''
    elif 'transcriptome_id' in merged_df.columns and 'protein_id' not in merged_df.columns:
        merged_df['transcriptome_id'] = merged_df['transcriptome_id'].astype(str)
        merged_df['protein_id'] = merged_df['transcriptome_id'].map(gene_to_protein).fillna('')
        merged_df['display_name'] = merged_df['transcriptome_id'].map(gene_to_product).fillna('')
    elif 'protein_id' in merged_df.columns and 'transcriptome_id' not in merged_df.columns:
        merged_df['protein_id'] = merged_df['protein_id'].astype(str)
        protein_to_gene = {str(v): k for k, v in gene_to_protein.items()}
        merged_df['transcriptome_id'] = merged_df['protein_id'].map(protein_to_gene).fillna(merged_df['protein_id'])
        merged_df['display_name'] = merged_df['transcriptome_id'].map(gene_to_product).fillna('')
    else:
        log.error("Merged dataframe either has both columns or neither - check your data structure")

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

# =============================================================================
# ModelSEED reactions: loading, placeholder pathways, EC/rxn lookup, pathway lookups
# =============================================================================

def _assign_placeholder_pathways(
    reactions_df: pd.DataFrame,
    rxn_id_col: str = "id",
    pathway_col: str = "pathways",
    compound_ids_col: str = "compound_ids",
    placeholder_source: str = "NOPATHWAY",
) -> pd.DataFrame:
    """Assign a unique placeholder pathway ID to reactions that have compounds but no pathway.

    ModelSEED reactions with a null `pathways` but a non-empty `compound_ids` list still
    represent a real biochemical link between a reaction (transcript, via EC -> rxn) and
    its compounds (metabolites, via cpd IDs). To preserve that link without inventing a
    false biological pathway name, this assigns each such reaction its own unique,
    non-descriptive pathway ID derived from the reaction ID.

    The placeholder is written in the same `"Source: id1;id2"` format used by real
    ModelSEED pathway data (see `_parse_pipe_field`), e.g. `"NOPATHWAY: NOPATHWAY_rxn40535"`,
    so it round-trips correctly through the same parsing logic as real pathway sources.

    Parameters
    ----------
    reactions_df : pd.DataFrame
        ModelSEED reactions table (e.g., loaded from modelseed_reactions.tsv).
    rxn_id_col : str
        Column containing the reaction ID (e.g. 'rxn40535').
    pathway_col : str
        Column containing pathway annotation; may contain null/'null'/'' for many rows.
    compound_ids_col : str
        Column containing a ';'-joined list of compound IDs participating in the reaction.
    placeholder_source : str
        Synthetic "source" name used for the placeholder pathway entries, so these rows
        are easy to identify/filter downstream (e.g., to exclude from biological pathway
        enrichment, or to explicitly opt in via `pathway_sources=(..., "NOPATHWAY")`).

    Returns
    -------
    pd.DataFrame
        Copy of `reactions_df` with `pathway_col` updated: rows that had no pathway but
        do have compounds get a unique placeholder entry. All other rows are left
        untouched (including rows that have neither a pathway nor compounds, which
        remain null since there's nothing to link).
    """
    df = reactions_df.copy()

    # Normalize literal "null"/empty strings to true NaN so .isna() catches them
    df[pathway_col] = df[pathway_col].replace(
        to_replace=["null", "NULL", "None", ""], value=pd.NA
    )

    has_no_pathway = df[pathway_col].isna()
    has_compounds = (
        df[compound_ids_col].notna()
        & (df[compound_ids_col].astype(str).str.strip() != "")
        & (df[compound_ids_col].astype(str).str.strip().str.lower() != "null")
    )

    needs_placeholder = has_no_pathway & has_compounds

    df.loc[needs_placeholder, pathway_col] = (
        placeholder_source + ": " + placeholder_source + "_"
        + df.loc[needs_placeholder, rxn_id_col].astype(str)
    )

    n_assigned = int(needs_placeholder.sum())
    if n_assigned:
        log.info(
            "Assigned %d unique placeholder pathway IDs (source='%s') to reactions "
            "with compounds but no annotated pathway.",
            n_assigned, placeholder_source,
        )

    return df


@lru_cache(maxsize=1)
def _get_modelseed_reactions(cache_path: Path) -> pd.DataFrame:
    """Load the ModelSEED reactions table, fetching and caching it if needed.

    Result is memoized (per `cache_path`) since this is called from several
    independent lookup builders (`_build_ec_to_rxn`, `_build_rxn_to_pathways`,
    `_build_cpd_to_pathways`, `_build_ec_to_pathways`) that would otherwise
    each re-read and re-process the same TSV from disk.
    """

    _MODELSEED_REACTIONS_URL = (
        "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
        "master/Biochemistry/reactions.tsv"
    )

    if cache_path.exists():
        log.info(f"Loading ModelSEED reactions from local cache: {cache_path}")
    else:
        log.info(f"Fetching ModelSEED reactions table from {_MODELSEED_REACTIONS_URL}")
        resp = requests.get(_MODELSEED_REACTIONS_URL, timeout=30)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        log.info(f"Saved ModelSEED reactions cache to {cache_path}")

    reactions_df = pd.read_csv(cache_path, sep="\t", low_memory=False)
    reactions_df = _assign_placeholder_pathways(reactions_df)

    return reactions_df


def _parse_pipe_field(value) -> dict[str, list[str]]:
    """Parse a ModelSEED 'aliases' or 'pathways' style field.

    Format: "Source1: id1;id2|Source2: id3;id4"
    Returns {source: [ids, ...]}.
    """
    result: dict[str, list[str]] = {}
    if not isinstance(value, str) or not value.strip():
        return result

    for entry in value.split("|"):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        source, ids = entry.split(":", 1)
        source = source.strip()
        id_list = [i.strip() for i in ids.split(";") if i.strip()]
        if id_list:
            result.setdefault(source, []).extend(id_list)
    return result


def _parse_stoichiometry(stoich) -> list[str]:
    """Extract compound IDs referenced in a ModelSEED stoichiometry string.

    Format: "coeff:cpdid:compartment_index:compartment_id:cpd_name;coeff:cpdid:...".
    Kept as a fallback for cases where a `compound_ids` column isn't available;
    prefer the `compound_ids` column directly when present (see `_build_cpd_to_pathways`).
    """
    cpd_ids: list[str] = []
    if not isinstance(stoich, str):
        return cpd_ids
    for term in stoich.split(";"):
        parts = term.split(":")
        if len(parts) >= 2:
            cpd_ids.append(parts[1].strip())
    return cpd_ids


def _build_ec_to_rxn(cache_path: Path) -> dict[str, str]:
    """Return a mapping of EC Number to semicolon-joined ModelSEED rxn IDs.

    This is the primary route for transcripts annotated with EC numbers
    (e.g. via eggNOG-mapper, InterProScan, KOfamScan --ec, PRIAM, etc.).
    """
    df = _get_modelseed_reactions(cache_path)

    if "ec_numbers" not in df.columns or "id" not in df.columns:
        log.error(
            "ModelSEED reactions TSV does not contain expected columns "
            "'id' / 'ec_numbers'. Available: %s", list(df.columns)
        )
        return {}

    ec_map: dict[str, set[str]] = {}
    for rxn_id, ec_field in zip(df["id"], df["ec_numbers"]):
        if not isinstance(ec_field, str) or not ec_field.strip():
            continue
        for ec in ec_field.split("|"):
            ec = ec.strip()
            if ec:
                ec_map.setdefault(ec, set()).add(rxn_id)

    return {ec: ";".join(sorted(rxns)) for ec, rxns in ec_map.items()}


def _build_rxn_to_pathways(cache_path: Path) -> dict[str, dict[str, set[str]]]:
    """Return rxn_id -> {pathway_source: {pathway_ids}}.

    Includes placeholder entries under source "NOPATHWAY" for reactions that
    have compounds but no real pathway annotation (see `_assign_placeholder_pathways`).
    """
    df = _get_modelseed_reactions(cache_path)
    if "pathways" not in df.columns or "id" not in df.columns:
        log.error(
            "ModelSEED reactions TSV missing 'pathways' column. Available: %s",
            list(df.columns),
        )
        return {}

    result: dict[str, dict[str, set[str]]] = {}
    for rxn_id, pw_field in zip(df["id"], df["pathways"]):
        parsed = _parse_pipe_field(pw_field)
        if parsed:
            result[rxn_id] = {src: set(ids) for src, ids in parsed.items()}
    return result


def _build_cpd_to_pathways(cache_path: Path) -> dict[str, dict[str, set[str]]]:
    """Return cpd_id -> {pathway_source: {pathway_ids}}.

    Derived by walking through every reaction a compound participates in
    and unioning the pathway annotations of those reactions. Uses the
    `compound_ids` column directly when available (simpler and avoids
    stoichiometry-string parsing edge cases); falls back to parsing
    `stoichiometry` otherwise.
    """
    df = _get_modelseed_reactions(cache_path)
    if "pathways" not in df.columns:
        log.error(
            "ModelSEED reactions TSV missing 'pathways' column. Available: %s",
            list(df.columns),
        )
        return {}

    use_compound_ids_col = "compound_ids" in df.columns

    if not use_compound_ids_col and "stoichiometry" not in df.columns:
        log.error(
            "ModelSEED reactions TSV missing both 'compound_ids' and 'stoichiometry' "
            "columns; cannot determine compound membership. Available: %s",
            list(df.columns),
        )
        return {}

    cpd_pathways: dict[str, dict[str, set[str]]] = {}
    cpd_source_col = df["compound_ids"] if use_compound_ids_col else df["stoichiometry"]

    for cpd_source_val, pw_field in zip(cpd_source_col, df["pathways"]):
        pathways = _parse_pipe_field(pw_field)
        if not pathways:
            continue

        if use_compound_ids_col:
            cpd_ids = (
                [c.strip() for c in cpd_source_val.split(";") if c.strip()]
                if isinstance(cpd_source_val, str)
                else []
            )
        else:
            cpd_ids = _parse_stoichiometry(cpd_source_val)

        for cpd_id in cpd_ids:
            entry = cpd_pathways.setdefault(cpd_id, {})
            for source, ids in pathways.items():
                entry.setdefault(source, set()).update(ids)

    return cpd_pathways


# def _build_ec_to_pathways(cache_path: Path) -> dict[str, dict[str, set[str]]]:
#     """Return ec_number -> {pathway_source: {pathway_ids}} directly.

#     Shortcut that skips the intermediate rxn_id; equivalent to composing
#     `_build_ec_to_rxn` with `_build_rxn_to_pathways`. Currently not called by
#     `add_modelseed_pathway_column` (which goes through the rxn column instead)
#     but kept as a documented, ready-to-use alternative entry point.
#     """
#     df = _get_modelseed_reactions(cache_path)
#     if "ec_numbers" not in df.columns or "pathways" not in df.columns:
#         log.error(
#             "ModelSEED reactions TSV missing 'ec_numbers'/'pathways' columns. "
#             "Available: %s", list(df.columns),
#         )
#         return {}

#     ec_pathways: dict[str, dict[str, set[str]]] = {}
#     for ec_field, pw_field in zip(df["ec_numbers"], df["pathways"]):
#         if not isinstance(ec_field, str) or not ec_field.strip():
#             continue
#         pathways = _parse_pipe_field(pw_field)
#         if not pathways:
#             continue
#         for ec in ec_field.split("|"):
#             ec = ec.strip()
#             if not ec:
#                 continue
#             entry = ec_pathways.setdefault(ec, {})
#             for source, ids in pathways.items():
#                 entry.setdefault(source, set()).update(ids)
#     return ec_pathways


# =============================================================================
# ModelSEED compounds: loading, InChIKey lookup (exact/prefix), fuzzy name fallback
# =============================================================================

@lru_cache(maxsize=1)
def _get_modelseed_compounds(cache_path: Path) -> pd.DataFrame:
    """Load the ModelSEED compounds table, fetching and caching it if needed.

    Result is memoized (per `cache_path`) since this is called from several
    independent lookup builders (`_build_inchikey_to_cpd`, `_build_inchikey_prefix_to_cpd`,
    `_build_fuzzy_name_candidates`).
    """
    _MODELSEED_COMPOUNDS_URL = (
        "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
        "master/Biochemistry/compounds.tsv"
    )
    if cache_path.exists():
        log.info(f"Loading ModelSEED compounds from local cache: {cache_path}")
    else:
        log.info(f"Fetching ModelSEED compounds table from {_MODELSEED_COMPOUNDS_URL}")
        resp = requests.get(_MODELSEED_COMPOUNDS_URL, timeout=30)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text, encoding="utf-8")
        log.info(f"Saved ModelSEED compounds cache to {cache_path}")

    return pd.read_csv(cache_path, sep="\t", low_memory=False)


def _inchikey_prefix(inchikey: str, n_chars: int = 25) -> str:
    """Return the first two dash-separated sections of an InChIKey.

    Standard InChIKey: 14-char skeleton + '-' + 10-char stereo layer + '-' + 1-char flag.
    The first two sections together = 25 chars (14 + 1 + 10).
    """
    return inchikey.strip()[:n_chars]


def _build_inchikey_to_cpd(cache_path: Path) -> dict[str, str]:
    """Return a mapping of exact InChIKey to semicolon-joined ModelSEED CPD IDs."""
    df = _get_modelseed_compounds(cache_path)

    if "inchikey" not in df.columns or "id" not in df.columns:
        log.error(
            "ModelSEED compounds TSV does not contain expected columns "
            "'id' / 'inchikey'. Available: %s", list(df.columns)
        )
        return {}

    df = df[["id", "inchikey"]].dropna(subset=["inchikey"])
    df = df[df["inchikey"].str.strip() != ""]

    return df.groupby("inchikey")["id"].agg(lambda ids: ";".join(ids)).to_dict()


def _build_inchikey_prefix_to_cpd(cache_path: Path) -> dict[str, str]:
    """Return a mapping of 25-char InChIKey prefix to semicolon-joined ModelSEED CPD IDs."""
    df = _get_modelseed_compounds(cache_path)

    if "inchikey" not in df.columns or "id" not in df.columns:
        log.error(
            "ModelSEED compounds TSV does not contain expected columns "
            "'id' / 'inchikey'. Available: %s", list(df.columns)
        )
        return {}

    df = df[["id", "inchikey"]].dropna(subset=["inchikey"])
    df = df[df["inchikey"].str.strip() != ""]
    df = df.assign(_prefix=df["inchikey"].map(_inchikey_prefix))

    return (
        df.groupby("_prefix")["id"]
        .agg(lambda ids: ";".join(sorted(set(ids))))
        .to_dict()
    )


def _build_fuzzy_name_candidates(cache_path: Path) -> tuple[list[str], list[str]]:
    """Return parallel lists of (name+aliases text, cpd id) for fuzzy matching."""
    df = _get_modelseed_compounds(cache_path)

    if "name" not in df.columns or "id" not in df.columns:
        log.error(
            "ModelSEED compounds TSV does not contain expected columns "
            "'id' / 'name'. Available: %s", list(df.columns)
        )
        return [], []

    df = df[["id", "name", "aliases"]].dropna(subset=["name"])
    combined_text = df["name"].fillna("") + " " + df.get("aliases", pd.Series("", index=df.index)).fillna("")

    return combined_text.tolist(), df["id"].tolist()


def match_inchikey(
    inchikey: str,
    exact_lookup: dict[str, str],
    prefix_lookup: dict[str, str],
) -> str | None:
    """Try exact InChIKey match, then fall back to 25-char prefix match."""
    inchikey = inchikey.strip()
    hit = exact_lookup.get(inchikey)
    if hit:
        return hit
    return prefix_lookup.get(_inchikey_prefix(inchikey))

def match_name_fuzzy(
    name: str,
    fuzzy_texts: list[str],
    fuzzy_cpds: list[str],
    score_cutoff: float = 90.0,
) -> str | None:
    """Fuzzy substring match of `name` against ModelSEED name/aliases text.

    Logs a warning whenever this fallback actually assigns a cpd.
    """
    if not name or pd.isna(name) or not fuzzy_texts:
        return None

    result = process.extractOne(
        name,
        fuzzy_texts,
        scorer=fuzz.partial_ratio,
        score_cutoff=score_cutoff,
    )
    if result is None:
        return None

    matched_text, score, idx = result
    cpd_id = fuzzy_cpds[idx]
    log.warning(
        "Fuzzy name match used: '%s' ~ ModelSEED '%s' (cpd=%s, score=%.1f)",
        name, matched_text, cpd_id, score,
    )
    return cpd_id

def resolve_modelseed_ids(
    inchikey_str,
    compound_name_str,
    exact_lookup: dict[str, str],
    prefix_lookup: dict[str, str],
    fuzzy_texts: list[str],
    fuzzy_cpds: list[str],
    key_sep: str = ";;",
    name_sep: str = ";;",
    val_sep: str = ";",
    fuzzy_score_cutoff: float = 85.0,
):
    """Resolve one annotation_df row to a set of unique ModelSEED cpd IDs.

    Tries, in order:
      1. Exact InChIKey match (for each key in a multi-key string)
      2. 25-char InChIKey prefix match (structure + stereo layer only)
      3. Fuzzy substring match of Compound_Name against name/aliases (fallback only)
    """
    cpds: set[str] = set()

    if not pd.isna(inchikey_str):
        for key in str(inchikey_str).split(key_sep):
            key = key.strip()
            if not key:
                continue
            hit = match_inchikey(key, exact_lookup, prefix_lookup)
            if hit:
                cpds.update(hit.split(val_sep))

    if not cpds and not pd.isna(compound_name_str):
        for name in str(compound_name_str).split(name_sep):
            name = name.strip()
            if not name:
                continue
            fuzzy_hit = match_name_fuzzy(name, fuzzy_texts, fuzzy_cpds, fuzzy_score_cutoff)
            if fuzzy_hit:
                cpds.update(fuzzy_hit.split(val_sep))

    return val_sep.join(sorted(cpds)) if cpds else pd.NA


# =============================================================================
# Generic multi-value lookup merge helper (replaces the old `_merge_modelseed_ids`)
# =============================================================================

def _merge_multivalue_lookup(
    series: pd.Series,
    lookup: dict[str, str],
    key_sep: str = ";;",
    val_sep: str = ";",
) -> pd.Series:
    """Vectorized resolution of a (possibly multi-value) column via a str->str lookup dict.

    Splits each cell on `key_sep`, looks up each individual key, splits each hit's
    `val_sep`-joined value, and re-aggregates unique results per original row.
    Used for both InChIKey -> cpd and EC number -> rxn resolution wherever an
    exact/prefix-only lookup (no fuzzy fallback needed) is sufficient.

    Rows where every key misses the lookup are returned as NaN.
    """
    exploded = series.str.split(key_sep).explode().str.strip()
    mapped = exploded.map(lookup).str.split(val_sep).explode()
    result = mapped.dropna().groupby(level=0).agg(lambda s: val_sep.join(sorted(set(s))))
    return result.reindex(series.index)


# =============================================================================
# Unified pathway column for integrated transcript + metabolite features
# =============================================================================

def add_modelseed_pathway_column(
    df: pd.DataFrame,
    cache_path: Path,
    pathway_sources: tuple[str, ...] = ("KEGG", "MetaCyc", "PlantCyc", "NOPATHWAY"),
    rxn_col: str = "tx_modelseed_rxn",
    cpd_col: str = "mx_modelseed_id",
    out_col: str = "modelseed_pathway",
    sep: str = ";",
    missing_token: str = "Unassigned",
) -> pd.DataFrame:
    """Add a `modelseed_pathway` column that unifies transcripts and metabolites
    onto the shared ModelSEED pathway namespace.

    Transcript rows are resolved via `tx_modelseed_rxn` (rxn IDs -> pathways),
    metabolite rows via `mx_modelseed_id` (cpd IDs -> pathways). Both lookups
    draw from the same `pathway_sources`, so a transcript and a metabolite
    that land in the same pathway get an identical value in `out_col` --
    that shared string is your cross-data-type join key.

    Reactions with compounds but no real pathway get a unique placeholder
    pathway under source "NOPATHWAY" (see `_assign_placeholder_pathways`).
    Include "NOPATHWAY" in `pathway_sources` (the default) to preserve these
    transcript<->metabolite links even when no biological pathway is known;
    exclude it if you only want real, named pathways.

    Parameters
    ----------
    pathway_sources:
        Pathway annotation source(s) from the ModelSEED reactions `pathways`
        field to pull from, e.g. "KEGG", "MetaCyc", "PlantCyc", "NOPATHWAY".
        Multiple sources are unioned per row.
    rxn_col, cpd_col:
        Column names to look for. If not present in `df` as given, this function
        also tries to auto-detect columns ending in "_modelseed_rxn" / "_modelseed_id"
        (useful since `annotate_integrated_features` prefixes columns by dataset name,
        e.g. "transcriptome_1_modelseed_rxn" rather than the assumed "tx_modelseed_rxn").
    sep:
        Delimiter for multi-valued IDs in `rxn_col`/`cpd_col`, and also used
        to join multiple resolved pathway IDs in `out_col`.
    missing_token:
        Sentinel string used in this table for "no annotation" (both empty
        cells and the literal "Unassigned").
    """
    rxn_to_pathways = _build_rxn_to_pathways(cache_path)
    cpd_to_pathways = _build_cpd_to_pathways(cache_path)

    if rxn_col not in df.columns:
        candidates = [c for c in df.columns if c.endswith("_modelseed_rxn") or c == "modelseed_rxn"]
        if candidates:
            log.info(f"'{rxn_col}' not found; auto-detected transcript rxn column(s): {candidates}")
            rxn_col = candidates[0]
        else:
            rxn_col = None

    if cpd_col not in df.columns:
        candidates = [c for c in df.columns if c.endswith("_modelseed_id") or c == "modelseed_id"]
        if candidates:
            log.info(f"'{cpd_col}' not found; auto-detected metabolite cpd column(s): {candidates}")
            cpd_col = candidates[0]
        else:
            cpd_col = None

    def _is_missing(val: object) -> bool:
        return (
            val is None
            or (isinstance(val, float) and pd.isna(val))
            or not isinstance(val, str)
            or val.strip() in ("", missing_token)
        )

    def _resolve(raw_ids: object, lookup: dict[str, dict[str, set[str]]]) -> set[str]:
        pathways: set[str] = set()
        if _is_missing(raw_ids):
            return pathways
        for _id in raw_ids.split(sep):
            _id = _id.strip()
            if not _id or _id == missing_token:
                continue
            entry = lookup.get(_id, {})
            for source in pathway_sources:
                pathways.update(entry.get(source, set()))
        return pathways

    def _row_pathway(row: pd.Series) -> str:
        pathways: set[str] = set()
        if rxn_col is not None:
            pathways |= _resolve(row[rxn_col], rxn_to_pathways)
        if cpd_col is not None:
            pathways |= _resolve(row[cpd_col], cpd_to_pathways)
        return sep.join(sorted(pathways)) if pathways else missing_token

    out = df.copy()
    out[out_col] = out.apply(_row_pathway, axis=1)

    n_mapped = (out[out_col] != missing_token).sum()

    n_tx_mapped = (
        (~out[rxn_col].apply(_is_missing) & (out[out_col] != missing_token)).sum()
        if rxn_col is not None else 0
    )
    n_mx_mapped = (
        (~out[cpd_col].apply(_is_missing) & (out[out_col] != missing_token)).sum()
        if cpd_col is not None else 0
    )
    log.info(
        f"Assigned {out_col} for {n_mapped}/{len(out)} rows "
        f"(transcripts: {n_tx_mapped}, metabolites: {n_mx_mapped}; "
        f"sources={pathway_sources})"
    )
    return out


# =============================================================================
# Validation helpers
# =============================================================================

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

    overlap_pct = (
        len(common_metabolites) / len(raw_data_metabolites) * 100
        if len(raw_data_metabolites) > 0 else 0
    )

    log.info(f"Metabolite ID Validation Results:")
    log.info(f"  Raw data metabolites: {len(raw_data_metabolites)}")
    log.info(f"  Annotation metabolites: {len(annotation_metabolites)}")
    log.info(f"  Common metabolites: {len(common_metabolites)} ({overlap_pct:.1f}% of raw data)")

    if overlap_pct < 50:
        log.warning(f"Low metabolite ID overlap ({overlap_pct:.1f}%) between raw data and annotations")
    if overlap_pct == 0:
        log.info("Raw data metabolite IDs (first 10):")
        log.info(list(raw_data_metabolites)[:10])
        log.info("Annotation metabolite IDs (first 10):")
        log.info(list(annotation_metabolites)[:10])
        raise ValueError("No matching metabolite IDs found between raw data and annotations, something is wrong.")


# =============================================================================
# Annotation table generators
# =============================================================================

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
            matches = fbmn_data[fbmn_data['metabolite_id'] == met_id]
            if len(matches) > 0:
                def concat_unique_values(series):
                    """Concatenate unique non-null values with ;;"""
                    unique_vals = sorted({
                        str(val).strip() for val in series.dropna()
                        if str(val).strip() and str(val).strip().lower() not in ['nan', 'none', '']
                    })
                    return ';;'.join(unique_vals) if unique_vals else None

                inchikey_cols = ['InChiKey', 'InChIKey', 'inchikey', 'INCHIKEY']
                inchikey_values = []
                for col in inchikey_cols:
                    if col in matches.columns:
                        inchikey_values.extend(matches[col].dropna().tolist())

                mapping_data.append({
                    'metabolite_id': met_id,
                    'molecular_formula': concat_unique_values(matches.get('molecular_formula', pd.Series())),
                    'Compound_Name': concat_unique_values(matches.get('Compound_Name', pd.Series())),
                    'Smiles': concat_unique_values(matches.get('Smiles', pd.Series())),
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

        mapping_data.append({
            'metabolite_id': met_id,
            'molecular_formula': None,
            'Compound_Name': None,
            'Smiles': None,
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

    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.rename(columns={'metabolite_id': 'metabolome_id'}, inplace=True)
    mapping_df = mapping_df.set_index('metabolome_id')
    mapping_df = mapping_df.map(lambda x: str(x).replace('|', ';;') if isinstance(x, str) else x)
    mapping_df['display_name'] = mapping_df['Compound_Name']

    # Add ModelSEED cpd ID based on InChIKey (exact -> prefix -> fuzzy name fallback)
    if not mapping_df.empty:
        compounds_cache_path = Path(output_dir) / "modelseed_compounds.tsv"
        exact_lookup = _build_inchikey_to_cpd(compounds_cache_path)
        prefix_lookup = _build_inchikey_prefix_to_cpd(compounds_cache_path)
        fuzzy_texts, fuzzy_cpds = _build_fuzzy_name_candidates(compounds_cache_path)

        mapping_df["modelseed_id"] = mapping_df.apply(
            lambda row: resolve_modelseed_ids(
                row["InChiKey"],
                row["Compound_Name"],
                exact_lookup,
                prefix_lookup,
                fuzzy_texts,
                fuzzy_cpds,
            ),
            axis=1,
        )

    if not raw_data.empty and not mapping_df.empty:
        _validate_annotation_metabolite_ids(mapping_df, raw_data)
    else:
        if raw_data.empty:
            log.warning("Raw data is empty, skipping validation")
        if mapping_df.empty:
            log.warning("Annotation table is empty, skipping validation")

    os.makedirs(output_dir, exist_ok=True)
    write_integration_file(data=mapping_df, output_dir=output_dir, filename=output_filename, indexing=True)

    total_rows = len(mapping_df)
    annotated_count = mapping_df.dropna(subset=['INCHI', 'InChiKey'], how='all').shape[0]

    multi_annotation_count = 0
    for col in ['Compound_Name', 'superclass', 'class', 'subclass']:
        if col in mapping_df.columns:
            multi_count = mapping_df[col].str.contains(';;', na=False).sum()
            if multi_count > 0:
                multi_annotation_count = max(multi_annotation_count, multi_count)

    log.info(f"Created annotation mapping with {total_rows} rows")
    log.info(f"Metabolites with annotations: {annotated_count}")
    log.info(f"Metabolites without annotations: {total_rows - annotated_count}")
    log.info(f"Metabolites with multiple annotations: {multi_annotation_count}")

    return mapping_df


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
        annotation_df = _process_plant_annotations(raw_data_dir, output_dir, output_filename)
    else:
        raise ValueError(f"Invalid genome_type '{genome_type}'. Must be one of: 'microbe', 'algal', 'metagenome', 'plant'")

    if not raw_data.empty and not annotation_df.empty:
        _validate_annotation_gene_ids(annotation_df, raw_data)
    else:
        raise ValueError("Either input raw_data or computed annotation_df is empty. Cannot validate gene IDs.")

    annotation_df = annotation_df.set_index('transcriptome_id')
    annotation_df = annotation_df.map(lambda x: str(x).replace('|', ';;') if isinstance(x, str) else x)

    if not annotation_df.empty:
        ec_to_rxn = _build_ec_to_rxn(cache_path=Path(output_dir) / "modelseed_reactions.tsv")
        annotation_df['modelseed_rxn'] = _merge_multivalue_lookup(
            annotation_df['kegg_ec'], ec_to_rxn, key_sep=";;", val_sep=";"
        )

    write_integration_file(annotation_df, output_dir, output_filename, indexing=True)

    return annotation_df


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

    all_features = integrated_data.index.tolist()
    log.info(f"Annotating {len(all_features)} features from integrated data")

    final_annotation_df = pd.DataFrame(index=all_features)
    final_annotation_df.index.name = 'feature_id'

    dataset_stats = {}

    if datasets is not None:
        for dataset in datasets:
            if hasattr(dataset, 'annotation_table') and dataset.annotation_table is not None:
                log.info(f"Processing annotation table for {dataset.dataset_name}")

                ann_table = dataset.annotation_table.copy()

                potential_id_cols = []
                for col in ann_table.columns:
                    if 'id' in col.lower() or col == ann_table.index.name:
                        potential_id_cols.append(col)

                ann_table.columns = [f"{dataset.dataset_name}_{col}" for col in ann_table.columns]

                matches = ann_table.index.intersection(all_features)

                dataset_features = [f for f in all_features if f.startswith(f"{dataset.dataset_name}_")]

                dataset_stats[dataset.dataset_name] = {
                    'total_features_in_integrated': len(dataset_features),
                    'total_annotations_available': len(ann_table),
                    'features_with_annotations': len(matches),
                    'annotation_columns': len(ann_table.columns)
                }

                final_annotation_df = final_annotation_df.join(ann_table, how='left')

                log.info(f"  Dataset features in integrated data: {len(dataset_features)}")
                log.info(f"  Annotation records available: {len(ann_table)}")
                log.info(f"  Features with annotations: {len(matches)}")
                log.info(f"  Annotation columns added: {len(ann_table.columns)}")

    # Add ModelSEED pathway column (auto-detects dataset-prefixed rxn/cpd columns
    # if the "tx_"/"mx_" naming convention isn't exactly followed)
    final_annotation_df = add_modelseed_pathway_column(
        final_annotation_df, cache_path=Path(output_dir) / "modelseed_reactions.tsv"
    )

    final_annotation_df = final_annotation_df.fillna('Unassigned')

    final_annotation_df = final_annotation_df.reset_index()

    n_features = len(final_annotation_df)
    n_annotation_cols = len(final_annotation_df.columns) - 1

    non_unassigned_mask = (final_annotation_df.iloc[:, 1:] != 'Unassigned').any(axis=1)
    n_annotated_features = non_unassigned_mask.sum()

    log.info(f"Overall annotation summary:")
    log.info(f"  Total features: {n_features}")
    log.info(f"  Features with annotations: {n_annotated_features}")
    log.info(f"  Features without annotations: {n_features - n_annotated_features}")
    log.info(f"  Total annotation columns: {n_annotation_cols}")

    if dataset_stats:
        log.info(f"Dataset-specific annotation summaries:")
        for dataset_name, stats in dataset_stats.items():
            dataset_features = [f for f in all_features if f.startswith(f"{dataset_name}_")]
            if dataset_features:
                dataset_feature_indices = final_annotation_df[
                    final_annotation_df['feature_id'].isin(dataset_features)
                ].index

                dataset_annotation_cols = [col for col in final_annotation_df.columns
                                         if col.startswith(f"{dataset_name}_")]

                if dataset_annotation_cols:
                    dataset_annotated_mask = (
                        final_annotation_df.loc[dataset_feature_indices, dataset_annotation_cols] != 'Unassigned'
                    ).any(axis=1)
                    dataset_annotated_count = dataset_annotated_mask.sum()
                else:
                    dataset_annotated_count = 0

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

    if output_dir:
        write_integration_file(
            data=final_annotation_df,
            output_dir=output_dir,
            filename=output_filename,
            indexing=False
        )

    return final_annotation_df

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
        ds.dataset_name: {"outdir": ds.output_dir, 
                          "apid": getattr(ds, "apid", None), 
                          "raw": ds._raw_metadata_filename,
                          "metab_fraction": ds.metabolite_fraction if hasattr(ds, "metabolite_fraction") else None}
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

def _build_group_means(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    Build a feature x group mean matrix from feature x sample data.
    """
    if sample_col not in metadata.columns:
        raise ValueError(f"Column '{sample_col}' not found in metadata.")
    if group_col not in metadata.columns:
        raise ValueError(f"Column '{group_col}' not found in metadata.")

    meta = metadata[[sample_col, group_col]].dropna().copy()
    meta[sample_col] = meta[sample_col].astype(str)
    meta[group_col] = meta[group_col].astype(str)

    # Keep only metadata rows whose samples exist in data columns
    available_samples = [c for c in data.columns if str(c) in set(meta[sample_col])]
    if len(available_samples) == 0:
        raise ValueError("No overlapping sample IDs between metadata and data columns.")

    data_sub = data[[c for c in data.columns if str(c) in set(available_samples)]].copy()
    meta_sub = meta[meta[sample_col].isin([str(c) for c in data_sub.columns])].copy()

    group_to_samples = (
        meta_sub.groupby(group_col)[sample_col]
        .apply(lambda s: [x for x in s.tolist() if x in [str(c) for c in data_sub.columns]])
    .to_dict()
    )
    group_to_samples = {g: s for g, s in group_to_samples.items() if len(s) > 0}

    if len(group_to_samples) < 2:
        raise ValueError(
            f"Need at least 2 non-empty groups to compute LFC. Found groups: {list(group_to_samples.keys())}"
        )

    # Use string-indexed data columns for robust matching
    data_sub.columns = data_sub.columns.astype(str)

    group_means = pd.DataFrame(
        {g: data_sub[samples].mean(axis=1) for g, samples in group_to_samples.items()},
        index=data_sub.index
    )
    return group_means

def integrate_metadata(
    datasets: list,
    metadata_vars: List[str] = [],
    unifying_col: str = "unique_group",
    output_filename: str = "integrated_metadata",
    output_dir: str = None,
    method: Literal["replicate_matched", "lfc"] = "replicate_matched",
    group_col: str = "group",
    overlap_only: bool = True,
    lfc_pairs: Optional[List[Union[str, Tuple[str, str]]]] = None,
) -> pd.DataFrame:
    """
    Integrate metadata in one of two modes:

    - replicate_matched: original sample-level integration by unifying_col
    - lfc: contrast-level integration where each row is a groupA_vs_groupB contrast
    """

    def _resolve_pairs(groups: List[str]) -> List[Tuple[str, str]]:
        groups = sorted(list({str(g) for g in groups}))
        if len(groups) < 2:
            raise ValueError("At least 2 groups are required to build LFC contrasts.")

        if lfc_pairs is None:
            return list(combinations(groups, 2))

        valid_groups = set(groups)
        resolved: List[Tuple[str, str]] = []

        for pair in lfc_pairs:
            if isinstance(pair, str):
                if "_vs_" not in pair:
                    raise ValueError(
                        f"Invalid lfc_pairs entry '{pair}'. Use 'A_vs_B' or ('A','B')."
                    )
                a, b = pair.split("_vs_", 1)
            elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                a, b = str(pair[0]), str(pair[1])
            else:
                raise ValueError(
                    f"Invalid lfc_pairs entry '{pair}'. Use 'A_vs_B' or ('A','B')."
                )

            if a == b:
                raise ValueError(f"Invalid contrast {a}_vs_{b}: groups must differ.")
            if a not in valid_groups or b not in valid_groups:
                raise ValueError(
                    f"Contrast {a}_vs_{b} contains unknown groups. Valid groups: {sorted(valid_groups)}"
                )
            resolved.append((a, b))

        # Deduplicate while preserving order
        deduped: List[Tuple[str, str]] = []
        seen = set()
        for p in resolved:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    if method not in {"replicate_matched", "lfc"}:
        raise ValueError("method must be 'replicate_matched' or 'lfc'.")

    # Original behavior
    if method == "replicate_matched":
        log.info("Creating a single integrated (shared) metadata table across datasets...")

        metadata_tables = [ds.linked_metadata for ds in datasets if hasattr(ds, "linked_metadata")]
        if not metadata_tables:
            raise ValueError("No datasets with linked_metadata available.")

        subset_cols = metadata_vars + [unifying_col]
        integrated_metadata = metadata_tables[0][subset_cols].copy()

        for i, table in enumerate(metadata_tables[1:], start=2):
            integrated_metadata = integrated_metadata.merge(
                table[subset_cols],
                on=unifying_col,
                suffixes=(None, f"_{i}"),
                how="outer",
            )

            for column in metadata_vars:
                col1 = column
                col2 = f"{column}_{i}"
                if col2 in integrated_metadata.columns:
                    integrated_metadata[column] = integrated_metadata[col1].combine_first(
                        integrated_metadata[col2]
                    )
                    integrated_metadata.drop(columns=[col2], inplace=True)

        integrated_metadata.rename(columns={unifying_col: "sample"}, inplace=True)
        integrated_metadata.sort_values("sample", inplace=True)
        integrated_metadata.drop_duplicates(inplace=True)
        integrated_metadata.set_index("sample", inplace=True)

        if output_dir:
            log.info("Writing integrated metadata table...")
            write_integration_file(
                data=integrated_metadata,
                output_dir=output_dir,
                filename=output_filename,
            )

        return integrated_metadata

    # LFC mode
    log.info("Creating integrated metadata table in LFC mode (contrast-level rows)...")

    group_sets = []
    for ds in datasets:
        if not hasattr(ds, "linked_metadata") or ds.linked_metadata is None or ds.linked_metadata.empty:
            raise ValueError(f"Dataset {ds.dataset_name} missing linked_metadata.")
        if group_col not in ds.linked_metadata.columns:
            raise ValueError(
                f"Dataset {ds.dataset_name} linked_metadata missing group column '{group_col}'."
            )

        ds_groups = set(ds.linked_metadata[group_col].dropna().astype(str).unique())
        if len(ds_groups) < 2:
            raise ValueError(
                f"Dataset {ds.dataset_name} has fewer than 2 groups in '{group_col}'."
            )
        group_sets.append(ds_groups)

    if not group_sets:
        raise ValueError("No valid datasets available to build LFC metadata.")

    if overlap_only and len(group_sets) > 1:
        groups = sorted(list(set.intersection(*group_sets)))
    else:
        groups = sorted(list(set.union(*group_sets)))

    if len(groups) < 2:
        raise ValueError("Not enough groups available to build LFC contrasts.")

    pairs = _resolve_pairs(groups)

    rows = []
    for a, b in pairs:
        contrast = f"{a}_vs_{b}"
        rows.append(
            {
                "sample": contrast,
                "contrast": contrast,
                "group_a": a,
                "group_b": b,
                group_col: contrast,
            }
        )

    integrated_metadata = pd.DataFrame(rows).drop_duplicates(subset=["sample"]).set_index("sample")

    # Keep expected columns present for downstream compatibility
    for col in metadata_vars:
        if col not in integrated_metadata.columns:
            integrated_metadata[col] = np.nan

    if output_dir:
        log.info("Writing integrated metadata table...")
        write_integration_file(
            data=integrated_metadata,
            output_dir=output_dir,
            filename=output_filename,
        )

    return integrated_metadata


def integrate_data(
    datasets: list,
    overlap_only: bool = True,
    output_filename: str = "integrated_data",
    output_dir: str = None,
    method: Literal["replicate_matched", "lfc"] = "replicate_matched",
    group_col: str = "group",
    sample_col: str = "unique_group",
    data_attr: str = "devarianced_data",
    pseudocount: float = 1.0,
    lfc_pairs: Optional[List[Union[str, Tuple[str, str]]]] = None,
) -> pd.DataFrame:
    """
    Integrate datasets in one of two modes:

    - replicate_matched: original behavior (feature x matched-sample matrix)
    - lfc: feature x contrast matrix where columns are pairwise group log2 fold-change
    """

    def _resolve_pairs(groups: List[str]) -> List[Tuple[str, str]]:
        groups = sorted(list({str(g) for g in groups}))
        if len(groups) < 2:
            raise ValueError("At least 2 groups are required to compute LFC.")

        if lfc_pairs is None:
            return list(combinations(groups, 2))

        valid_groups = set(groups)
        resolved: List[Tuple[str, str]] = []

        for pair in lfc_pairs:
            if isinstance(pair, str):
                if "_vs_" not in pair:
                    raise ValueError(
                        f"Invalid lfc_pairs entry '{pair}'. Use 'A_vs_B' or ('A','B')."
                    )
                a, b = pair.split("_vs_", 1)
            elif isinstance(pair, (tuple, list)) and len(pair) == 2:
                a, b = str(pair[0]), str(pair[1])
            else:
                raise ValueError(
                    f"Invalid lfc_pairs entry '{pair}'. Use 'A_vs_B' or ('A','B')."
                )

            if a == b:
                raise ValueError(f"Invalid contrast {a}_vs_{b}: groups must differ.")
            if a not in valid_groups or b not in valid_groups:
                raise ValueError(
                    f"Contrast {a}_vs_{b} contains unknown groups. Valid groups: {sorted(valid_groups)}"
                )
            resolved.append((a, b))

        deduped: List[Tuple[str, str]] = []
        seen = set()
        for p in resolved:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        return deduped

    if method not in {"replicate_matched", "lfc"}:
        raise ValueError("method must be 'replicate_matched' or 'lfc'.")

    # Original behavior
    if method == "replicate_matched":
        log.info(f"Creating a single integrated feature matrix across datasets (data_attr='{data_attr}')...")

        dataset_data = {}
        sample_sets = {}

        for ds in datasets:
            source_df = getattr(ds, data_attr, None)
            if source_df is not None and not source_df.empty:
                data_copy = source_df.copy()
                if not data_copy.index.astype(str).str.startswith(f"{ds.dataset_name}_").all():
                    data_copy.index = [f"{ds.dataset_name}_{idx}" for idx in data_copy.index]

                dataset_data[ds.dataset_name] = data_copy
                sample_sets[ds.dataset_name] = set(data_copy.columns.astype(str))
                log.info(f"\tAdding {data_copy.shape[0]} features from {ds.dataset_name} (source: {data_attr})")
            else:
                raise ValueError(
                    f"Dataset '{ds.dataset_name}' missing attribute '{data_attr}'. "
                    f"Run the appropriate processing step first."
                )

        if overlap_only and len(dataset_data) > 1:
            log.info("\tRestricting to overlapping samples across all datasets...")
            overlapping_samples = set.intersection(*sample_sets.values())

            if not overlapping_samples:
                raise ValueError("No overlapping samples found across datasets.")

            shared_cols = sorted(list(overlapping_samples))
            for ds_name, data in dataset_data.items():
                data.columns = data.columns.astype(str)
                dataset_data[ds_name] = data[shared_cols]
                log.info(f"\t{ds_name}: {len(shared_cols)} overlapping samples")

        integrated_data = pd.concat(dataset_data.values(), axis=0)
        integrated_data.index.name = "features"
        integrated_data = integrated_data.fillna(0)

        log.info(
            f"Final integrated dataset: {integrated_data.shape[0]} features x {integrated_data.shape[1]} samples"
        )

        if output_dir:
            log.info("Writing integrated data table...")
            write_integration_file(integrated_data, output_dir, output_filename, indexing=True)

        return integrated_data

    # LFC mode
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0 in LFC mode.")

    log.info("Creating integrated feature matrix in LFC mode...")

    dataset_data = {}
    contrast_sets = {}

    for ds in datasets:
        if not hasattr(ds, data_attr):
            raise ValueError(f"Dataset {ds.dataset_name} has no attribute '{data_attr}'.")

        source_df = getattr(ds, data_attr)
        if source_df is None or source_df.empty:
            raise ValueError(
                f"Dataset {ds.dataset_name} attribute '{data_attr}' is empty."
            )

        if not hasattr(ds, "linked_metadata") or ds.linked_metadata is None or ds.linked_metadata.empty:
            raise ValueError(f"Dataset {ds.dataset_name} missing linked_metadata for LFC mode.")

        # Build group means from feature x sample matrix
        group_means = _build_group_means(
            data=source_df,
            metadata=ds.linked_metadata,
            sample_col=sample_col,
            group_col=group_col,
        )

        pairs = _resolve_pairs(list(group_means.columns))

        lfc_df = pd.DataFrame(index=group_means.index)
        for a, b in pairs:
            contrast = f"{a}_vs_{b}"
            lfc_df[contrast] = np.log2(
                (group_means[a].astype(float) + pseudocount)
                / (group_means[b].astype(float) + pseudocount)
            )

        # Add dataset prefix to feature names
        if not lfc_df.index.astype(str).str.startswith(f"{ds.dataset_name}_").all():
            lfc_df.index = [f"{ds.dataset_name}_{idx}" for idx in lfc_df.index]

        dataset_data[ds.dataset_name] = lfc_df
        contrast_sets[ds.dataset_name] = set(lfc_df.columns.astype(str))

        log.info(
            f"\t{ds.dataset_name}: {lfc_df.shape[0]} features x {lfc_df.shape[1]} contrasts"
        )

    if not dataset_data:
        raise ValueError("No datasets available for LFC integration.")

    if overlap_only and len(dataset_data) > 1:
        common_contrasts = set.intersection(*contrast_sets.values())
        if not common_contrasts:
            raise ValueError(
                "No shared LFC contrasts across datasets. Use overlap_only=False or align groups."
            )
        final_cols = sorted(list(common_contrasts))
        log.info(f"\tRestricting to {len(final_cols)} shared contrasts.")
    else:
        final_cols = sorted(list(set.union(*contrast_sets.values())))
        log.info(f"\tUsing union of contrasts ({len(final_cols)} total).")

    for ds_name in dataset_data:
        dataset_data[ds_name] = dataset_data[ds_name].reindex(columns=final_cols)

    integrated_data = pd.concat(dataset_data.values(), axis=0)
    integrated_data.index.name = "features"
    integrated_data = integrated_data.fillna(0)

    log.info(
        f"Final LFC integrated dataset: {integrated_data.shape[0]} features x {integrated_data.shape[1]} contrasts"
    )

    if output_dir:
        log.info("Writing integrated data table...")
        write_integration_file(integrated_data, output_dir, output_filename, indexing=True)

    return integrated_data

def integrate_data_condition_resolution(
    datasets: list,
    overlap_only: bool = True,
    output_filename: str = "integrated_data",
    output_dir: str = None,
    group_col: str = "group",
    sample_col: str = "unique_group",
    condition_approach: str = "centroid",
    control_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    Condition-resolution integration workflow.

    Steps applied to each dataset independently, then concatenated:

    1. **log2 transform** — ``log2(x + 1)`` applied to ``replicate_filtered_data``.
    2. **Variance filter** — features with near-zero variance across all samples are
       dropped (prevents noise amplification during Z-scoring).
    3. **Collapse replicates** — compute the per-condition mean (centroid) for each
       experimental condition.
    4. **Condition approach** (config choice):
       - ``"centroid"`` — use the condition means directly.
       - ``"lfc"``      — subtract the control group mean from all other condition
                          means so that values represent the response to treatment.
    5. **Row-wise Z-score** — standardise each feature across conditions so that
       different data types are on the same scale.
    6. **Vertical concatenation** — stack the per-dataset matrices into a single
       features × conditions matrix.

    Parameters
    ----------
    datasets : list
        Dataset objects with ``replicate_filtered_data`` and ``linked_metadata``.
    overlap_only : bool
        If True, restrict to conditions shared across all datasets.
    output_filename : str
        Output CSV filename (without extension).
    output_dir : str
        Directory to write the output file.
    group_col : str
        Metadata column containing condition/group labels.
    sample_col : str
        Metadata column containing unique sample identifiers matching data columns.
    condition_approach : str
        ``"centroid"`` or ``"lfc"``.
    control_group : str or None
        Name of the control condition for LFC subtraction.  If None, the
        alphabetically first condition is used.

    Returns
    -------
    pd.DataFrame
        Integrated features × conditions matrix (row-wise Z-scored).
    """
    if condition_approach not in {"centroid", "lfc"}:
        raise ValueError(
            f"condition_approach must be 'centroid' or 'lfc', got '{condition_approach}'."
        )

    dataset_matrices: Dict[str, pd.DataFrame] = {}
    condition_sets: Dict[str, set] = {}

    for ds in datasets:
        source_df = getattr(ds, 'replicate_filtered_data', None)
        if source_df is None or source_df.empty:
            raise ValueError(
                f"Dataset '{ds.dataset_name}' has no replicate_filtered_data. "
                "Run replicability_test_all_datasets() first."
            )
        if not hasattr(ds, 'linked_metadata') or ds.linked_metadata is None or ds.linked_metadata.empty:
            raise ValueError(
                f"Dataset '{ds.dataset_name}' missing linked_metadata."
            )
        if group_col not in ds.linked_metadata.columns:
            raise ValueError(
                f"Dataset '{ds.dataset_name}' linked_metadata missing column '{group_col}'."
            )

        # ── Step 1: log2 transform ────────────────────────────────────────────
        log.info(f"  [{ds.dataset_name}] Applying log2(x+1) transform...")
        log2_df = np.log2(source_df.astype(float) + 1)

        # ── Step 2: variance filter ───────────────────────────────────────────
        row_var = log2_df.var(axis=1)
        low_var_mask = row_var > 0.0  # drop perfectly constant features
        n_dropped = (~low_var_mask).sum()
        if n_dropped:
            log.info(f"  [{ds.dataset_name}] Dropped {n_dropped} zero-variance features after log2.")
        log2_df = log2_df.loc[low_var_mask]

        # ── Step 3: collapse replicates to condition means ────────────────────
        meta = ds.linked_metadata
        # Build sample → condition mapping using sample_col → group_col
        if sample_col in meta.columns:
            sample_to_group = meta.set_index(sample_col)[group_col].to_dict()
        else:
            # Fall back: index is already the sample identifier
            sample_to_group = meta[group_col].to_dict()

        # Align data columns to metadata
        common_samples = [c for c in log2_df.columns if c in sample_to_group]
        if not common_samples:
            raise ValueError(
                f"Dataset '{ds.dataset_name}': no overlap between data columns and "
                f"metadata '{sample_col}' values."
            )
        log2_df = log2_df[common_samples]

        # Group by condition and compute mean
        condition_labels = pd.Series(
            [sample_to_group[s] for s in common_samples],
            index=common_samples,
            name=group_col,
        )
        condition_means = log2_df.T.groupby(condition_labels).mean().T  # features × conditions

        log.info(
            f"  [{ds.dataset_name}] Collapsed {len(common_samples)} samples → "
            f"{condition_means.shape[1]} conditions."
        )

        # ── Step 4: condition approach ────────────────────────────────────────
        if condition_approach == "lfc":
            conditions = sorted(condition_means.columns.tolist())
            ctrl = control_group if control_group is not None else conditions[0]
            if ctrl not in condition_means.columns:
                raise ValueError(
                    f"Dataset '{ds.dataset_name}': control_group '{ctrl}' not found in "
                    f"conditions {conditions}."
                )
            log.info(
                f"  [{ds.dataset_name}] LFC approach: subtracting control '{ctrl}' mean..."
            )
            ctrl_mean = condition_means[ctrl]
            condition_means = condition_means.subtract(ctrl_mean, axis=0)
            # Drop the control column (it becomes all zeros)
            condition_means = condition_means.drop(columns=[ctrl])

        # ── Step 5: row-wise Z-score across conditions ────────────────────────
        log.info(f"  [{ds.dataset_name}] Applying row-wise Z-score across conditions...")
        row_mean = condition_means.mean(axis=1)
        row_std = condition_means.std(axis=1, ddof=1).replace(0, 1)  # avoid /0
        zscore_df = condition_means.subtract(row_mean, axis=0).divide(row_std, axis=0)

        # Add dataset prefix to feature names
        if not zscore_df.index.astype(str).str.startswith(f"{ds.dataset_name}_").all():
            zscore_df.index = [f"{ds.dataset_name}_{idx}" for idx in zscore_df.index]

        dataset_matrices[ds.dataset_name] = zscore_df
        condition_sets[ds.dataset_name] = set(zscore_df.columns.astype(str))

        log.info(
            f"  [{ds.dataset_name}]: {zscore_df.shape[0]} features × "
            f"{zscore_df.shape[1]} conditions"
        )

    if not dataset_matrices:
        raise ValueError("No datasets available for condition_resolution integration.")

    # ── Step 6: align conditions and concatenate ──────────────────────────────
    if overlap_only and len(dataset_matrices) > 1:
        shared_conditions = set.intersection(*condition_sets.values())
        if not shared_conditions:
            raise ValueError(
                "No shared conditions across datasets. Use overlap_only=False or align groups."
            )
        final_cols = sorted(list(shared_conditions))
        log.info(f"Restricting to {len(final_cols)} shared conditions.")
    else:
        all_conditions = set.union(*condition_sets.values())
        final_cols = sorted(list(all_conditions))
        log.info(f"Using union of conditions ({len(final_cols)} total).")

    for ds_name in dataset_matrices:
        dataset_matrices[ds_name] = dataset_matrices[ds_name].reindex(columns=final_cols, fill_value=0.0)

    integrated_data = pd.concat(dataset_matrices.values(), axis=0)
    integrated_data.index.name = "features"
    integrated_data = integrated_data.fillna(0)

    log.info(
        f"Final condition_resolution integrated dataset: "
        f"{integrated_data.shape[0]} features × {integrated_data.shape[1]} conditions"
    )

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
    sigma_threshold: float = 2.0,
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
    
    elif method == "sigma":
        if group_col not in metadata.columns:
            raise ValueError(f"Column '{group_col}' not found in metadata.")
        
        log.info(f"Removing features with outlier replicates (sigma_threshold={sigma_threshold})...")
        
        groups = metadata[group_col].unique()
        group_sample_map = {
            group: [s for s in metadata[metadata[group_col] == group]['unique_group'].tolist()
                   if s in data.columns]
            for group in groups
        }
        valid_groups = {k: v for k, v in group_sample_map.items() if len(v) >= 2}
        
        if not valid_groups:
            log.info("No groups with sufficient replicates (>=2) found. Keeping all features.")
            replicable_data = data
        else:
            keep_mask = pd.Series(True, index=data.index)
            
            for group, samples in valid_groups.items():
                group_data = data[samples]
                group_mean = group_data.mean(axis=1)
                group_std = group_data.std(axis=1, ddof=1)
                
                # Avoid division by zero for constant features
                group_std_safe = group_std.replace(0, np.nan)
                
                # Z-score each replicate relative to the group mean
                z_scores = group_data.subtract(group_mean, axis=0).divide(group_std_safe, axis=0).abs()
                
                # Flag features where ANY replicate exceeds sigma_threshold
                has_outlier = (z_scores > sigma_threshold).any(axis=1)
                keep_mask = keep_mask & (~has_outlier)
            
            replicable_data = data.loc[keep_mask]
        
        log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - replicable_data.shape[0]} to keep {replicable_data.shape[0]}.")
    
    else:
        raise ValueError("Currently only 'variance', 'sigma', or 'none' methods are supported for removing low replicable features.")

    log.info(f"Saving replicable data for {dataset_name}...")
    write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
    return replicable_data


def normalize_integrated_data(
    data: pd.DataFrame,
    method: str,
    metadata: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    log2: bool = True,
    group_col: str = "group",
    sample_col: str = "unique_group",
    pseudocount: float = 1.0,
    lfc_pairs: list = None,
) -> pd.DataFrame:
    """
    Normalize the integrated feature matrix (features × samples, all datasets combined)
    after integration.

    This is the post-integration normalization step that replaces the old per-dataset
    ``scale_data()`` step. It operates on the combined matrix so that all features
    are normalized on the same scale.

    Parameters
    ----------
    data : pd.DataFrame
        Integrated feature matrix (features × samples). Rows = features, columns = samples.
        Should be the replicate-filtered devarianced data concatenated across datasets.
    method : str
        Normalization method:
        - ``"lfc"``            : log2 fold-change of group medians vs. all-group median.
                                 Produces a features × contrasts matrix where each column
                                 is a pairwise group comparison (groupA_vs_groupB).
        - ``"vst"``            : variance-stabilizing transformation (arcsinh(sqrt(x))) + z-score.
        - ``"zscore"``         : log2(x+1) then z-score per sample (column-wise).
        - ``"modified_zscore"``: log2(x+1) then modified z-score (median-based) per sample.
        - ``"rank_normal"``    : rank-based inverse normal transformation per sample.
    metadata : pd.DataFrame
        Sample metadata. Required for ``"lfc"`` method (needs ``group_col`` and ``sample_col``).
        For other methods, pass the integrated metadata for reference.
    output_filename : str
        Output filename (without extension).
    output_dir : str
        Output directory.
    log2 : bool, default True
        Whether to log2-transform before scaling. Used by vst/zscore/modified_zscore methods.
        Ignored for lfc (which computes its own log2 ratios).
    group_col : str, default "group"
        Metadata column containing group labels. Used by ``"lfc"`` method.
    sample_col : str, default "unique_group"
        Metadata column containing sample identifiers matching data columns. Used by ``"lfc"``.
    pseudocount : float, default 1.0
        Pseudocount added before log2 transformation in lfc mode to avoid log(0).
    lfc_pairs : list of [str, str], optional
        Specific pairwise contrasts to compute for ``"lfc"`` method.
        Each element is [groupA, groupB] and the LFC is log2(median_A / median_B).
        If None, all pairwise combinations are computed.

    Returns
    -------
    pd.DataFrame
        Normalized feature matrix. For ``"lfc"``, columns are contrast names
        (e.g. ``"groupA_vs_groupB"``). For all other methods, columns are sample names.
    """
    valid_methods = {"lfc", "vst", "zscore", "modified_zscore", "rank_normal"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    log.info(f"Normalizing integrated data using method='{method}' ({data.shape[0]} features × {data.shape[1]} samples)...")

    if method == "lfc":
        # Build group medians from the integrated matrix
        if group_col not in metadata.columns:
            raise ValueError(f"group_col '{group_col}' not found in metadata columns: {metadata.columns.tolist()}")

        # Map sample → group
        if sample_col in metadata.columns:
            sample_to_group = metadata.set_index(sample_col)[group_col].to_dict()
        else:
            sample_to_group = metadata[group_col].to_dict()

        # Restrict to samples present in data
        common_samples = [s for s in data.columns if s in sample_to_group]
        if not common_samples:
            raise ValueError("No samples in data match the metadata index/sample_col.")

        data_common = data[common_samples]
        groups = sorted(set(sample_to_group[s] for s in common_samples))

        # Compute per-group medians (features × groups)
        group_medians = pd.DataFrame(index=data.index)
        for grp in groups:
            grp_samples = [s for s in common_samples if sample_to_group[s] == grp]
            if grp_samples:
                group_medians[grp] = data_common[grp_samples].median(axis=1)

        # Determine contrasts
        if lfc_pairs is None:
            contrasts = list(combinations(groups, 2))
        else:
            contrasts = [tuple(p) for p in lfc_pairs]

        # Compute log2 fold-changes
        lfc_df = pd.DataFrame(index=data.index)
        for grp_a, grp_b in contrasts:
            col_name = f"{grp_a}_vs_{grp_b}"
            med_a = group_medians[grp_a] + pseudocount
            med_b = group_medians[grp_b] + pseudocount
            lfc_df[col_name] = np.log2(med_a / med_b)

        normalized = lfc_df
        log.info(f"LFC normalization complete: {normalized.shape[0]} features × {normalized.shape[1]} contrasts")

    else:
        # Sample-wise scaling methods
        df = data.copy().astype(float)

        if log2 and method in {"vst", "zscore", "modified_zscore"}:
            df = np.log2(df + 1)

        if method == "vst":
            # Variance-stabilizing transformation: arcsinh(sqrt(x)) then z-score per sample
            df = np.arcsinh(np.sqrt(df.clip(lower=0)))
            normalized = df.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col, axis=0)

        elif method == "zscore":
            # Z-score per sample (column-wise)
            normalized = df.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col, axis=0)

        elif method == "modified_zscore":
            # Modified z-score (median-based) per sample
            def _modified_zscore(col):
                med = col.median()
                mad = (col - med).abs().median()
                if mad > 0:
                    return 0.6745 * (col - med) / mad
                return col - med
            normalized = df.apply(_modified_zscore, axis=0)

        elif method == "rank_normal":
            # Rank-based inverse normal transformation per sample
            arr = quantile_transform(df.values, n_quantiles=min(df.shape[0], 1000),
                                     output_distribution='normal', random_state=0)
            normalized = pd.DataFrame(arr, index=df.index, columns=df.columns)

        log.info(f"{method} normalization complete: {normalized.shape[0]} features × {normalized.shape[1]} samples")

    write_integration_file(normalized, output_dir, output_filename, indexing=True)
    return normalized


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
    metadata_variables: List[str] = ["group"],
    title: str = "PCA Plot",
    output_dir: str = None,):
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
    
        if output_dir:
            # Save the plot if output_dir is specified
            filename = f"{plot_title.replace(' ', '_').replace('-', '_')}.pdf"
            log.info(f"Saving plot to {output_dir}/{filename}...")
            fig.savefig(f"{output_dir}/{filename}")
        
        plt.close(fig)

    # Print sample locations on PCA as table of X,Y coordinates
    coord_df = pca_df[['unique_group', 'PCA1', 'PCA2']]
    print("PCA Sample Coordinates:")
    display(coord_df)

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
        with pd.option_context('future.no_silent_downcasting', True):
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
            if meta_var == "group":
                continue
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
    if "group" in pdf_metadata_vars:
        pdf_metadata_vars.remove("group")
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

def plot_simple_histogram(
    dataframe: pd.DataFrame,
    plot_title: str,
    output_dir: str = None,
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
    ylog: bool = True,
) -> None:

    plt.figure(figsize=(6, 4))
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

    plt.xlabel('Normalized Abundance')
    if xlog:
        plt.xscale('log')
        plt.xlabel('Normalized Abundance (log)')

    plt.ylabel('Frequency')
    if ylog:
        plt.yscale('log')
        plt.ylabel('Frequency (log)')

    plt.title(plot_title)

    if output_dir:
        # Save the plot if output_dir is specified
        filename = f"{plot_title.replace(' ', '_')}.pdf"
        log.info(f"Saving plot to {output_dir}/{filename}...")
        plt.savefig(f"{output_dir}/{filename}")

    plt.show()
    plt.close()


def plot_data_variance_histogram(
    dataframes: dict[str, pd.DataFrame],
    datatype: str,
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
    ylog: bool = False,
    output_dir: str = None,
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

    Returns:
        None
    """

    plt.figure(figsize=(6, 4))
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

    plt.xlabel('Normalized Abundance')
    if xlog:
        plt.xscale('log')
        plt.xlabel('Normalized Abundance (log)')

    plt.ylabel('Frequency')
    if ylog:
        plt.yscale('log')
        plt.ylabel('Frequency (log)')

    plt.title(f'Histogram of Integrated Datasets ({datatype})')

    plt.legend()

    # Save the plot if output_dir is specified
    filename = f"distribution_of_{datatype}_datasets.pdf"
    log.info(f"Saving plot to {output_dir}/{filename}...")
    plt.savefig(f"{output_dir}/{filename}")
    
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
        sns.violinplot(x='color_shape_group', y='abundance', data=linked_data, hue='color_shape_group', palette='viridis', legend=False)
        sns.stripplot(x='color_shape_group', y='abundance', data=linked_data, color='k', alpha=0.5, jitter=True)
        plt.xlabel(f'{color_group} and {shape_group}')
        plt.ylabel('Z-scored abundance')
        plt.title(f'Abundance of {feature} by {color_group} and {shape_group}')
        plt.xticks(rotation=90)
    else:
        # Plot the data
        plt.figure(figsize=(12, 8))
        sns.violinplot(x=metadata_group, y='abundance', data=linked_data, hue=metadata_group, palette='viridis', legend=False)
        sns.stripplot(x=metadata_group, y='abundance', data=linked_data, color='k', alpha=0.5, jitter=True)
        plt.xlabel(metadata_group)
        plt.ylabel('Z-scored abundance')
        plt.title(f'Abundance of {feature} by {metadata_group}')
        plt.xticks(rotation=90)

    # Save before showing
    if save_plot and output_dir:
        output_subdir = f"{output_dir}/boxplots"
        os.makedirs(output_subdir, exist_ok=True)
        filename = f"abundance_of_{feature}_by_{metadata_group}.pdf"
        log.info(f"Saving plot to {output_subdir}/{filename}")
        plt.savefig(f"{output_subdir}/{filename}", bbox_inches='tight')
    
    # Show after saving
    plt.show()
    plt.close()
    
    return

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
        
        plt.figure(figsize=(10, 7))
        sns.violinplot(
            x='color_shape_group', 
            y='mean_abundance', 
            data=linked_data, 
            hue='color_shape_group',
            palette='viridis',
            legend=False
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
        
        if save_plot and output_dir:
            output_subdir = f"{output_dir}/boxplots"
            os.makedirs(output_subdir, exist_ok=True)
            filename = f"avg_abundance_of_{submodule_name}_nodes_by_{metadata_group}.pdf"
            log.info(f"Saving plot to {output_subdir}/{filename}")
            plt.savefig(f"{output_subdir}/{filename}")
        
        plt.show()
    else:
        # Single metadata variable
        plt.figure(figsize=(10, 7))
        sns.violinplot(
            x=metadata_group, 
            y='mean_abundance', 
            data=linked_data, 
            hue=metadata_group,
            palette='viridis',
            legend=False
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
        
        if save_plot and output_dir:
            output_subdir = f"{output_dir}/boxplots"
            os.makedirs(output_subdir, exist_ok=True)
            filename = f"avg_abundance_of_{submodule_name}_nodes_by_{metadata_group}.pdf"
            log.info(f"Saving plot to {output_subdir}/{filename}")
            plt.savefig(f"{output_subdir}/{filename}")
        
        plt.show()

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
        ax = sns.heatmap(subset_corr_matrix, annot=False, cmap='coolwarm', cbar_ax=cbar_ax, vmin=vmin, vmax=vmax, **kwargs)
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

# ===================================
# Pathway-module relatedness
# ===================================

def compute_feature_lfc_scores(quant_df: pd.DataFrame) -> pd.Series:
    """Compute a robust per-feature magnitude-of-change score.

    For each feature (row), takes the median of the absolute value across all
    pairwise comparison columns, ignoring NaNs. Using the median (rather than
    sum or mean) makes this robust to outlier contrasts and avoids biasing
    toward features/pathways with more populated comparison columns.

    Parameters
    ----------
    quant_df : pd.DataFrame
        Feature x comparison table of log fold changes, indexed by feature id.

    Returns
    -------
    pd.Series
        Index: feature id. Values: median(|LFC|) across all comparisons.
    """
    return quant_df.abs().median(axis=1, skipna=True)

def _explode_node_pathways(
    node_df: pd.DataFrame,
    feature_id_col: str | None = None,
    pathway_col: str = "modelseed_pathway",
    submodule_col: str = "submodule",
    pathway_sep: str = ";",
    missing_token: str = "Unassigned",
    exclude_nopathway: bool = False,
    nopathway_prefix: str = "NOPATHWAY_",
) -> pd.DataFrame:
    """Explode node_df's pathway_sep-joined pathway column into long format.

    Returns one row per (feature, pathway) pair, each carrying its submodule.
    Rows with a missing/"Unassigned" pathway or submodule are dropped, since
    they don't contribute to a pathway x submodule grouping.

    This is the single source of truth for (feature, pathway, submodule)
    membership -- both `build_pathway_submodule_matrices` (counts/heatmaps)
    and `compute_pathway_submodule_enrichment` (hypergeometric stats) call
    this same function with the same arguments, guaranteeing the plotted
    matrix and the significance test always agree on what "belongs" where.
    """
    df = node_df.copy()

    if feature_id_col is None:
        df = df.reset_index().rename(columns={df.index.name or "index": "feature_id"})
        feature_id_col = "feature_id"
    elif feature_id_col not in df.columns:
        raise KeyError(f"feature_id_col '{feature_id_col}' not found in node_df columns or index.")

    df = df[[feature_id_col, pathway_col, submodule_col]].dropna(subset=[pathway_col, submodule_col])
    df = df[df[pathway_col].astype(str).str.strip() != ""]
    df = df[df[pathway_col] != missing_token]
    df = df[df[submodule_col] != missing_token]

    exploded = df.assign(
        **{pathway_col: df[pathway_col].str.split(pathway_sep)}
    ).explode(pathway_col)
    exploded[pathway_col] = exploded[pathway_col].str.strip()
    exploded = exploded[exploded[pathway_col] != ""]

    if exclude_nopathway:
        n_before = exploded[pathway_col].nunique()
        exploded = exploded[~exploded[pathway_col].str.startswith(nopathway_prefix)]
        n_after = exploded[pathway_col].nunique()
        log.info(
            f"Excluded {n_before - n_after} placeholder '{nopathway_prefix}*' pathways "
            f"({n_after} real pathways remain)."
        )

    return exploded.rename(
        columns={feature_id_col: "feature_id", pathway_col: "pathway", submodule_col: "submodule"}
    )


def compute_submodule_sizes(
    node_df: pd.DataFrame,
    feature_id_col: str | None = None,
    submodule_col: str = "submodule",
    missing_token: str = "Unassigned",
) -> pd.Series:
    """Count unique features assigned to each submodule.

    Uses each feature's single `submodule_col` assignment directly (not the
    exploded pathway table), since submodule membership is 1:1 per feature --
    counting via the exploded table would inflate sizes for features that
    belong to multiple pathways.

    Parameters
    ----------
    node_df : pd.DataFrame
        Node table with `submodule_col`. Feature id may be the index or a
        named column (see `feature_id_col`).
    feature_id_col : str, optional
        Column name for feature id. If None, uses `node_df.index`.
    missing_token : str
        Sentinel value indicating "no submodule assigned"; excluded from counts.

    Returns
    -------
    pd.Series
        Index: submodule name. Values: number of unique features, sorted descending.
    """
    df = node_df.copy()

    if feature_id_col is None:
        df = df.reset_index().rename(columns={df.index.name or "index": "feature_id"})
        feature_id_col = "feature_id"
    elif feature_id_col not in df.columns:
        raise KeyError(f"feature_id_col '{feature_id_col}' not found in node_df columns or index.")

    df = df.dropna(subset=[submodule_col])
    df = df[df[submodule_col] != missing_token]

    return (
        df.groupby(submodule_col)[feature_id_col]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_features")
    )


def _resolve_qualifying_submodules(
    node_df: pd.DataFrame,
    feature_id_col: str | None,
    submodule_col: str,
    missing_token: str,
    min_submodule_size: int,
) -> list[str] | None:
    """Shared min_submodule_size resolution used by both the matrix builder
    and the enrichment test, so both apply the identical submodule filter.

    Returns None if min_submodule_size <= 0 (no filtering).
    """
    if min_submodule_size <= 0:
        return None

    submodule_sizes = compute_submodule_sizes(
        node_df, feature_id_col=feature_id_col, submodule_col=submodule_col, missing_token=missing_token
    )
    qualifying = submodule_sizes[submodule_sizes >= min_submodule_size].index.tolist()
    n_dropped = submodule_sizes.shape[0] - len(qualifying)
    if n_dropped:
        log.info(
            f"Dropped {n_dropped} submodule(s) with fewer than {min_submodule_size} features "
            f"({len(qualifying)} submodule(s) remain)."
        )
    if not qualifying:
        raise ValueError(
            f"No submodules meet min_submodule_size={min_submodule_size}; "
            f"largest submodule has {submodule_sizes.max() if not submodule_sizes.empty else 0} features."
        )
    return qualifying


def build_pathway_submodule_matrices(
    node_df: pd.DataFrame,
    quant_df: pd.DataFrame | None = None,
    feature_id_col: str | None = None,
    pathway_col: str = "modelseed_pathway",
    submodule_col: str = "submodule",
    pathway_sep: str = ";",
    missing_token: str = "Unassigned",
    exclude_nopathway: bool = False,
    nopathway_prefix: str = "NOPATHWAY_",
    top_n: int = 50,
    rank_by: Literal["n_features", "cumulative_abs_lfc"] = "n_features",
    min_submodule_size: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build pathway x submodule count matrices, restricted to the top-N pathways.

    Parameters
    ----------
    node_df : pd.DataFrame
        Feature-level table with pathway and submodule columns.
    quant_df : pd.DataFrame, optional
        Feature x comparison LFC table, required when `rank_by="cumulative_abs_lfc"`.
    feature_id_col, pathway_col, submodule_col, pathway_sep, missing_token,
    exclude_nopathway, nopathway_prefix : see `_explode_node_pathways`.
    top_n : int
        Number of top-ranked pathways to keep in the returned matrices.
    rank_by : {"n_features", "cumulative_abs_lfc"}
        Ranking metric for pathway selection.
    min_submodule_size : int
        Minimum number of unique features a submodule must contain (across the
        whole `node_df`, independent of pathway membership) to be included in
        the output matrices. Submodules below this threshold are dropped
        entirely -- both their columns AND any features belonging to them --
        before pathway ranking/top-N selection, so the reported pathway stats
        and heatmap are self-consistent. Set to 0 (default) to disable.

    Returns
    -------
    counts_df : pd.DataFrame
        Top-N pathways (rows) x qualifying submodules (columns), raw feature counts.
    row_normalized_df : pd.DataFrame
        Same shape, each row divided by its row sum.
    pathway_stats_df : pd.DataFrame
        Full (pre-top_n-filter) per-pathway stats table, computed only from
        features in qualifying submodules.
    """
    if rank_by == "cumulative_abs_lfc" and quant_df is None:
        raise ValueError("quant_df is required when rank_by='cumulative_abs_lfc'.")

    exploded = _explode_node_pathways(
        node_df,
        feature_id_col=feature_id_col,
        pathway_col=pathway_col,
        submodule_col=submodule_col,
        pathway_sep=pathway_sep,
        missing_token=missing_token,
        exclude_nopathway=exclude_nopathway,
        nopathway_prefix=nopathway_prefix,
    )

    if exploded.empty:
        raise ValueError("No (feature, pathway, submodule) rows remain after filtering; check inputs.")

    qualifying_submodules = _resolve_qualifying_submodules(
        node_df, feature_id_col, submodule_col, missing_token, min_submodule_size
    )
    if qualifying_submodules is not None:
        exploded = exploded[exploded["submodule"].isin(qualifying_submodules)]
        if exploded.empty:
            raise ValueError(
                "No (feature, pathway, submodule) rows remain after applying min_submodule_size filter."
            )

    # --- Per-pathway stats (computed only from features in qualifying submodules) ---
    stats = exploded.groupby("pathway")["feature_id"].nunique().rename("n_features").to_frame()

    if rank_by == "cumulative_abs_lfc" or quant_df is not None:
        feature_scores = compute_feature_lfc_scores(quant_df)
        missing_features = set(exploded["feature_id"]) - set(feature_scores.index)
        if missing_features:
            log.warning(
                f"{len(missing_features)} features in node_df are missing from quant_df "
                f"and will be scored as 0 for cumulative_abs_lfc ranking "
                f"(e.g. {sorted(missing_features)[:5]})."
            )
        exploded = exploded.assign(
            feature_lfc_score=exploded["feature_id"].map(feature_scores).fillna(0.0)
        )
        cum_lfc = exploded.groupby("pathway")["feature_lfc_score"].sum().rename("cumulative_abs_lfc")
        stats = stats.join(cum_lfc, how="left")

    pathway_stats_df = stats.reset_index().sort_values(rank_by, ascending=False).reset_index(drop=True)

    # --- Select top N pathways, then build the crosstab ---
    top_pathways = pathway_stats_df.head(top_n)["pathway"].tolist()
    exploded_top = exploded[exploded["pathway"].isin(top_pathways)]

    counts_df = pd.crosstab(exploded_top["pathway"], exploded_top["submodule"])
    counts_df = counts_df.reindex(index=top_pathways)  # preserve rank order

    row_normalized_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    log.info(
        f"Built pathway x submodule matrix: {counts_df.shape[0]} pathways "
        f"(top {top_n} by {rank_by}) x {counts_df.shape[1]} submodules "
        f"(min_submodule_size={min_submodule_size})."
    )

    return counts_df, row_normalized_df, pathway_stats_df


def compute_pathway_submodule_enrichment(
    node_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    feature_id_col: str | None = None,
    pathway_col: str = "modelseed_pathway",
    submodule_col: str = "submodule",
    pathway_sep: str = ";",
    missing_token: str = "Unassigned",
    exclude_nopathway: bool = False,
    nopathway_prefix: str = "NOPATHWAY_",
    min_submodule_size: int = 0,
    fdr_method: str = "fdr_bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Hypergeometric over-representation test for the pathway x submodule
    pairs in `counts_df`.

    Tests whether a pathway's features are enriched within a given submodule
    more than expected by chance, given the pathway size, module size, and
    background universe -- correcting for the size confounds that a raw
    concentration heatmap cannot account for.

    Re-derives the same exploded (feature, pathway, submodule) table and
    min_submodule_size filtering as `build_pathway_submodule_matrices`, so
    background sizes (M, n, N) are computed on an identical universe to
    whatever produced `counts_df`. Pass the SAME feature_id_col/pathway_col/
    submodule_col/pathway_sep/missing_token/exclude_nopathway/nopathway_prefix/
    min_submodule_size arguments you used to build `counts_df`, or the two
    will silently diverge.

    Parameters
    ----------
    node_df : pd.DataFrame
        The same raw node table passed to `build_pathway_submodule_matrices`.
    counts_df : pd.DataFrame
        Output of `build_pathway_submodule_matrices` -- top-N pathways (rows)
        x qualifying submodules (columns). Enrichment is computed only for
        these cells; `k` (overlap) is read directly from `counts_df` rather
        than recomputed, guaranteeing exact agreement with the plotted matrix.
    fdr_method : str
        Passed to `statsmodels.stats.multitest.multipletests` (default
        Benjamini-Hochberg FDR).
    alpha : float
        Significance threshold used to populate the `significant` column.

    Returns
    -------
    pd.DataFrame with columns: pathway, submodule, k, n, N, M, expected,
    fold_enrichment, pvalue, padj, significant -- one row per (pathway,
    submodule) cell in `counts_df`. FDR correction is applied across exactly
    these cells (not the full, untested pathway universe).
    """
    exploded = _explode_node_pathways(
        node_df,
        feature_id_col=feature_id_col,
        pathway_col=pathway_col,
        submodule_col=submodule_col,
        pathway_sep=pathway_sep,
        missing_token=missing_token,
        exclude_nopathway=exclude_nopathway,
        nopathway_prefix=nopathway_prefix,
    )

    qualifying_submodules = _resolve_qualifying_submodules(
        node_df, feature_id_col, submodule_col, missing_token, min_submodule_size
    )
    if qualifying_submodules is not None:
        exploded = exploded[exploded["submodule"].isin(qualifying_submodules)]

    universe = exploded[["feature_id", "submodule"]].drop_duplicates()
    M = universe["feature_id"].nunique()
    module_sizes = universe.groupby("submodule")["feature_id"].nunique()
    pathway_sizes = exploded.groupby("pathway")["feature_id"].nunique()

    records = []
    for pathway in counts_df.index:
        n = pathway_sizes.get(pathway, 0)
        for submodule in counts_df.columns:
            N = module_sizes.get(submodule, 0)
            if n == 0 or N == 0:
                continue
            k = int(counts_df.loc[pathway, submodule])
            # P(X >= k): survival function is P(X > x), so use sf(k - 1)
            pval = hypergeom.sf(k - 1, M, n, N)
            expected = n * N / M
            records.append({
                "pathway": pathway,
                "submodule": submodule,
                "k": k, "n": n, "N": N, "M": M,
                "expected": expected,
                "fold_enrichment": (k / expected) if expected > 0 else np.nan,
                "pvalue": pval,
            })

    result = pd.DataFrame.from_records(records)
    reject, padj, _, _ = multipletests(result["pvalue"], alpha=alpha, method=fdr_method)
    result["padj"] = padj
    result["significant"] = reject

    log.info(
        f"Computed hypergeometric enrichment for {len(result)} pathway x submodule "
        f"pairs; {result['significant'].sum()} significant at padj<{alpha} ({fdr_method})."
    )

    return result.sort_values("padj").reset_index(drop=True)

def add_significance_annotations(
    clustergrid: sns.matrix.ClusterGrid,
    matrix_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    cluster_rows: bool,
    cluster_cols: bool,
    alpha_stars: dict[float, str] | None = None,
) -> None:
    """Overlay hypergeometric significance stars on a ClusterGrid's heatmap Axes.

    Must be called BEFORE the ClusterGrid's `.figure` is extracted/returned --
    dendrogram reordering (`reordered_ind`) only exists on the ClusterGrid
    object itself, not on the plain Figure.

    Parameters
    ----------
    clustergrid : sns.matrix.ClusterGrid
        Object returned directly by `sns.clustermap(...)`, before `.figure`
        is accessed.
    matrix_df : pd.DataFrame
        The exact dataframe passed into that `sns.clustermap` call
        (`counts_df` or `row_normalized_df`) -- used to map reordered
        positions back to (pathway, submodule) labels.
    enrichment_df : pd.DataFrame
        Output of `compute_pathway_submodule_enrichment`.
    cluster_rows, cluster_cols : bool
        Whether clustering was enabled for this call. When False,
        `clustergrid.dendrogram_row`/`dendrogram_col` is None, so the
        original (unreordered) index/column order is used instead.
    alpha_stars : dict[float, str], optional
        Mapping of padj thresholds to marker strings, checked tightest-first.
        Defaults to {0.001: "***", 0.01: "**", 0.05: "*"}.
    """
    if alpha_stars is None:
        alpha_stars = {0.001: "***", 0.01: "**", 0.05: "*"}

    padj_lookup = enrichment_df.set_index(["pathway", "submodule"])["padj"]

    row_order = (
        clustergrid.dendrogram_row.reordered_ind if cluster_rows else range(len(matrix_df.index))
    )
    col_order = (
        clustergrid.dendrogram_col.reordered_ind if cluster_cols else range(len(matrix_df.columns))
    )

    ordered_rows = matrix_df.index[list(row_order)]
    ordered_cols = matrix_df.columns[list(col_order)]

    ax = clustergrid.ax_heatmap
    stroke = [pe.withStroke(linewidth=1.5, foreground="black")]

    for i, pathway in enumerate(ordered_rows):
        for j, submodule in enumerate(ordered_cols):
            padj = padj_lookup.get((pathway, submodule), np.nan)
            if pd.isna(padj):
                continue
            for thresh, stars in sorted(alpha_stars.items()):
                if padj < thresh:
                    ax.text(
                        j + 0.5, i + 0.7, stars,
                        ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold",
                        path_effects=stroke,
                    )
                    break


def plot_pathway_submodule_clustermaps(
    counts_df: pd.DataFrame,
    row_normalized_df: pd.DataFrame,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    method: str = "average",
    metric: str = "euclidean",
    counts_cmap: str = "viridis",
    normalized_cmap: str = "magma",
    normalized_vmax: float | Literal["auto"] = "auto",
    normalized_gamma: float | None = None,
    figsize: tuple[float, float] = (16, 14),
    output_dir: str | Path | None = None,
    counts_filename: str = "pathway_submodule_heatmap_counts.png",
    normalized_filename: str = "pathway_submodule_heatmap_normalized.png",
    dpi: int = 300,
    enrichment_df: pd.DataFrame | None = None,
    alpha_stars: dict[float, str] | None = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Plot raw-count and row-normalized pathway x submodule heatmaps with shared clustering.

    Parameters
    ----------
    counts_df, row_normalized_df : pd.DataFrame
        Outputs of `build_pathway_submodule_matrices`.
    cluster_rows, cluster_cols : bool
        Whether to hierarchically cluster rows/columns (shared linkage
        computed from `row_normalized_df`, applied to both panels so pathway/
        submodule ordering is identical across the two figures).
    method, metric : str
        Passed to `scipy.cluster.hierarchy.linkage`.
    counts_cmap, normalized_cmap : str
        Colormaps for each panel.
    normalized_vmax : float or "auto"
        Upper bound of the color scale for the row-normalized heatmap.
        "auto" (default) uses the actual maximum value in `row_normalized_df`,
        so the color range reflects your real data spread instead of the full
        theoretical 0-1 range (which is misleading when most cells are small,
        e.g. <0.1, and only a few approach 1.0). Pass a fixed float instead if
        you want a stable scale across multiple runs/comparisons.
    normalized_gamma : float, optional
        If set, applies a `matplotlib.colors.PowerNorm(gamma=normalized_gamma)`
        to the color mapping instead of a plain linear scale. Values < 1
        (e.g. 0.3-0.5) compress the high end and stretch the low end, making
        small proportions visually distinguishable instead of all reading as
        "dark/near-zero" under a linear scale dominated by a handful of large
        values. Leave as None for a standard linear scale.
    output_dir, counts_filename, normalized_filename, dpi :
        Figure-saving options. If `output_dir` is None, figures are not saved.
    enrichment_df : pd.DataFrame, optional
        Output of `compute_pathway_submodule_enrichment`. If provided,
        significance stars are overlaid on both panels (annotation happens
        while each ClusterGrid is still live, before `.figure` is extracted).
    alpha_stars : dict[float, str], optional
        padj threshold -> marker string for the significance overlay. Only
        used if `enrichment_df` is provided.

    Returns
    -------
    (fig_counts, fig_normalized) : tuple[plt.Figure, plt.Figure]
    """
    row_linkage = linkage(row_normalized_df.values, method=method, metric=metric) if cluster_rows else None
    col_linkage = linkage(row_normalized_df.values.T, method=method, metric=metric) if cluster_cols else None

    common_kwargs = dict(
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        figsize=figsize,
        dendrogram_ratio=(0.15, 0.1),
        cbar_pos=(0.02, 0.83, 0.03, 0.15),
    )

    # --- Counts panel ---
    g_counts = sns.clustermap(
        counts_df,
        cmap=counts_cmap,
        annot=False,
        fmt="d",
        **common_kwargs,
    )
    g_counts.ax_heatmap.set_xlabel("Submodule")
    g_counts.ax_heatmap.set_ylabel("Pathway")

    counts_title = "Feature count per pathway x submodule"
    if enrichment_df is not None:
        # Annotate while g_counts is still a ClusterGrid -- reordered_ind
        # is unavailable once only `.figure` is kept.
        add_significance_annotations(
            g_counts, counts_df, enrichment_df, cluster_rows, cluster_cols, alpha_stars
        )
        counts_title += "\n* padj<0.05  ** padj<0.01  *** padj<0.001 (hypergeometric, BH-FDR)"
    g_counts.figure.suptitle(counts_title, y=1.04 if enrichment_df is not None else 1.02, fontsize=10)

    # --- Row-normalized panel ---
    resolved_vmax = row_normalized_df.values.max() if normalized_vmax == "auto" else normalized_vmax

    norm_kwargs: dict = {}
    if normalized_gamma is not None:
        # PowerNorm handles vmin/vmax itself; don't pass them separately alongside `norm`
        norm_kwargs["norm"] = PowerNorm(gamma=normalized_gamma, vmin=0, vmax=resolved_vmax)
    else:
        norm_kwargs["vmin"] = 0
        norm_kwargs["vmax"] = resolved_vmax

    g_norm = sns.clustermap(
        row_normalized_df,
        cmap=normalized_cmap,
        annot=False,
        fmt=".2f",
        **norm_kwargs,
        **common_kwargs,
    )
    g_norm.ax_heatmap.set_xlabel("Submodule")
    g_norm.ax_heatmap.set_ylabel("Pathway")

    if enrichment_df is not None:
        add_significance_annotations(
            g_norm, row_normalized_df, enrichment_df, cluster_rows, cluster_cols, alpha_stars
        )

    g_norm.figure.suptitle(
        f"Row-normalized fraction of pathway's features per submodule "
        f"(scale: 0-{resolved_vmax:.2f})",
        y=1.02,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        g_counts.figure.savefig(output_dir / counts_filename, dpi=dpi, bbox_inches="tight")
        g_norm.figure.savefig(output_dir / normalized_filename, dpi=dpi, bbox_inches="tight")
        log.info(f"Saved heatmaps to {output_dir / counts_filename} and {output_dir / normalized_filename}")

    return g_counts.figure, g_norm.figure

def generate_pathway_submodule_heatmap(
    node_df: pd.DataFrame,
    quant_df: pd.DataFrame | None = None,
    top_n: int = 50,
    rank_by: Literal["n_features", "cumulative_abs_lfc"] = "n_features",
    exclude_nopathway: bool = False,
    min_submodule_size: int = 0,
    normalized_vmax: float | Literal["auto"] = "auto",
    normalized_gamma: float | None = None,
    output_dir: str | Path | None = None,
    fdr_method: str = "fdr_bh",
    alpha: float = 0.05,
    **plot_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, plt.Figure, plt.Figure]:
    """End-to-end: build pathway x submodule matrices, run hypergeometric
    enrichment, and plot both heatmap panels with significance stars overlaid.

    See `build_pathway_submodule_matrices`, `compute_pathway_submodule_enrichment`,
    and `plot_pathway_submodule_clustermaps` for parameter details.

    Parameters
    ----------
    node_df, quant_df, top_n, rank_by, exclude_nopathway, min_submodule_size :
        See `build_pathway_submodule_matrices`.
    normalized_vmax, normalized_gamma :
        See `plot_pathway_submodule_clustermaps`.
    output_dir :
        Directory to save figures to. If None, figures are not saved to disk.
    fdr_method : str
        Multiple-testing correction method for `compute_pathway_submodule_enrichment`
        (default Benjamini-Hochberg FDR).
    alpha : float
        Significance threshold used to populate `enrichment_df["significant"]`
        and the star overlay.
    **plot_kwargs
        Any additional keyword arguments forwarded to
        `plot_pathway_submodule_clustermaps` (e.g. `method`, `metric`,
        `counts_cmap`, `figsize`, `counts_filename`, `dpi`, `alpha_stars`, etc.).

    Returns
    -------
    (counts_df, row_normalized_df, pathway_stats_df, enrichment_df, fig_counts, fig_normalized)
    """
    counts_df, row_normalized_df, pathway_stats_df = build_pathway_submodule_matrices(
        node_df,
        quant_df=quant_df,
        top_n=top_n,
        rank_by=rank_by,
        exclude_nopathway=exclude_nopathway,
        min_submodule_size=min_submodule_size,
    )

    enrichment_df = compute_pathway_submodule_enrichment(
        node_df,
        counts_df,
        exclude_nopathway=exclude_nopathway,
        min_submodule_size=min_submodule_size,
        fdr_method=fdr_method,
        alpha=alpha,
    )

    fig_counts, fig_normalized = plot_pathway_submodule_clustermaps(
        counts_df,
        row_normalized_df,
        normalized_vmax=normalized_vmax,
        normalized_gamma=normalized_gamma,
        output_dir=output_dir,
        enrichment_df=enrichment_df,
        **plot_kwargs,
    )

    return counts_df, row_normalized_df, pathway_stats_df, enrichment_df, fig_counts, fig_normalized


def compare_groups_to_pathways(
    node_table: pd.DataFrame,
    annotation_table: pd.DataFrame,
    quant_df: pd.DataFrame | None = None,
    pathway_col: str = "modelseed_pathway",
    group_col: str = "group",
    top_n: int = 50,
    rank_by: Literal["n_features", "cumulative_abs_lfc"] = "n_features",
    exclude_nopathway: bool = False,
    min_group_size: int = 3,
    fdr_method: str = "fdr_bh",
    alpha: float = 0.05,
    output_dir: str | None = None,
    **plot_kwargs,
) -> Dict[str, Any]:
    """
    Compare data-driven feature groups against knowledge-driven pathway annotations.

    Combines two complementary analyses into a single call:

    1. **Pathway × group heatmap** — builds a pathway × group count matrix
       (top ``top_n`` pathways by feature count or cumulative |LFC|), runs
       hypergeometric over-representation tests (BH-FDR corrected), and plots
       both a raw-count and a row-normalised clustermap with significance stars.

    2. **Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI)** —
       single-number summaries of how well the data-driven grouping recovers
       pathway structure.  Features with no pathway annotation or assigned to
       the HDBSCAN noise class (``"noise"``) are excluded from the ARI/NMI
       calculation.

    Parameters
    ----------
    node_table : pd.DataFrame
        Feature-level node table produced by ``group_features()`` (or any of
        the three grouping backends).  Must be indexed by feature ID and contain
        a ``group`` column (and optionally a ``submodule`` column — both are
        accepted).
    annotation_table : pd.DataFrame
        Feature annotation table indexed by feature ID.  Must contain
        ``pathway_col`` (default ``"modelseed_pathway"``).
    quant_df : pd.DataFrame, optional
        Feature × comparison LFC matrix.  Required only when
        ``rank_by="cumulative_abs_lfc"``.
    pathway_col : str
        Column in ``annotation_table`` that holds pathway labels.  Pathways may
        be semicolon-separated; the first token is used as the primary pathway
        for ARI/NMI.
    group_col : str
        Column in ``node_table`` that holds the data-driven group labels.
        Defaults to ``"group"``.  Falls back to ``"submodule"`` if ``"group"``
        is absent.
    top_n : int
        Number of top-ranked pathways to show in the heatmap.
    rank_by : {"n_features", "cumulative_abs_lfc"}
        Pathway ranking metric.
    exclude_nopathway : bool
        If True, features with no pathway annotation are excluded from the
        heatmap (they are always excluded from ARI/NMI).
    min_group_size : int
        Groups with fewer than this many features are dropped from both the
        heatmap and the ARI/NMI calculation.
    fdr_method : str
        Multiple-testing correction method (default ``"fdr_bh"``).
    alpha : float
        Significance threshold for the enrichment test and star overlay.
    output_dir : str, optional
        Directory to save heatmap PDFs.  If ``None``, figures are displayed
        only.
    **plot_kwargs
        Forwarded to ``plot_pathway_submodule_clustermaps`` (e.g. ``figsize``,
        ``counts_cmap``, ``dpi``).

    Returns
    -------
    dict with keys:
        ``counts_df``         – pathway × group raw-count matrix
        ``row_normalized_df`` – row-normalised version of ``counts_df``
        ``pathway_stats_df``  – per-pathway stats (n_features, optional LFC score)
        ``enrichment_df``     – hypergeometric enrichment results
        ``fig_counts``        – matplotlib Figure (raw counts)
        ``fig_normalized``    – matplotlib Figure (row-normalised)
        ``ari``               – Adjusted Rand Index (float)
        ``nmi``               – Normalised Mutual Information (float)
        ``n_features_compared`` – number of features used for ARI/NMI
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import LabelEncoder

    # ------------------------------------------------------------------ #
    # 1. Resolve group column                                              #
    # ------------------------------------------------------------------ #
    if group_col not in node_table.columns:
        if "submodule" in node_table.columns:
            log.info(
                f"Column '{group_col}' not found in node_table; "
                "falling back to 'submodule'."
            )
            group_col = "submodule"
        else:
            raise ValueError(
                f"Neither '{group_col}' nor 'submodule' found in node_table columns: "
                f"{node_table.columns.tolist()}"
            )

    # ------------------------------------------------------------------ #
    # 2. Merge pathway annotation onto node table                         #
    # ------------------------------------------------------------------ #
    node_df = node_table.copy()

    # Ensure 'submodule' column exists for generate_pathway_submodule_heatmap
    if "submodule" not in node_df.columns:
        node_df["submodule"] = node_df[group_col]

    if pathway_col not in node_df.columns:
        if pathway_col not in annotation_table.columns:
            raise ValueError(
                f"Pathway column '{pathway_col}' not found in annotation_table. "
                f"Available columns: {annotation_table.columns.tolist()}"
            )
        node_df = node_df.join(annotation_table[[pathway_col]], how="left")

    log.info(
        f"Annotated node table: {node_df.shape[0]} features, "
        f"{node_df[group_col].nunique()} groups, "
        f"{node_df[pathway_col].nunique()} unique pathways"
    )

    # ------------------------------------------------------------------ #
    # 3. Pathway × group heatmap + hypergeometric enrichment              #
    # ------------------------------------------------------------------ #
    log.info("Building pathway × group heatmap and running enrichment tests...")
    (
        counts_df,
        row_normalized_df,
        pathway_stats_df,
        enrichment_df,
        fig_counts,
        fig_normalized,
    ) = generate_pathway_submodule_heatmap(
        node_df=node_df,
        quant_df=quant_df,
        top_n=top_n,
        rank_by=rank_by,
        exclude_nopathway=exclude_nopathway,
        min_submodule_size=min_group_size,
        output_dir=output_dir,
        fdr_method=fdr_method,
        alpha=alpha,
        **plot_kwargs,
    )

    n_sig = int(enrichment_df["significant"].sum()) if not enrichment_df.empty else 0
    log.info(
        f"Significant pathway-group enrichments (padj < {alpha}): "
        f"{n_sig} of {len(enrichment_df)} tested pairs"
    )

    # ------------------------------------------------------------------ #
    # 4. ARI and NMI                                                      #
    # ------------------------------------------------------------------ #
    comparison_df = node_df[[group_col, pathway_col]].copy()

    # Drop features with no pathway or in the noise class
    comparison_df = comparison_df[
        comparison_df[pathway_col].notna()
        & (comparison_df[pathway_col].astype(str) != "Unassigned")
        & (comparison_df[pathway_col].astype(str) != "")
        & comparison_df[group_col].notna()
        & (comparison_df[group_col].astype(str) != "noise")
    ]

    # Apply min_group_size filter
    if min_group_size > 0:
        group_sizes = comparison_df[group_col].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index
        comparison_df = comparison_df[comparison_df[group_col].isin(valid_groups)]

    ari, nmi = np.nan, np.nan
    n_compared = len(comparison_df)

    if n_compared >= 2:
        # Use primary pathway (first token if semicolon-separated)
        comparison_df = comparison_df.copy()
        comparison_df["_primary_pathway"] = (
            comparison_df[pathway_col].astype(str).str.split(";").str[0].str.strip()
        )

        le_group = LabelEncoder()
        le_pathway = LabelEncoder()
        group_labels = le_group.fit_transform(comparison_df[group_col].astype(str))
        pathway_labels = le_pathway.fit_transform(comparison_df["_primary_pathway"])

        ari = float(adjusted_rand_score(pathway_labels, group_labels))
        nmi = float(normalized_mutual_info_score(pathway_labels, group_labels))

        log.info(f"Features used for ARI/NMI comparison : {n_compared:,}")
        log.info(f"Data-driven groups                   : {comparison_df[group_col].nunique()}")
        log.info(f"Unique pathways (primary)             : {comparison_df['_primary_pathway'].nunique()}")
        log.info(f"Adjusted Rand Index (ARI)             : {ari:.4f}  (1=perfect, 0=random)")
        log.info(f"Normalised Mutual Info (NMI)          : {nmi:.4f}  (1=perfect, 0=none)")
    else:
        log.warning(
            f"Only {n_compared} features remain after filtering for ARI/NMI — "
            "skipping calculation."
        )

    return {
        "counts_df": counts_df,
        "row_normalized_df": row_normalized_df,
        "pathway_stats_df": pathway_stats_df,
        "enrichment_df": enrichment_df,
        "fig_counts": fig_counts,
        "fig_normalized": fig_normalized,
        "ari": ari,
        "nmi": nmi,
        "n_features_compared": n_compared,
    }