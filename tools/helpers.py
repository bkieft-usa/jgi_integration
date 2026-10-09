from __future__ import annotations

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
import time
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from tqdm.auto import tqdm
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
from typing import List, Tuple, Union, Optional, Dict, Any, Callable
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
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from plotly.subplots import make_subplots
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib import cm

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

# --- Pure preprocessing functions ---
import tools.preprocessing as prep

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
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Select the top ``max_features`` most variable features by row-wise variance.

    Parameters
    ----------
    data : pd.DataFrame
        Feature-by-sample (or feature-by-condition) matrix.
    max_features : int
        Number of top-variance features to retain.

    Returns
    -------
    pd.DataFrame
        Subset of ``data`` containing only the top-variance features.
    """
    var_series = data.var(axis=1).sort_values(ascending=False)
    top_idx = var_series.index[:max_features]
    log.info(f"Variance selection: keeping top {max_features} most variable features.")
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
    datasets: List = None,
) -> pd.DataFrame:
    """
    Apply feature selection to the integrated matrix.

    Supported methods (set ``analysis.feature_selection.method`` in ``analysis.yml``):

    * ``"variance"``     — keep the most variable features. When ``datasets`` is given, each
      dataset's mean-adjusted log-scale variance (from ``replicate_filtered_data``) is ranked
      separately within a shared ``max_features`` budget; the z-scored integrated matrix itself
      has uniform variance and cannot be ranked. Selected blocks are then weighted so every
      dataset carries equal total variance (disable with ``params.block_scale: false``).
    * ``"feature_list"`` — keep only features present in a user-supplied text file.

    Parameters
    ----------
    data : pd.DataFrame
        Integrated feature matrix (features x samples or features x conditions).
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
        if datasets:
            blocks = {}
            for ds in datasets:
                df = ds.replicate_filtered_data.copy()
                if not df.index.astype(str).str.startswith(f"{ds.dataset_name}_").all():
                    df.index = [f"{ds.dataset_name}_{i}" for i in df.index]
                blocks[ds.dataset_name] = df.loc[df.index.isin(data.index)]
            chosen = prep.select_top_variable(blocks, max_features)
            selected = [f for feats in chosen.values() for f in feats]
            subset = data.loc[selected]
            if params_block.get("block_scale", True):
                subset, weights = prep.block_scale(subset, tuple(f"{ds.dataset_name}_" for ds in datasets))
                log.info(f"Block weights applied per dataset: {weights}")
        else:
            subset = variance_selection(data, max_features=max_features)
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
        For Pearson / Spearman   → 1/(n_samples-1)
        For Bicor / Cosine / Centered-Cosine   → 1
    """
    n = X.shape[0]

    if method == "bicor":
        # Biweight midcorrelation: median/MAD-based weights, then unit-norm columns
        med = np.median(X, axis=0, keepdims=True)
        dev = X - med
        mad = np.median(np.abs(dev), axis=0, keepdims=True)
        mad[mad == 0] = np.inf  # constant feature -> zero vector
        u = dev / (9.0 * mad)
        w = (1.0 - u ** 2) ** 2 * (np.abs(u) < 1.0)
        Xt = dev * w
        norm = np.linalg.norm(Xt, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        return Xt / norm, 1.0

    if method == "spearman":
        X = rankdata(X, axis=0)

    if method in ["pearson", "spearman"]:
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
    palette = cm.viridis(np.linspace(0, 1, max(3, len(prefixes))))
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
    show_network_plot: bool = False,
) -> None:
    """
    Build a correlation graph from a long-format table, optionally
    detect submodules (connected components, Louvain, Leiden)
    and write everything to disk.

    Parameters are identical to your original function; the only
    behavioural change is the expanded ``submodule_mode`` options.

    datasets : List, optional
        List of dataset objects with annotation_table attributes
    show_network_plot : bool, default False
        If True, render the interactive Plotly network widget inline in the
        notebook.  Set to False (default) to skip rendering and save time
        when running the workflow programmatically.
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

    # interactive plot (opt-in)
    if show_network_plot:
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
    else:
        log.info("Network plot skipped (show_network_plot=False). Set show_network_plot=True to render inline.")

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


def group_features_nmf(
    data: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 50,
    n_runs: int = 3,
    max_iter: int = 500,
    output_dir: str = None,
    output_filenames: Dict[str, str] = None,
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    integrated_data: pd.DataFrame = None,
    integrated_metadata: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group features using Non-negative Matrix Factorization (NMF) with automatic
    component selection via reconstruction-error elbow detection.

    The input matrix is shifted to be non-negative (``X − min(X)``) before
    factorization so that LFC or Z-score values are handled correctly.

    For each candidate k in ``[k_min, k_max]``, NMF is run ``n_runs`` times
    and the median reconstruction error (Frobenius norm of ``X − WH``) is
    recorded.  The optimal k is chosen at the elbow of the error curve using
    the maximum-distance method (largest perpendicular distance from the line
    connecting the first and last points).

    Each feature is assigned to the component for which its W loading is
    highest.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix (features x columns).  Can contain negative values
        (LFC or Z-score); a non-negative shift is applied internally.
    k_min : int
        Minimum number of NMF components to evaluate (default 2).
    k_max : int
        Maximum number of NMF components to evaluate (default 50).
    n_runs : int
        Number of random restarts per k for stability (default 3).
    max_iter : int
        Maximum NMF iterations per run (default 500).
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys ``"node_table"`` and ``"edge_table"``.
    datasets : List
        Dataset objects with a ``.dataset_name`` attribute (used for coloring).
    annotation_df : pd.DataFrame, optional
        Feature annotation table; merged on feature ID if provided.
    integrated_data : pd.DataFrame
        Accepted for API consistency; not used.
    integrated_metadata : pd.DataFrame
        Accepted for API consistency; not used.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(node_table, edge_table)`` where ``node_table`` is indexed by
        feature ID with a ``'group'`` column, and ``edge_table`` is empty.
    """
    from sklearn.decomposition import NMF as _NMF

    log.info("Grouping features by NMF with automatic k selection (elbow method)...")

    # Drop non-numeric / all-NaN rows
    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
    numeric_data = numeric_data.loc[~(numeric_data == 0).all(axis=1)]

    # Shift to non-negative
    X_min = numeric_data.values.min()
    X = (numeric_data.values - X_min).astype(np.float64)

    k_max_eff = min(k_max, X.shape[0] - 1, X.shape[1] - 1)
    if k_max_eff < k_min:
        raise ValueError(
            f"k_max_eff={k_max_eff} < k_min={k_min}. "
            "Reduce k_min or provide a larger feature matrix."
        )

    k_range = list(range(k_min, k_max_eff + 1))
    errors: List[float] = []

    log.info(f"  Evaluating k = {k_min}…{k_max_eff} ({n_runs} runs each)...")
    for k in k_range:
        run_errors = []
        for _ in range(n_runs):
            model = _NMF(
                n_components=k,
                init="nndsvda",
                max_iter=max_iter,
                random_state=None,
            )
            W = model.fit_transform(X)
            H = model.components_
            err = np.linalg.norm(X - W @ H, "fro")
            run_errors.append(err)
        errors.append(float(np.median(run_errors)))
        log.info(f"    k={k}: median reconstruction error = {errors[-1]:.4f}")

    # Elbow detection: maximum perpendicular distance from the line
    # connecting (k_min, errors[0]) → (k_max_eff, errors[-1])
    pts = np.array(list(zip(k_range, errors)), dtype=float)
    p1, p2 = pts[0], pts[-1]
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        best_k = k_min
    else:
        unit = line_vec / line_len
        dists = np.abs(np.cross(unit, pts - p1))
        best_k = k_range[int(np.argmax(dists))]

    log.info(f"  Elbow detected at k={best_k}. Fitting final NMF model...")

    # Final fit with best_k (multiple restarts, keep lowest error)
    best_err = np.inf
    best_W: Optional[np.ndarray] = None
    for _ in range(max(n_runs, 5)):
        model = _NMF(
            n_components=best_k,
            init="nndsvda",
            max_iter=max_iter,
            random_state=None,
        )
        W = model.fit_transform(X)
        H = model.components_
        err = np.linalg.norm(X - W @ H, "fro")
        if err < best_err:
            best_err = err
            best_W = W

    # Assign each feature to its highest-loading component
    component_assignments = np.argmax(best_W, axis=1)
    group_labels = [f"group_{c + 1}" for c in component_assignments]

    feature_ids = numeric_data.index.tolist()
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

    if annotation_df is not None and not annotation_df.empty:
        ann = annotation_df.copy()
        if 'feature_id' in ann.columns:
            ann = ann.set_index('feature_id')
        ann_cols = [c for c in ann.columns if c not in node_table.columns]
        node_table = node_table.join(ann[ann_cols], how='left')

    n_groups = node_table['group'].nunique()
    log.info(f"  NMF grouping complete: k={best_k}, {n_groups} groups across {len(node_table)} features.")

    write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
    empty_edge_table = pd.DataFrame(columns=['source', 'target', 'weight'])
    write_integration_file(data=empty_edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")

    return node_table, empty_edge_table


def group_features_leiden_knn(
    data: pd.DataFrame,
    n_neighbors: int = 15,
    resolution_min: float = 0.1,
    resolution_max: float = 2.0,
    resolution_steps: int = 20,
    output_dir: str = None,
    output_filenames: Dict[str, str] = None,
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    integrated_data: pd.DataFrame = None,
    integrated_metadata: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group features using Leiden community detection on an approximate
    k-nearest-neighbor graph built from the feature row vectors.

    The k-NN graph is constructed with ``pynndescent`` (O(n log n)) using
    cosine distance, which is appropriate for both Z-scored and LFC vectors.
    Leiden is then run over a sweep of resolution values; the resolution that
    maximises graph modularity is selected automatically.

    Parameters
    ----------
    data : pd.DataFrame
        Feature matrix (features x columns).
    n_neighbors : int
        Number of nearest neighbors per feature (default 15).
    resolution_min : float
        Lower bound of the resolution sweep (default 0.1).
    resolution_max : float
        Upper bound of the resolution sweep (default 2.0).
    resolution_steps : int
        Number of resolution values to evaluate (default 20).
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys ``"node_table"`` and ``"edge_table"``.
    datasets : List
        Dataset objects with a ``.dataset_name`` attribute.
    annotation_df : pd.DataFrame, optional
        Feature annotation table; merged on feature ID if provided.
    integrated_data : pd.DataFrame
        Accepted for API consistency; not used.
    integrated_metadata : pd.DataFrame
        Accepted for API consistency; not used.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(node_table, edge_table)`` where ``node_table`` is indexed by
        feature ID with a ``'group'`` column, and ``edge_table`` is empty.
    """
    try:
        from pynndescent import NNDescent
    except ImportError:
        raise ImportError(
            "The 'pynndescent' package is required for group_features_leiden_knn(). "
            "Install it with: pip install pynndescent"
        )

    log.info("Grouping features by Leiden community detection on k-NN graph...")

    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
    numeric_data = numeric_data.loc[~(numeric_data == 0).all(axis=1)]
    X = numeric_data.values.astype(np.float64)
    n_features = X.shape[0]

    k_eff = min(n_neighbors, n_features - 1)
    if k_eff < 2:
        raise ValueError(
            f"Need at least 3 features for k-NN graph; got {n_features}."
        )

    # Build approximate k-NN graph
    log.info(f"  Building approximate k-NN graph (n_neighbors={k_eff}, metric='cosine')...")
    index = NNDescent(X, metric="cosine", n_neighbors=k_eff + 1, random_state=42)
    indices, distances = index.neighbor_graph  # shape (n, k+1); col 0 = self

    # Convert to igraph for Leiden
    edges: List[Tuple[int, int]] = []
    weights: List[float] = []
    for i in range(n_features):
        for j_pos in range(1, indices.shape[1]):  # skip self (col 0)
            j = int(indices[i, j_pos])
            if j > i:  # upper triangle only → undirected
                sim = max(0.0, 1.0 - float(distances[i, j_pos]))  # cosine similarity
                edges.append((i, j))
                weights.append(sim)

    ig_g = ig.Graph(n=n_features, edges=edges, directed=False)
    ig_g.es["weight"] = weights
    log.info(f"  k-NN graph: {n_features} nodes, {len(edges)} edges.")

    # Resolution sweep — maximise modularity
    resolutions = np.linspace(resolution_min, resolution_max, resolution_steps)
    best_resolution = float(resolutions[0])
    best_modularity = -np.inf
    best_partition = None

    log.info(f"  Sweeping resolution {resolution_min}…{resolution_max} ({resolution_steps} steps)...")
    for res in resolutions:
        partition = leidenalg.find_partition(
            ig_g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(res),
            seed=42,
        )
        mod = partition.modularity
        if mod > best_modularity:
            best_modularity = mod
            best_resolution = float(res)
            best_partition = partition

    n_communities = len(best_partition)
    log.info(
        f"  Best resolution={best_resolution:.3f} → "
        f"{n_communities} communities (modularity={best_modularity:.4f})."
    )

    # Build group labels
    membership = best_partition.membership  # list of community IDs per node
    group_labels = [f"group_{m + 1}" for m in membership]

    feature_ids = numeric_data.index.tolist()
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

    if annotation_df is not None and not annotation_df.empty:
        ann = annotation_df.copy()
        if 'feature_id' in ann.columns:
            ann = ann.set_index('feature_id')
        ann_cols = [c for c in ann.columns if c not in node_table.columns]
        node_table = node_table.join(ann[ann_cols], how='left')

    n_groups = node_table['group'].nunique()
    log.info(f"  Leiden k-NN grouping complete: {n_groups} groups across {len(node_table)} features.")

    write_integration_file(data=node_table, output_dir=output_dir, filename=output_filenames["node_table"], indexing=True, index_label="node_id")
    empty_edge_table = pd.DataFrame(columns=['source', 'target', 'weight'])
    write_integration_file(data=empty_edge_table, output_dir=output_dir, filename=output_filenames["edge_table"], indexing=True, index_label="edge_index")

    return node_table, empty_edge_table


def group_features_wgcna(
    data: pd.DataFrame,
    power_range: List[int] = None,
    power: int | None = None,
    r2_threshold: float = 0.85,
    signed: bool = True,
    min_module_size: int = 30,
    merge_cut_height: float = 0.25,
    deep_split: int = 2,
    output_dir: str = None,
    output_filenames: Dict[str, str] = None,
    datasets: List = None,
    annotation_df: Optional[pd.DataFrame] = None,
    integrated_data: pd.DataFrame = None,
    integrated_metadata: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Weighted Gene Co-expression Network Analysis (WGCNA) implemented in pure
    Python using numpy/scipy/sklearn — no R dependency required.

    **Algorithm**

    1. **Soft-threshold (β) selection** — sweeps ``power_range`` values and
       fits a linear model to ``log(k)`` vs ``log(p(k))`` (scale-free topology
       criterion).  The smallest β where R² ≥ ``r2_threshold`` is chosen.
       If ``power`` is set explicitly, the sweep is skipped.

    2. **Adjacency matrix** — ``A_ij = |cor(i,j)|^β`` (unsigned) or
       ``((1 + cor(i,j)) / 2)^β`` (signed, default).

    3. **Topological Overlap Matrix (TOM)** — reduces noise by considering
       shared neighbours:
       ``TOM_ij = (Σ_k A_ik·A_kj + A_ij) / (min(k_i, k_j) + 1 − A_ij)``

    4. **Hierarchical clustering** on ``1 − TOM`` dissimilarity (average linkage).

    5. **Dynamic tree cut** — cuts the dendrogram using a simplified
       ``deep_split`` heuristic to detect modules of at least
       ``min_module_size`` features.

    6. **Module merging** — modules whose eigengenes (PC1) are correlated
       above ``1 − merge_cut_height`` are merged.

    7. **Diagnostic outputs** written to ``<output_dir>/wgcna_results/``:

       * ``soft_threshold_plot.pdf`` — R² and mean connectivity vs β
       * ``dendrogram_modules.pdf`` — cluster dendrogram + module colour bar
       * ``module_eigengenes.csv`` — ME × sample matrix
       * ``module_membership_kme.csv`` — feature × module kME table
       * ``intramodular_connectivity_kin.csv`` — per-feature kIN
       * ``module_trait_correlation.pdf`` — ME × trait heatmap
         (requires ``integrated_metadata`` with numeric columns)
       * ``gs_vs_mm_plots/`` — GS vs MM scatter for each module × trait pair

    Parameters
    ----------
    data : pd.DataFrame
        Feature × samples (or feature × contrasts) matrix.
    power_range : list of int, optional
        β values to sweep for soft-threshold selection.
        Default: ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]``.
    power : int, optional
        If set, skip the soft-threshold sweep and use this β directly.
    r2_threshold : float, default 0.85
        Minimum scale-free topology R² to accept a β value.
    signed : bool, default True
        Use signed adjacency (preserves direction of correlation).
    min_module_size : int, default 30
        Minimum number of features per module.
    merge_cut_height : float, default 0.25
        Modules with ME correlation > ``1 − merge_cut_height`` are merged.
    output_dir : str
        Base output directory.
    output_filenames : Dict[str, str]
        Dict with keys ``"node_table"`` and ``"edge_table"``.
    datasets : List
        Dataset objects with ``.dataset_name`` attribute.
    annotation_df : pd.DataFrame, optional
        Feature annotation table.
    integrated_data : pd.DataFrame
        Accepted for API consistency; not used.
    integrated_metadata : pd.DataFrame, optional
        Sample metadata with numeric trait columns for module-trait correlation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``(node_table, edge_table)`` where ``node_table`` is indexed by
        feature ID with a ``'group'`` column (module assignment), and
        ``edge_table`` is empty.
    """
    from scipy.cluster.hierarchy import linkage as _linkage, fcluster as _fcluster, dendrogram as _dendrogram
    from scipy.spatial.distance import squareform as _squareform
    from sklearn.decomposition import PCA as _PCA

    if power_range is None:
        power_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

    wgcna_dir = os.path.join(output_dir, "wgcna_results")
    os.makedirs(wgcna_dir, exist_ok=True)
    gs_mm_dir = os.path.join(wgcna_dir, "gs_vs_mm_plots")
    os.makedirs(gs_mm_dir, exist_ok=True)

    # ── Prepare data matrix ───────────────────────────────────────────────────
    numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
    numeric_data = numeric_data.loc[~(numeric_data == 0).all(axis=1)]
    X = numeric_data.values.astype(np.float64)   # features × samples
    n_features, n_samples = X.shape
    feature_ids = numeric_data.index.tolist()

    log.info(f"WGCNA: {n_features} features × {n_samples} samples/contrasts")

    # ── Step 1: Pearson correlation matrix ────────────────────────────────────
    log.info("WGCNA: Computing Pearson correlation matrix...")
    # Standardise rows for correlation
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xc / norms
    cor_mat = (Xn @ Xn.T) / (n_samples - 1)
    np.clip(cor_mat, -1.0, 1.0, out=cor_mat)
    np.fill_diagonal(cor_mat, 1.0)

    # ── Step 2: Soft-threshold selection ─────────────────────────────────────
    if power is None:
        log.info(f"WGCNA: Sweeping soft-threshold β ∈ {power_range}...")
        r2_vals, mean_k_vals = [], []
        for beta in power_range:
            if signed:
                adj = ((1.0 + cor_mat) / 2.0) ** beta
            else:
                adj = np.abs(cor_mat) ** beta
            np.fill_diagonal(adj, 0.0)
            k = adj.sum(axis=1)

            # Quantile-based binning instead of equal-width histogram bins,
            # so skewed degree distributions still populate multiple bins.
            nbins = min(20, n_features // 5 + 1)          # same cap you used before
            quantile_edges = np.quantile(k, np.linspace(0, 1, nbins + 1))
            quantile_edges = np.unique(quantile_edges)     # drop duplicate edges (ties in k)
            if len(quantile_edges) < 4:                    # not enough distinct values to bin meaningfully
                r2_vals.append(0.0)
                mean_k_vals.append(float(k.mean()))
                continue

            k_hist, k_edges = np.histogram(k, bins=quantile_edges)
            k_centers = (k_edges[:-1] + k_edges[1:]) / 2
            mask = (k_hist > 0) & (k_centers > 0)
            if mask.sum() < 3:
                r2_vals.append(0.0)
                mean_k_vals.append(float(k.mean()))
                continue

            log_k = np.log10(k_centers[mask])
            log_p = np.log10(k_hist[mask] / k_hist[mask].sum())
            A = np.vstack([log_k, np.ones(len(log_k))]).T
            slope, intercept = np.linalg.lstsq(A, log_p, rcond=None)[0]
            ss_res = np.sum((log_p - (slope * log_k + intercept)) ** 2)
            ss_tot = np.sum((log_p - log_p.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_vals.append(r2 if slope < 0 else 0.0)       # enforce negative-slope criterion
            mean_k_vals.append(float(k.mean()))

        # Choose smallest β with R² ≥ threshold
        chosen_power = None
        for i, beta in enumerate(power_range):
            if r2_vals[i] >= r2_threshold:
                chosen_power = beta
                break
        if chosen_power is None:
            chosen_power = power_range[int(np.argmax(r2_vals))]
            log.warning(
                f"WGCNA: No β reached R²≥{r2_threshold}. "
                f"Using β={chosen_power} (best R²={max(r2_vals):.3f}). "
                "Consider lowering r2_threshold or checking data quality."
            )
        else:
            log.info(f"WGCNA: Selected β={chosen_power} (R²={r2_vals[power_range.index(chosen_power)]:.3f})")

        # Plot soft-threshold diagnostics
        fig_st, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(power_range, r2_vals, "o-", color="#440154")
        ax1.axhline(r2_threshold, color="red", linestyle="--", label=f"R²={r2_threshold}")
        ax1.axvline(chosen_power, color="orange", linestyle=":", label=f"β={chosen_power}")
        ax1.set_xlabel("Soft-threshold power (β)")
        ax1.set_ylabel("Scale-free topology R²")
        ax1.set_title("Soft-threshold selection")
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax2.plot(power_range, mean_k_vals, "s-", color="#21908c")
        ax2.axvline(chosen_power, color="orange", linestyle=":", label=f"β={chosen_power}")
        ax2.set_xlabel("Soft-threshold power (β)")
        ax2.set_ylabel("Mean connectivity")
        ax2.set_title("Mean connectivity vs β")
        ax2.legend()
        fig_st.tight_layout()
        fig_st.savefig(os.path.join(wgcna_dir, "soft_threshold_plot.pdf"), bbox_inches="tight")
        plt.close(fig_st)
        log.info(f"WGCNA: Saved soft_threshold_plot.pdf")
    else:
        chosen_power = power
        log.info(f"WGCNA: Using user-specified β={chosen_power}")

    # ── Step 3: Adjacency matrix ──────────────────────────────────────────────
    log.info(f"WGCNA: Building {'signed' if signed else 'unsigned'} adjacency (β={chosen_power})...")
    if signed:
        adj = ((1.0 + cor_mat) / 2.0) ** chosen_power
    else:
        adj = np.abs(cor_mat) ** chosen_power
    np.fill_diagonal(adj, 0.0)
    _mean_k = adj.sum(axis=1).mean()
    log.info(f"WGCNA: β={chosen_power} → mean connectivity={_mean_k:.2f} "
             f"(max possible={n_features - 1}); "
             f"{'network looks near-complete, consider raising β' if _mean_k > 0.3 * (n_features - 1) else 'OK'}")

    # ── Step 4: Topological Overlap Matrix (TOM) ──────────────────────────────
    log.info("WGCNA: Computing Topological Overlap Matrix (TOM)...")
    k = adj.sum(axis=1)                          # connectivity per feature
    # Numerator: shared neighbour overlap + direct connection
    numerator = adj @ adj + adj                  # (n × n)
    # Denominator: min(k_i, k_j) + 1 − A_ij
    ki = k[:, np.newaxis]
    kj = k[np.newaxis, :]
    denominator = np.minimum(ki, kj) + 1.0 - adj
    denominator = np.where(denominator == 0, 1e-10, denominator)
    tom = numerator / denominator
    np.clip(tom, 0.0, 1.0, out=tom)
    np.fill_diagonal(tom, 1.0)
    dissimilarity = 1.0 - tom

    # ── Step 5: Hierarchical clustering ───────────────────────────────────────
    log.info("WGCNA: Hierarchical clustering on TOM dissimilarity...")
    dist_condensed = _squareform(dissimilarity, checks=False)
    Z = _linkage(dist_condensed, method="average")

    # ── Step 6: Dynamic tree cut (simplified) ─────────────────────────────────
    log.info(f"WGCNA: Cutting dendrogram (min_module_size={min_module_size}, deep_split={deep_split})...")
    max_height = Z[:, 2].max()
    _max_height = Z[:, 2].max()
    log.info(f"WGCNA: dendrogram height range = [0, {_max_height:.4f}]")
    if _max_height < 0.1:
        raise ValueError("WGCNA: dendrogram is nearly flat (max height < 0.1) — "
                        "TOM dissimilarities show little structure; check adjacency/β.")
    # Scan from coarse (near root) to fine (near leaves); skip the exact endpoints
    candidate_heights = np.linspace(max_height, 0, 50)[1:-1]

    cut_results = []
    for h in candidate_heights:
        labels = _fcluster(Z, t=h, criterion="distance")
        counts = pd.Series(labels).value_counts()
        n_valid = int((counts >= min_module_size).sum())
        cut_results.append((h, n_valid, labels))

    valid_cuts = [c for c in cut_results if c[1] > 0]

    if not valid_cuts:
        log.warning(
            "WGCNA: No cut height produced modules ≥ min_module_size. "
            "All features will be unassigned (grey). "
            "Consider lowering min_module_size or checking TOM/adjacency."
        )
        raw_labels = np.zeros(n_features, dtype=int)
    else:
        # deep_split (0-4) selects position along the coarse→fine scan:
        # 0 → earliest valid cut (largest height, fewer/larger modules)
        # 4 → latest valid cut (smallest height, more/smaller modules)
        idx = int(round((deep_split / 4.0) * (len(valid_cuts) - 1)))
        idx = min(max(idx, 0), len(valid_cuts) - 1)
        base_height, n_modules_at_height, raw_labels = valid_cuts[idx]
        log.info(
            f"WGCNA: Selected cut height={base_height:.4f} "
            f"({n_modules_at_height} candidate modules ≥ min_module_size, "
            f"deep_split={deep_split} → index {idx}/{len(valid_cuts) - 1})"
        )

    # Merge small modules into "grey" (unassigned)  ── (unchanged, still needed
    # because the chosen height's labels may include some clusters just under
    # min_module_size that weren't part of the n_valid count logic above)
    label_counts = pd.Series(raw_labels).value_counts()
    small_labels = label_counts[label_counts < min_module_size].index
    raw_labels = np.where(np.isin(raw_labels, small_labels), 0, raw_labels)

    # Re-number modules 1..N (0 = grey/unassigned)
    unique_nonzero = sorted(set(raw_labels) - {0})
    remap = {old: new for new, old in enumerate(unique_nonzero, start=1)}
    remap[0] = 0
    module_labels = np.array([remap[l] for l in raw_labels])
    n_modules = len(unique_nonzero)
    log.info(f"WGCNA: {n_modules} modules detected before merging ({(module_labels == 0).sum()} features unassigned)")

    # ── Step 7: Module eigengenes (ME = PC1 per module) ───────────────────────
    log.info("WGCNA: Computing module eigengenes...")
    sample_names = numeric_data.columns.tolist()
    me_dict: Dict[int, np.ndarray] = {}
    for mod_id in range(1, n_modules + 1):
        mask = module_labels == mod_id
        if mask.sum() < 2:
            me_dict[mod_id] = np.zeros(n_samples)
            continue
        mod_X = X[mask]
        pca = _PCA(n_components=1)
        me = pca.fit_transform(mod_X.T).flatten()  # samples × 1 → (n_samples,)
        # Ensure ME is positively correlated with average module expression
        avg = mod_X.mean(axis=0)
        if np.corrcoef(me, avg)[0, 1] < 0:
            me = -me
        me_dict[mod_id] = me

    me_df = pd.DataFrame(
        {f"ME{mod_id}": me_dict[mod_id] for mod_id in range(1, n_modules + 1)},
        index=sample_names,
    )

    # ── Step 8: Module merging ────────────────────────────────────────────────
    if n_modules > 1 and merge_cut_height < 1.0:
        log.info(f"WGCNA: Merging modules with ME correlation > {1 - merge_cut_height:.2f}...")
        me_cor = np.corrcoef(me_df.values.T)
        me_dist = _squareform(1.0 - me_cor, checks=False)
        me_Z = _linkage(me_dist, method="average")
        me_cut = _fcluster(me_Z, t=merge_cut_height, criterion="distance")

        # Build merge map: old module → new module
        merge_map: Dict[int, int] = {}
        for new_id, old_ids in enumerate(
            [np.where(me_cut == c)[0] + 1 for c in sorted(set(me_cut))], start=1
        ):
            for old_id in old_ids:
                merge_map[old_id] = new_id

        module_labels = np.array([
            merge_map.get(l, 0) if l != 0 else 0
            for l in module_labels
        ])
        n_modules_merged = len(set(module_labels) - {0})
        log.info(f"WGCNA: {n_modules_merged} modules after merging")

        # Recompute MEs after merging
        me_dict = {}
        for mod_id in sorted(set(module_labels) - {0}):
            mask = module_labels == mod_id
            if mask.sum() < 2:
                me_dict[mod_id] = np.zeros(n_samples)
                continue
            mod_X = X[mask]
            pca = _PCA(n_components=1)
            me = pca.fit_transform(mod_X.T).flatten()
            avg = mod_X.mean(axis=0)
            if np.corrcoef(me, avg)[0, 1] < 0:
                me = -me
            me_dict[mod_id] = me

        me_df = pd.DataFrame(
            {f"ME{mod_id}": me_dict[mod_id] for mod_id in sorted(me_dict)},
            index=sample_names,
        )

    final_n_modules = len(set(module_labels) - {0})
    log.info(f"WGCNA: Final module count: {final_n_modules}")

    # ── Step 9: Module Membership (kME) ───────────────────────────────────────
    log.info("WGCNA: Computing module membership (kME)...")
    kme_dict: Dict[str, np.ndarray] = {}
    for col in me_df.columns:
        me_vec = me_df[col].values
        kme_col = np.array([
            float(np.corrcoef(X[i], me_vec)[0, 1]) if n_samples > 2 else 0.0
            for i in range(n_features)
        ])
        kme_dict[col] = kme_col

    kme_df = pd.DataFrame(kme_dict, index=feature_ids)
    kme_df.index.name = "feature_id"
    kme_df.to_csv(os.path.join(wgcna_dir, "module_membership_kme.csv"))
    log.info("WGCNA: Saved module_membership_kme.csv")

    # ── Step 10: Intramodular connectivity (kIN) ──────────────────────────────
    log.info("WGCNA: Computing intramodular connectivity (kIN)...")
    kin_values = np.zeros(n_features)
    for mod_id in sorted(set(module_labels) - {0}):
        mask = np.where(module_labels == mod_id)[0]
        if len(mask) < 2:
            continue
        sub_adj = adj[np.ix_(mask, mask)]
        kin_values[mask] = sub_adj.sum(axis=1)

    kin_df = pd.DataFrame({
        "feature_id": feature_ids,
        "module": [f"module_{l}" if l != 0 else "grey" for l in module_labels],
        "kIN": kin_values,
    }).set_index("feature_id")
    kin_df.to_csv(os.path.join(wgcna_dir, "intramodular_connectivity_kin.csv"))
    log.info("WGCNA: Saved intramodular_connectivity_kin.csv")

    # ── Step 11: Module eigengenes table ──────────────────────────────────────
    me_df.to_csv(os.path.join(wgcna_dir, "module_eigengenes.csv"))
    log.info("WGCNA: Saved module_eigengenes.csv")

    # ── Step 12: Dendrogram + module colour bar ───────────────────────────────
    log.info("WGCNA: Plotting cluster dendrogram with module colour bar...")
    viridis_colors = plt.cm.get_cmap("tab20", max(final_n_modules, 1))
    module_colors = [
        viridis_colors(module_labels[i] - 1) if module_labels[i] != 0 else (0.7, 0.7, 0.7, 1.0)
        for i in range(n_features)
    ]

    fig_dend, (ax_dend, ax_bar) = plt.subplots(
        2, 1, figsize=(max(12, n_features // 50), 6),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    dend = _dendrogram(Z, ax=ax_dend, no_labels=True, color_threshold=0,
                       above_threshold_color="gray", link_color_func=lambda k: "gray")
    ax_dend.set_title("WGCNA Cluster Dendrogram (1 − TOM dissimilarity)")
    ax_dend.set_ylabel("Height")
    ax_dend.set_xticks([])

    # Reorder module colours by dendrogram leaf order
    leaf_order = dend["leaves"]
    bar_colors = [module_colors[i] for i in leaf_order]
    for j, color in enumerate(bar_colors):
        ax_bar.add_patch(plt.Rectangle((j, 0), 1, 1, color=color))
    ax_bar.set_xlim(0, n_features)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis("off")
    ax_bar.set_title("Module colours", loc="left", fontsize=8)

    fig_dend.tight_layout()
    fig_dend.savefig(os.path.join(wgcna_dir, "dendrogram_modules.pdf"), bbox_inches="tight")
    plt.close(fig_dend)
    log.info("WGCNA: Saved dendrogram_modules.pdf")

    # ── Step 13: Module-trait correlation heatmap ─────────────────────────────
    if integrated_metadata is not None and not integrated_metadata.empty and not me_df.empty:
        log.info("WGCNA: Computing module-trait correlations...")
        # Align metadata to samples
        meta_aligned = integrated_metadata.reindex(me_df.index)
        numeric_traits = meta_aligned.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

        if not numeric_traits.empty:
            mt_cor = pd.DataFrame(index=me_df.columns, columns=numeric_traits.columns, dtype=float)
            mt_pval = pd.DataFrame(index=me_df.columns, columns=numeric_traits.columns, dtype=float)
            from scipy.stats import pearsonr as _pearsonr
            for me_col in me_df.columns:
                for trait_col in numeric_traits.columns:
                    me_vec = me_df[me_col].values
                    trait_vec = numeric_traits[trait_col].values
                    valid = ~(np.isnan(me_vec) | np.isnan(trait_vec))
                    if valid.sum() > 2:
                        r, p = _pearsonr(me_vec[valid], trait_vec[valid])
                        mt_cor.loc[me_col, trait_col] = r
                        mt_pval.loc[me_col, trait_col] = p
                    else:
                        mt_cor.loc[me_col, trait_col] = np.nan
                        mt_pval.loc[me_col, trait_col] = np.nan

            mt_cor = mt_cor.astype(float)
            mt_pval = mt_pval.astype(float)
            mt_cor.to_csv(os.path.join(wgcna_dir, "module_trait_correlation.csv"))

            # Heatmap with correlation values and p-value stars
            fig_mt, ax_mt = plt.subplots(
                figsize=(max(6, len(numeric_traits.columns) * 1.2),
                         max(4, len(me_df.columns) * 0.5))
            )
            im = ax_mt.imshow(mt_cor.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax_mt, label="Pearson r")
            ax_mt.set_xticks(range(len(numeric_traits.columns)))
            ax_mt.set_xticklabels(numeric_traits.columns, rotation=45, ha="right", fontsize=8)
            ax_mt.set_yticks(range(len(me_df.columns)))
            ax_mt.set_yticklabels(me_df.columns, fontsize=8)
            ax_mt.set_title("Module-Trait Correlation")

            for i, me_col in enumerate(me_df.columns):
                for j, trait_col in enumerate(numeric_traits.columns):
                    r_val = mt_cor.loc[me_col, trait_col]
                    p_val = mt_pval.loc[me_col, trait_col]
                    if pd.isna(r_val):
                        continue
                    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    ax_mt.text(j, i, f"{r_val:.2f}{stars}", ha="center", va="center",
                               fontsize=6, color="white" if abs(r_val) > 0.5 else "black")

            fig_mt.tight_layout()
            fig_mt.savefig(os.path.join(wgcna_dir, "module_trait_correlation.pdf"), bbox_inches="tight")
            plt.close(fig_mt)
            log.info("WGCNA: Saved module_trait_correlation.pdf and .csv")

            # ── Step 14: GS vs MM scatter plots ──────────────────────────────
            log.info("WGCNA: Generating GS vs MM scatter plots...")
            for me_col in me_df.columns:
                mod_id_str = me_col.replace("ME", "")
                try:
                    mod_id = int(mod_id_str)
                except ValueError:
                    continue
                mod_mask = module_labels == mod_id
                if mod_mask.sum() < 5:
                    continue
                mm_vals = kme_df[me_col].values[mod_mask]
                mod_feature_ids = [feature_ids[i] for i in range(n_features) if mod_mask[i]]

                for trait_col in numeric_traits.columns:
                    trait_vec = numeric_traits[trait_col].values
                    gs_vals = np.array([
                        float(np.corrcoef(X[i], trait_vec)[0, 1])
                        if (~np.isnan(trait_vec)).sum() > 2 else 0.0
                        for i in range(n_features) if mod_mask[i]
                    ])
                    valid = ~(np.isnan(mm_vals) | np.isnan(gs_vals))
                    if valid.sum() < 5:
                        continue

                    fig_gs, ax_gs = plt.subplots(figsize=(5, 5))
                    ax_gs.scatter(mm_vals[valid], gs_vals[valid], alpha=0.6, s=20, color="#440154")
                    if valid.sum() > 2:
                        from scipy.stats import pearsonr as _pr
                        r_gs, p_gs = _pr(mm_vals[valid], gs_vals[valid])
                        ax_gs.set_title(
                            f"{me_col} vs {trait_col}\nr={r_gs:.3f}, p={p_gs:.2e}",
                            fontsize=9
                        )
                    ax_gs.set_xlabel(f"Module Membership (kME, {me_col})", fontsize=8)
                    ax_gs.set_ylabel(f"Gene Significance (GS, {trait_col})", fontsize=8)
                    ax_gs.axhline(0, color="gray", linewidth=0.5)
                    ax_gs.axvline(0, color="gray", linewidth=0.5)
                    safe_trait = trait_col.replace("/", "_").replace(" ", "_")
                    fig_gs.tight_layout()
                    fig_gs.savefig(
                        os.path.join(gs_mm_dir, f"{me_col}_{safe_trait}_gs_vs_mm.pdf"),
                        bbox_inches="tight"
                    )
                    plt.close(fig_gs)
            log.info(f"WGCNA: Saved GS vs MM plots to {gs_mm_dir}/")

    # ── Step 15: Build node table ─────────────────────────────────────────────
    group_labels_str = [
        f"module_{l}" if l != 0 else "grey"
        for l in module_labels
    ]

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
        "group": group_labels_str,
        "datatype_color": [_get_color(f) for f in feature_ids],
        "datatype_shape": [_get_shape(f) for f in feature_ids],
        "kIN": kin_values,
        "wgcna_module": group_labels_str,
    }, index=feature_ids)
    node_table.index.name = "node_id"

    # Merge kME columns
    node_table = node_table.join(kme_df, how="left")

    # Merge annotation columns if provided
    if annotation_df is not None and not annotation_df.empty:
        ann = annotation_df.copy()
        if "feature_id" in ann.columns:
            ann = ann.set_index("feature_id")
        ann_cols = [c for c in ann.columns if c not in node_table.columns]
        node_table = node_table.join(ann[ann_cols], how="left")

    n_groups = node_table["group"].nunique()
    log.info(f"WGCNA: {n_groups} groups (including grey) across {len(node_table)} features.")

    write_integration_file(
        data=node_table,
        output_dir=output_dir,
        filename=output_filenames["node_table"],
        indexing=True,
        index_label="node_id",
    )
    empty_edge_table = pd.DataFrame(columns=["source", "target", "weight"])
    write_integration_file(
        data=empty_edge_table,
        output_dir=output_dir,
        filename=output_filenames["edge_table"],
        indexing=True,
        index_label="edge_index",
    )

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
        Feature matrix (features x samples or features x contrasts).
    method : str
        One of ``"network_modules"``, ``"hierarchical_clustering"``,
        ``"hdbscan"``, ``"nmf"``, ``"leiden_knn"``, ``"wgcna"``.
    method_params : Dict
        Sub-block from config for the selected method.
    output_dir : str
        Directory to write output files.
    output_filenames : Dict[str, str]
        Dict with keys ``"node_table"`` and ``"edge_table"``.
    datasets : List
        List of dataset objects with a ``.dataset_name`` attribute.
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
            show_network_plot=method_params.get('show_network_plot', False),
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

    elif method == "nmf":
        return group_features_nmf(
            data=data,
            k_min=method_params.get('k_min', 2),
            k_max=method_params.get('k_max', 50),
            n_runs=method_params.get('n_runs', 3),
            max_iter=method_params.get('max_iter', 500),
            output_dir=output_dir,
            output_filenames=output_filenames,
            datasets=datasets,
            annotation_df=annotation_df,
            integrated_data=integrated_data,
            integrated_metadata=integrated_metadata,
        )

    elif method == "leiden_knn":
        return group_features_leiden_knn(
            data=data,
            n_neighbors=method_params.get('n_neighbors', 15),
            resolution_min=method_params.get('resolution_min', 0.1),
            resolution_max=method_params.get('resolution_max', 2.0),
            resolution_steps=method_params.get('resolution_steps', 20),
            output_dir=output_dir,
            output_filenames=output_filenames,
            datasets=datasets,
            annotation_df=annotation_df,
            integrated_data=integrated_data,
            integrated_metadata=integrated_metadata,
        )

    elif method == "wgcna":
        return group_features_wgcna(
            data=data,
            power_range=method_params.get('power_range', None),
            power=method_params.get('power', None),
            r2_threshold=method_params.get('r2_threshold', 0.85),
            signed=method_params.get('signed', True),
            min_module_size=method_params.get('min_module_size', 30),
            merge_cut_height=method_params.get('merge_cut_height', 0.25),
            deep_split=method_params.get('deep_split', 2),
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
            "Choose from: network_modules, hierarchical_clustering, hdbscan, nmf, leiden_knn, wgcna"
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

        # Check for transcriptomics / proteomics annotation and name
        elif str(n).startswith(('tx_', 'px_')):
            ds_prefix = str(n)[:2]
            tx_go_acc = data.get(f"{ds_prefix}_go_acc", "Unassigned")
            tx_display_name = data.get(f"{ds_prefix}_display_name", "Unassigned")
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
    sub_cols = cm.viridis(np.linspace(0, 1, n_mod))
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

# ====================================
# Dataset acquisition functions
# ====================================     

# def find_mx_parent_folder(
#     pid: str,
#     pi_name: str,
#     mx_dir: str,
#     polarity: str,
#     datatype: str,
#     chromatography: str,
#     filtered_mx: bool = True,
#     overwrite: bool = False,
# ) -> str:
#     """
#     Find the parent folder for metabolomics (MX) data on Google Drive using rclone.

#     Args:
#         pid (str): Proposal ID.
#         pi_name (str): PI name.
#         mx_dir (str): Local MX data directory.
#         polarity (str): Polarity ('positive', 'negative', 'multipolarity').
#         datatype (str): Data type ('peak-height', 'peak-area', etc.).
#         chromatography (str): Chromatography type.
#         filtered_mx (bool): Whether to use filtered data.
#         overwrite (bool): Overwrite existing results.

#     Returns:
#         str: Path to the final results folder, or None if not found.
#     """
    
#     if datatype == "peak-area":
#         datatype = "quant"
#     if filtered_mx and datatype == "quant":
#         log.info("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
#         return None
#     if polarity == "multipolarity":
#         mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
#     elif polarity in ["positive", "negative"]:
#         mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
#     else:
#         log.info(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
#         return None
#     if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
#         log.info("MX folder already downloaded and linked.")
#         return None
#     elif glob.glob(os.path.expanduser(mx_data_pattern)) and overwrite:
#         log.info("MX folder already downloaded and linked. Overwriting as per user request...")
#     elif not glob.glob(os.path.expanduser(mx_data_pattern)):
#         log.info("MX folder not found locally. Proceeding to find and link MX data from Google Drive...")

#     # Find project folder
#     cmd = f"rclone lsd JGI_Metabolomics_Projects: | grep -E '{pid}|{pi_name}'"
#     log.info("Finding MX parent folders...")
#     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
#     if result.stdout:
#         data = [line.split()[:5] for line in result.stdout.strip().split('\n')]
#         mx_parent = pd.DataFrame(data, columns=["dir", "date", "time", "size", "folder"])
#         mx_parent = mx_parent[["date", "time", "folder"]]
#         mx_parent["ix"] = range(1, len(mx_parent) + 1)
#         mx_parent = mx_parent[["ix", "date", "time", "folder"]]
#         mx_final_folders = []
#         # For each possible project folder (some will not be the right "final" folder)
#         log.info("Finding MX final folders...")
#         for project_folder in mx_parent["folder"].values:
#             cmd = f"rclone lsd --max-depth 2 JGI_Metabolomics_Projects:{project_folder}"
#             try:
#                 result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#             except:
#                 continue
#             if result.stdout or result.stderr:
#                 output = result.stdout if result.stdout else result.stderr
#                 data = [line.split()[:5] for line in output.strip().split('\n')]
#                 mx_final = pd.DataFrame(data, columns=["dir", "date", "time", "size", "folder"])
#                 mx_final = mx_final[["date", "time", "folder"]]
#                 mx_final["ix"] = range(1, len(mx_final) + 1)
#                 mx_final = mx_final[["ix", "date", "time", "folder"]]
#                 mx_final['parent_folder'] = project_folder
#                 mx_final_folders.append(mx_final)
#             else:
#                 return None

#         mx_final_combined = pd.concat(mx_final_folders, ignore_index=True)
#         untargeted_mx_final = mx_final_combined[
#             mx_final_combined["folder"].str.contains("Untargeted", case=False) & 
#             mx_final_combined["folder"].str.contains("final", case=False) & 
#             ~mx_final_combined["folder"].str.contains("pilot", case=False)
#         ]
#         if untargeted_mx_final.shape[0] > 1:
#             log.info("Warning! Multiple untargeted MX final folders found:")
#             log.info(untargeted_mx_final)
#             return None
#         elif untargeted_mx_final.shape[0] == 0:
#             log.info("Warning! No untargeted MX final folders found.")
#             return None
#         else:
#             final_results_folder = f"{untargeted_mx_final['parent_folder'].values[0]}/{untargeted_mx_final['folder'].values[0]}"

#         script_dir = f"{mx_dir}/scripts"
#         os.makedirs(script_dir, exist_ok=True)
#         script_name = f"{script_dir}/find_mx_files.sh"
#         with open(script_name, "w") as script_file:
#             script_file.write(f"cd {script_dir}/\n")
#             script_file.write(f"rclone lsd --max-depth 2 JGI_Metabolomics_Projects:{untargeted_mx_final['parent_folder'].values[0]}")
        
#         log.info("Using the following metabolomics final results folder for further analysis:")
#         log.info(untargeted_mx_final)
#         return final_results_folder
#     else:
#         log.info(f"Warning! No folders could be found with rclone lsd command: {cmd}")
#         return None

# def gather_mx_files(
#     mx_untargeted_remote: str,
#     mx_dir: str,
#     polarity: str,
#     datatype: str,
#     chromatography: str,
#     filtered_mx: bool = True,
#     extract: bool = False,
#     overwrite: bool = False
# ) -> tuple:
#     """
#     Link MX files from Google Drive to the local directory using rclone, and optionally extract them.

#     Args:
#         mx_untargeted_remote (str): Remote MX folder path.
#         mx_dir (str): Local MX data directory.
#         polarity (str): Polarity.
#         datatype (str): Data type.
#         chromatography (str): Chromatography type.
#         filtered_mx (bool): Use filtered data.
#         extract (bool): Extract archives after linking.
#         overwrite (bool): Overwrite existing results.

#     Returns:
#         tuple: DataFrames of archives and extractions, or None.
#     """
    
#     if datatype == "peak-area":
#         datatype = "quant"
#     if filtered_mx and datatype == "quant":
#         log.info("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
#         return None
#     if polarity == "multipolarity":
#         mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
#     elif polarity in ["positive", "negative"]:
#         mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
#     else:
#         log.info(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
#         return None
#     if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
#         log.info("MX data already linked.")
#         return "Archive already linked", "Archive already extracted"
#     else:
#         raise ValueError("You are not currently authorized to download metabolomics data from source. Please contact your JGI project manager for access.")
    
#     script_dir = f"{mx_dir}/scripts"
#     os.makedirs(script_dir, exist_ok=True)
#     script_name = f"{script_dir}/gather_mx_files.sh"
#     if chromatography == "C18":
#         chromatography = "C18_" # This (hopefully) removes C18-Lipid

#     # Create the script to link MX files
#     with open(script_name, "w") as script_file:
#         script_file.write(f"cd {script_dir}/\n")
#         script_file.write(f"rclone copy --include '*{chromatography}*.zip' --stats-one-line -v --max-depth 1 JGI_Metabolomics_Projects:{mx_untargeted_remote} {mx_dir};")
    
#     log.info("Linking MX files...")
#     result = subprocess.run(f"chmod +x {script_name} && {script_name}", shell=True, check=True, capture_output=True, text=True)

#     if result.stdout or result.stderr:
#         output = result.stdout if result.stdout else result.stderr
#         data = [line.split() for line in output.strip().split('\n')]
#         archives = pd.DataFrame(data)
#         if extract is True:
#             extractions = extract_mx_archives(mx_dir, chromatography)
#             display(extractions)
#             return archives, extractions
#         else:
#             return archives
#     else:
#         log.info(f"Warning! No files could be found with rclone copy command in {script_name}")
#         return None

# def extract_mx_archives(mx_dir: str, chromatography: str) -> pd.DataFrame:
#     """
#     Extract MX zip archives in the specified directory.

#     Args:
#         mx_dir (str): Directory containing MX archives.
#         chromatography (str): Chromatography type.

#     Returns:
#         pd.DataFrame: DataFrame of extracted files.
#     """
    
#     cmd = f"for archive in {mx_dir}/*{chromatography}*zip; do newdir={mx_dir}/$(basename $archive .zip); rm -rf $newdir; mkdir -p $newdir; unzip -j $archive -d $newdir; done"
#     log.info("Extracting the following archive to be used for MX data input:")
#     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
#     if result.stdout:
#         data = [line.split() for line in result.stdout.strip().split('\n')]
#         df = pd.DataFrame(data)
#         df = df[df.iloc[:, 0].str.contains('Archive', na=False)]
#         df.iloc[:, 1] = df.iloc[:, 1].str.replace(f"{mx_dir}/", "", regex=False)
#         return df
#     else:
#         log.info(f"No archives could be decompressed with unzip command: {cmd}")
#         return None


# def find_tx_files(
#     pid: str,
#     tx_dir: str,
#     tx_index: int,
#     overwrite: bool = False,
# ) -> pd.DataFrame:
#     """
#     Find TX files for a project using JAMO report select.

#     Args:
#         pid (str): Proposal ID.
#         tx_dir (str): TX data directory.
#         tx_index (int): Index of analysis project to use.
#         overwrite (bool): Overwrite existing results.

#     Returns:
#         pd.DataFrame: DataFrame of TX file information.
#     """
    
#     if os.path.exists(f"{tx_dir}/all_tx_portal_files.txt") and overwrite is False:
#         log.info("TX files already found.")
#         return pd.DataFrame()
#     else:
#         raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")
    
#     file_list = f"{tx_dir}/all_tx_portal_files.txt"
#     script_dir = f"{tx_dir}/scripts"
#     os.makedirs(script_dir, exist_ok=True)
#     script_name = f"{script_dir}/find_tx_files.sh"

#     if not os.path.exists(os.path.dirname(file_list)):
#         os.makedirs(os.path.dirname(file_list))
#     if not os.path.exists(os.path.dirname(script_name)):
#         os.makedirs(os.path.dirname(script_name))
    
#     log.info("Creating script to find TX files...")
#     script_content = (
#         f"jamo report select _id,metadata.analysis_project.analysis_project_id,metadata.library_name,metadata.analysis_project.status_name where "
#         f"metadata.proposal_id={pid} file_name=counts.txt "
#         f"| sed 's/\\[//g' | sed 's/\\]//g' | sed 's/u'\\''//g' | sed 's/'\\''//g' | sed 's/ //g' > {file_list}"
#     )

#     log.info("Finding TX files...")
#     subprocess.run(
#         f"echo \"{script_content}\" > {script_name} && chmod +x {script_name} && module load jamo && source {script_name}",
#         shell=True, check=True
#     )
    
#     files = pd.read_csv(file_list, header=None, sep="\t")
#     files.columns = ["fileID.counts", "APID", "libIDs", "status"]
#     files["nLibs"] = files["libIDs"].apply(lambda ll: len(ll.split(",")))
    
#     def get_tpm_file_id(apid):
#         result = subprocess.run(
#             f"module load jamo; jamo report select _id where file_name=tpm_counts.txt,metadata.analysis_project.analysis_project_id={apid}",
#             shell=True, capture_output=True, text=True
#         )
#         return result.stdout.strip()
    
#     files["fileID.tpm"] = files["APID"].apply(get_tpm_file_id)
    
#     def fetch_refs(apid):
#         try:
#             cmd = (
#                 f"x=$(curl https://rqc.jgi.lbl.gov/api/seq_jat_import/apid_to_ref/{apid}); "
#                 f"echo $(echo $x | jq .name)','$(echo $x | jq .Genome[].file_path)','$(echo $x | jq .Transcriptome[].file_path)','$(echo $x | jq .Annotation[].file_path)','$(echo $x | jq .KEGG[].file_path)"
#             )
#             result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#             ref_data = pd.read_csv(io.StringIO(result.stdout), header=None, sep=",")
#             ref_data.columns = ["ref_name", "ref_genome", "ref_transcriptome", "ref_gff", "ref_protein_kegg"]
#             ref_data["APID"] = apid
#             return ref_data
#         except Exception as e:
#             log.info(f"Error fetching refs for {apid}: {e}")
#             return pd.DataFrame({"ref_name": [np.nan], "ref_genome": [np.nan], "ref_transcriptome": [np.nan], "ref_gff": [np.nan], "ref_protein_kegg": [np.nan], "APID": [apid]})
    
#     refs = pd.concat([fetch_refs(apid) for apid in files["APID"].unique()], ignore_index=True)
#     files = files.merge(refs, on="APID", how="left").sort_values(by=["ref_name", "APID"]).reset_index(drop=True)
#     files["ix"] = files.index + 1
#     # Move "ix" column to the beginning
#     cols = ["ix"] + [col for col in files.columns if col != "ix"]
#     files = files[cols]
#     files.reset_index(drop=True)
    
#     if files.shape[0] > 0:
#         log.info(f"Using the value of 'tx_index' ({tx_index}) from the config file to choose the correct 'ix' column (change if incorrect): ")
#         files.to_csv(f"{tx_dir}/all_tx_portal_files.txt", sep="\t", index=False)
#         display(files)
#         return files
#     else:
#         log.info("No files found.")
#         return None

# def gather_tx_files(
#     file_list: pd.DataFrame,
#     tx_index: int = None,
#     tx_dir: str = None,
#     overwrite: bool = False
# ) -> str:
#     """
#     Link TX files to the working directory using JAMO.

#     Args:
#         file_list (pd.DataFrame): DataFrame of TX file information.
#         tx_index (int): Index of analysis project to use.
#         tx_dir (str): TX data directory.
#         overwrite (bool): Overwrite existing results.

#     Returns:
#         str: Analysis project ID (APID).
#     """

#     if glob.glob(f"{tx_dir}/*counts.txt") and os.path.exists(f"{tx_dir}/all_tx_portal_files.txt") and overwrite is False:
#         log.info("TX files already linked.")
#         tx_files = pd.read_csv(f"{tx_dir}/all_tx_portal_files.txt", sep="\t")
#         apid = int(tx_files.iloc[tx_index-1:,2].values)
#         return apid
#     else:
#         raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")


#     if tx_index is None:
#         log.info("There may be multiple APIDS or analyses for a given PI/Proposal ID and you have not specified which one to use!")
#         log.info("Please set 'tx_index' in the project config file by choosing the correct row from the table above.")
#         sys.exit(1)

#     script_dir = f"{tx_dir}/scripts"
#     os.makedirs(script_dir, exist_ok=True)
#     script_name = f"{script_dir}/gather_tx_files.sh"
#     log.info("Linking TX files...")

#     with open(script_name, "w") as script_file:
#         script_file.write(f"cd {script_dir}/\n")
        
#         file_list = file_list[file_list["ix"] == tx_index]
#         apid = file_list["APID"].values[0]
#         for _, row in file_list.iterrows():
#             for file_type in ["counts","tpm"]: # "tpm_counts" may be needed for older projects
#                 file_id = row[f"fileID.{file_type}"]
#                 apid = row['APID']
#                 filename = f"{tx_dir}/{row['ix']}_{apid}.{file_type}.txt"
#                 if file_type == "tpm":
#                     file_type = "tpm_counts"
#                 script_file.write(f"""
#                     module load jamo;
#                     if [ $(jamo info id {file_id} | cut -f3 -d' ') == 'PURGED' ]; then 
#                         echo file purged! fetching {file_id} - rerun later to link; jamo fetch id {file_id} 2>&1 > /dev/null
#                     else 
#                         if [ $(jamo info id {file_id} | cut -f3 -d' ') == 'RESTORE_IN_PROGRESS' ]; then 
#                             echo restore in progress for file id {file_id} - rerun later to link
#                         else
#                             echo "\t{file_type} file for APID {apid} with ID {file_id} linked to {filename}";
#                             jamo link id {file_id} 2>&1 > /dev/null;
#                             mv {file_id}.{file_type}.txt {filename}
#                         fi
#                     fi
#                     """)
#             for file_type in ["genome","transcriptome","gff","protein_kegg"]:
#                 file_path = row[f"ref_{file_type}"]
#                 try:
#                     apid = row['APID']
#                     filename = f"{tx_dir}/{os.path.basename(file_path)}"
#                     if file_path:
#                         script_file.write(f"""
#                             echo "\t{file_type} file for APID {apid} to {file_path}";
#                             ln -sf {file_path} {tx_dir}/
#                         """)
#                 except:
#                     log.info(f"\tError linking {file_type} file for APID {apid}. File does not exist or you may need to wait for files to be restored.")
#                     continue

#     subprocess.run(f"chmod +x {script_name} && {script_name}", shell=True, check=True)
    
#     if apid:
#         with open(f"{tx_dir}/apid.txt", "w") as f:
#             f.write(str(apid))
#         log.info(f"Working with APID: {apid} from tx_index {tx_index}.")
#         return apid
#     else:
#         log.info("Warning: Did not find APID. Check the index you selected from the tx_files object.")
#         return None

def get_mx_data(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    chromatography: str,
    polarity: str,
    datatype: str = "peak-height",
    filtered_mx: bool = True,
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
        log.warning(
            "Place the MX results file(s) under the chromatography directory "
            f"({input_dir}/{chromatography}/) before running the workflow."
        )
        return pd.DataFrame()


def load_raw_metadata(
    input_dir: str,
    output_dir: str,
    output_filename: str = "raw_metadata.csv",
) -> pd.DataFrame:
    """Load a pre-staged dataset metadata table from ``raw_metadata.csv``."""
    metadata_path = Path(input_dir) / "raw_metadata.csv"
    if not metadata_path.is_file():
        log.warning(
            "Required raw metadata file is missing: %s. "
            "Stage it before running the workflow.",
            metadata_path,
        )
        raise FileNotFoundError(f"Required raw metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        log.warning("Raw metadata file is empty: %s", metadata_path)
        raise ValueError(f"Raw metadata file is empty: {metadata_path}")

    write_integration_file(
        data=metadata,
        output_dir=output_dir,
        filename=output_filename,
        indexing=False,
    )
    log.info("Raw metadata loaded from %s", metadata_path)
    return metadata


def load_mx_metadata(
    input_dir: str,
    chromatography: str,
    output_dir: str,
    output_filename: str = "raw_metadata.csv",
) -> pd.DataFrame:
    """Load and merge pre-staged MX metadata files for all polarities.

    The data-gathering workflow may stage one polarity-specific metadata file
    such as ``*_positive_metadata.tab`` and ``*_negative_metadata.tab`` below
    the chromatography directory. A single ``raw_metadata.csv`` is also
    accepted for compatibility. Metadata is normalized using the original MX
    transformations, concatenated, and de-duplicated before being cached.
    """
    chromatography_dir = Path(input_dir) / chromatography
    metadata_files = sorted(chromatography_dir.glob("**/*_metadata.tab"))
    metadata_sep = "\t"
    if not metadata_files:
        metadata_files = sorted((Path(input_dir)).glob("raw_metadata.csv"))
        metadata_sep = None
    if not metadata_files:
        log.warning(
            "No MX metadata files found under %s. Expected polarity-specific "
            "*_metadata.tab files or raw_metadata.csv.",
            chromatography_dir,
        )
        raise FileNotFoundError(
            f"No MX raw_metadata.csv files found under {chromatography_dir}"
        )

    metadata_tables = []
    for metadata_path in metadata_files:
        metadata = pd.read_csv(
            metadata_path,
            sep=metadata_sep if metadata_sep is not None else None,
            engine="python",
        )
        metadata.columns = metadata.columns.str.replace(
            "ATTRIBUTE_sampletype", "full_sample_metadata"
        )
        if "filename" in metadata.columns:
            metadata["filename"] = metadata["filename"].astype(str).str.replace(
                ".mzML", "", regex=False
            )
        if "file" not in metadata.columns and len(metadata.columns) > 0:
            metadata = metadata.rename(columns={metadata.columns[0]: "file"})
        metadata_tables.append(metadata)

    merged_metadata = pd.concat(metadata_tables, axis=0, ignore_index=True)
    merged_metadata = merged_metadata.drop_duplicates().reset_index(drop=True)
    if "ix" in merged_metadata.columns:
        merged_metadata = merged_metadata.drop(columns="ix")
    merged_metadata.insert(0, "ix", merged_metadata.index + 1)

    write_integration_file(
        data=merged_metadata,
        output_dir=output_dir,
        filename=output_filename,
        indexing=False,
    )
    log.info("Merged MX metadata from %s", metadata_files)
    return merged_metadata

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
        identifier_column = tx_data.columns[0]
        
        # Add prefix 'tx_' if not already present
        tx_data[identifier_column] = tx_data[identifier_column].apply(
            lambda x: x if str(x).startswith('tx_') else f'tx_{x}'
        )
        
        log.info(f"TX data loaded from {tx_data_filename} and processing...")
        write_integration_file(data=tx_data, output_dir=output_dir, filename=output_filename, indexing=False)
        #display(tx_data.head())
        return tx_data
    else:
        log.warning(f"TX data file matching pattern {tx_data_pattern} not found.")
        log.warning("Have you run _get_raw_metadata() to link the TX data files?")
        return None
    
def get_px_data(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    datatype: str = "peak-height",
) -> pd.DataFrame:
    """
    Load PX (proteomics) peak-height data from ``<input_dir>/<datatype>.csv``.

    Layout matches the other tables: first column is the feature (protein) ID, remaining
    columns are samples. Feature IDs are prefixed with ``px_`` when not already present.
    """
    px_files = glob.glob(os.path.expanduser(f"{input_dir}/{datatype}.csv"))
    if not px_files:
        log.warning(f"PX data file {input_dir}/{datatype}.csv not found.")
        return None
    px_data = pd.read_csv(px_files[0])
    identifier_column = px_data.columns[0]
    px_data[identifier_column] = px_data[identifier_column].apply(
        lambda x: x if str(x).startswith('px_') else f'px_{x}'
    )
    log.info(f"PX data loaded from {px_files[0]}")
    write_integration_file(data=px_data, output_dir=output_dir, filename=output_filename, indexing=False)
    return px_data


def get_tx_metadata(
    tx_files: pd.DataFrame,
    output_dir: str,
    proposal_ID: str,
    apid: str,
    overwrite: bool = False,
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

def _validate_annotation_gene_ids(annotation_df: pd.DataFrame, raw_data: pd.DataFrame, prefix: str = "tx") -> None:
    """
    Validate that gene IDs in annotation table match those in raw data.
    
    Args:
        annotation_df (pd.DataFrame): Annotation table with transcriptome_id column
        raw_data (pd.DataFrame): Raw data with feature IDs in its first column
        prefix (str): Dataset prefix ("tx" or "px") applied to feature IDs
    """
    identifier_column = raw_data.columns[0]
    tag = f"{prefix}_"

    # If the prefix is used in raw data, ensure annotation gene IDs also have it
    if all(str(gid).startswith(tag) for gid in raw_data[identifier_column]):
        if not all(str(gid).startswith(tag) for gid in annotation_df['transcriptome_id']):
            annotation_df['transcriptome_id'] = annotation_df['transcriptome_id'].apply(lambda x: f'{tag}{x}')

    # Get gene IDs from both datasets
    raw_data_genes = set(raw_data[identifier_column].tolist())
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
        log.info(raw_data[identifier_column].tolist()[:10])
        log.info("Annotation transcriptome_ids:")
        log.info(annotation_df['transcriptome_id'].tolist()[:10])
        raise ValueError("No matching gene IDs found between raw data and annotations, something is wrong.")

def _validate_annotation_identifier_header(
    df: pd.DataFrame,
    file_path: str,
    identifier_column: str,
) -> None:
    if not len(df.columns) or df.columns[0] != identifier_column:
        actual_header = df.columns[0] if len(df.columns) else "<no columns>"
        raise ValueError(
            f"Annotation file {os.path.basename(file_path)} must have "
            f"'{identifier_column}' as its first column; found '{actual_header}'."
        )


def _process_microbe_annotations(
    raw_data_dir: str,
    output_dir: str,
    output_filename: str,
    identifier_column: str,
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
        df = _read_and_select_microbe_annotations(file_path, identifier_column)
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
    merged_df = _add_protein_id_mapping(
        merged_df,
        raw_data_dir,
        "microbe",
        identifier_attribute=identifier_column,
    )
    
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
    output_filename: str,
    identifier_column: str,
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
        df = _read_and_select_algal_annotations(file_path, identifier_column)
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
    merged_df = _add_protein_id_mapping(
        merged_df,
        raw_data_dir,
        "algal",
        identifier_attribute=identifier_column,
    )
    
    # Compress the dataframe to ensure one gene per row with semicolon-separated annotations
    final_df = _compress_annotation_table(merged_df)
    
    log.info(f"Final annotation table: {len(final_df)} genes with {len(final_df.columns)} annotation columns")
    
    # Save the merged annotation table
    write_integration_file(final_df, output_dir, output_filename, indexing=False)
    
    return final_df

def _process_plant_annotations(
    raw_data_dir: str,
    output_dir: str,
    output_filename: str,
    identifier_column: str,
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

    df = _read_and_select_plant_annotations(annotation_file, identifier_column)
    if df is None or df.empty:
        log.warning(f"No valid data in {os.path.basename(annotation_file)}")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df

    # Aggregate repeated annotation rows by the shared transcriptome identifier.
    agg_df = _aggregate_plant_annotations(df)
    log.info(f"  Processed {os.path.basename(annotation_file)}: {len(agg_df)} identifiers")

    # Fill NaN values with empty strings for consistency
    agg_df = agg_df.fillna('')

    merged_df = _add_protein_id_mapping(
        agg_df,
        raw_data_dir,
        "plant",
        identifier_attribute=identifier_column,
    )

    # Compress to one gene per row with semicolon-separated annotations
    final_df = _compress_annotation_table(merged_df)

    log.info(f"Final annotation table: {len(final_df)} genes with {len(final_df.columns)} annotation columns")

    # Save the merged annotation table
    write_integration_file(final_df, output_dir, output_filename, indexing=False)

    return final_df


def _read_and_select_plant_annotations(
    file_path: str,
    identifier_column: str,
) -> pd.DataFrame:
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

        _validate_annotation_identifier_header(df, file_path, identifier_column)

        # Strip the source-format marker after validating the literal header.
        df.columns = [c.lstrip('#') for c in df.columns]
        source_identifier = identifier_column.lstrip('#')

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
        if source_identifier not in df.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' was not found in "
                f"{os.path.basename(file_path)} after header normalization."
            )
        available_map[source_identifier] = 'transcriptome_id'
        df = df[list(available_map.keys())].copy()
        df.rename(columns=available_map, inplace=True)

        df['transcriptome_id'] = df['transcriptome_id'].astype(str)
        df = df[df['transcriptome_id'].str.strip() != '']
        if 'protein_id' in df.columns:
            df['protein_id'] = df['protein_id'].astype(str)

        return df

    except ValueError:
        raise
    except Exception as e:
        log.warning(f"Error reading plant annotation file {file_path}: {e}")
        return None


def _aggregate_plant_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate plant annotations by ``transcriptome_id``.

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

    annotation_columns = [col for col in df.columns if col != 'transcriptome_id']
    aggregated_df = (
        df.groupby('transcriptome_id')[annotation_columns]
        .agg(join_unique_values)
        .reset_index()
    )

    return aggregated_df


def _read_and_select_microbe_annotations(
    file_path: str,
    identifier_column: str,
) -> pd.DataFrame:
    """
    Read and select relevant columns from microbe annotation files.
    
    Args:
        file_path (str): Path to annotation file
        
    Returns:
        pd.DataFrame: Processed annotation dataframe
    """
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        _validate_annotation_identifier_header(df, file_path, identifier_column)
        
        # Determine annotation type based on filename
        filename = os.path.basename(file_path)
        
        if 'cog_annotation_table' in filename:
            selected_cols = [identifier_column, 'cog_id', 'cog_name']
            df = df[selected_cols].copy()
            df.rename(columns={identifier_column: 'transcriptome_id', 'cog_id': 'cog_acc', 'cog_name': 'cog_desc'}, inplace=True)
            
        elif 'ipr_annotation_table' in filename:
            selected_cols = [identifier_column, 'iprid', 'iprdesc', 'go_info']
            df = df[selected_cols].copy()
            df.rename(columns={identifier_column: 'transcriptome_id', 'iprid': 'ipr_acc', 'iprdesc': 'ipr_desc', 'go_info': 'go_acc'}, inplace=True)
            
        elif 'kegg_annotation_table' in filename:
            selected_cols = [identifier_column, 'ko_id', 'ko_name']
            df = df[selected_cols].copy()
            df.rename(columns={identifier_column: 'transcriptome_id', 'ko_id': 'kegg_acc', 'ko_name': 'kegg_desc'}, inplace=True)
            
        elif 'pfam_annotation_table' in filename:
            selected_cols = [identifier_column, 'pfam_id', 'pfam_name']
            df = df[selected_cols].copy()
            df.rename(columns={identifier_column: 'transcriptome_id', 'pfam_id': 'pfam_acc', 'pfam_name': 'pfam_desc'}, inplace=True)
            
        elif 'tigrfam_annotation_table' in filename:
            selected_cols = [identifier_column, 'tigrfam_id', 'tigrfam_name']
            df = df[selected_cols].copy()
            df.rename(columns={identifier_column: 'transcriptome_id', 'tigrfam_id': 'tigrfam_acc', 'tigrfam_name': 'tigrfam_desc'}, inplace=True)
            
        else:
            log.warning(f"Unknown annotation file type: {filename}")
            return None
            
        return df
        
    except ValueError:
        raise
    except Exception as e:
        log.warning(f"Error reading file {file_path}: {e}")
        return None

def _read_and_select_algal_annotations(
    file_path: str,
    identifier_column: str,
) -> pd.DataFrame:
    """
    Read and select relevant columns from algal annotation files.
    
    Args:
        file_path (str): Path to annotation file
        
    Returns:
        pd.DataFrame: Processed annotation dataframe
    """
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        _validate_annotation_identifier_header(df, file_path, identifier_column)
        
        # Determine annotation type based on filename
        filename = os.path.basename(file_path)
        
        if 'go_annotation_table' in filename:
            # Select proteinId and GO annotation columns
            selected_cols = [identifier_column, 'gotermId', 'goName', 'gotermType', 'goAcc']
            df = df[selected_cols].copy()
            df.rename(columns={
                identifier_column: 'protein_id',
                'gotermId': 'go_id',
                'goName': 'go_name',
                'gotermType': 'go_type',
                'goAcc': 'go_acc'
            }, inplace=True)
            
        elif 'ipr_annotation_table' in filename:
            # Select proteinId and InterPro annotation columns
            selected_cols = [identifier_column, 'iprId', 'iprDesc']
            df = df[selected_cols].copy()
            df.rename(columns={
                identifier_column: 'protein_id',
                'iprId': 'ipr_acc',
                'iprDesc': 'ipr_desc',
                'goAcc': 'go_acc'
            }, inplace=True)
            
        elif 'kegg_annotation_table' in filename:
            # Select proteinId and KEGG annotation columns
            selected_cols = [identifier_column, 'ecNum', 'definition']
            df = df[selected_cols].copy()
            df.rename(columns={
                identifier_column: 'protein_id',
                'ecNum': 'kegg_ec',
                'definition': 'kegg_desc'
            }, inplace=True)
            
        elif 'kog_annotation_table' in filename:
            # Select proteinId and KOG annotation columns
            selected_cols = [identifier_column, 'kogid', 'kogdefline']
            df = df[selected_cols].copy()
            df.rename(columns={
                identifier_column: 'protein_id',
                'kogid': 'kog_acc',
                'kogdefline': 'kog_desc'
            }, inplace=True)
            
        else:
            log.warning(f"Unknown annotation file type: {filename}")
            return None
            
        return df
        
    except ValueError:
        raise
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
    genome_type: str,
    identifier_attribute: str = None,
) -> pd.DataFrame:
    """
    Resolve the selected GFF3 identifier and display name for annotation rows.
    
    Args:
        merged_df (pd.DataFrame): Merged annotation dataframe
        raw_data_dir (str): Directory containing GFF3 file
        genome_type (str): Genome type ("microbe", "algal", etc.)
        identifier_attribute (str): GFF3 attribute selected by the counts header
        
    Returns:
        pd.DataFrame: Annotation dataframe with protein ID and display_name columns added
    """
    
    # Look for GFF3 file in the raw data directory
    gff_files = glob.glob(os.path.join(raw_data_dir, "*.gff3"))
    if not gff_files:
        gff_files = glob.glob(os.path.join(raw_data_dir, "genes.gff3"))
    
    if not gff_files:
        raise FileNotFoundError(
            f"No GFF3 file found in {raw_data_dir} for identifier "
            f"'{identifier_attribute}'."
        )
    
    gff_file = gff_files[0]  # Use first GFF3 file found
    log.info(f"Adding protein ID mapping and display names from {os.path.basename(gff_file)}")
    
    feature_types = {
        "algal": ("gene", "product_name"),
        "microbe": ("CDS", "product"),
        "plant": ("mRNA", "Name"),
    }
    feature_type, display_attribute = feature_types.get(
        genome_type, ("gene", "product")
    )
    identifier_attribute = identifier_attribute or "ID"
    identifier_to_display = {}
    
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            if fields[2] == feature_type:
                attributes = fields[8]
                identifier_match = re.search(
                    rf"(?:^|;){re.escape(identifier_attribute)}=([^;]+)",
                    attributes,
                )
                if not identifier_match:
                    continue

                identifier = identifier_match.group(1)
                display_match = re.search(
                    rf"(?:^|;){re.escape(display_attribute)}=([^;]+)",
                    attributes,
                )
                identifier_to_display[identifier] = (
                    display_match.group(1).strip() if display_match else identifier
                )

    log.info(
        f"Found {len(identifier_to_display)} {feature_type} records with "
        f"GFF3 identifier '{identifier_attribute}'"
    )
    if not identifier_to_display:
        raise ValueError(
            f"No GFF3 {feature_type} records contain the requested identifier "
            f"attribute '{identifier_attribute}'. Check the first-column "
            "header in counts.csv."
        )

    if genome_type == "algal" and 'protein_id' in merged_df.columns:
        merged_df['transcriptome_id'] = merged_df['protein_id'].astype(str)
    if 'transcriptome_id' not in merged_df.columns:
        raise ValueError(
            f"Annotation table for {genome_type} is missing the selected "
            "transcriptome identifier."
        )

    merged_df['transcriptome_id'] = merged_df['transcriptome_id'].astype(str)
    merged_df['display_name'] = merged_df['transcriptome_id'].map(
        identifier_to_display
    ).fillna('')
    if 'protein_id' not in merged_df.columns:
        merged_df['protein_id'] = ''

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

    # Dev branch splits reactions across reaction_00.tsv, reaction_01.tsv, ...
    _MODELSEED_REACTION_PART_URL = (
        "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
        "dev/Biochemistry/reaction_{:02d}.tsv"
    )

    if cache_path.exists():
        log.info(f"Loading ModelSEED reactions from local cache: {cache_path}")
    else:
        parts = []
        part_idx = 0
        while True:
            url = _MODELSEED_REACTION_PART_URL.format(part_idx)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:  # first missing index marks the end
                break
            resp.raise_for_status()
            parts.append(pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False))
            part_idx += 1

        if not parts:
            raise RuntimeError("No ModelSEED reaction_##.tsv files found on the dev branch.")

        log.info(f"Fetched and concatenated {len(parts)} ModelSEED reaction part files")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(cache_path, sep="\t", index=False)
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

    for entry in value.split(";"):
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
        for ec in ec_field.split(";"):
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
    # Dev branch splits compounds across compound_00.tsv, compound_01.tsv, ...
    _MODELSEED_COMPOUND_PART_URL = (
        "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
        "dev/Biochemistry/compound_{:02d}.tsv"
    )
    if cache_path.exists():
        log.info(f"Loading ModelSEED compounds from local cache: {cache_path}")
    else:
        parts = []
        part_idx = 0
        while True:
            url = _MODELSEED_COMPOUND_PART_URL.format(part_idx)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:  # first missing index marks the end
                break
            resp.raise_for_status()
            parts.append(pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False))
            part_idx += 1

        if not parts:
            raise RuntimeError("No ModelSEED compound_##.tsv files found on the dev branch.")

        log.info(f"Fetched and concatenated {len(parts)} ModelSEED compound part files")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(cache_path, sep="\t", index=False)
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
    #log.warning(
    #    "Fuzzy name match used: '%s' ~ ModelSEED '%s' (cpd=%s, score=%.1f)",
    #    name, matched_text, cpd_id, score,
    #)
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

    if rxn_col in df.columns:
        rxn_cols = [rxn_col]
    else:
        rxn_cols = []
    # Use every reaction column (tx_ and px_ share the TX annotation format)
    rxn_cols = list(dict.fromkeys(
        rxn_cols + [c for c in df.columns if c.endswith("_modelseed_rxn") or c == "modelseed_rxn"]
    ))
    if rxn_cols:
        log.info(f"Using reaction column(s) for pathway assignment: {rxn_cols}")

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
        for col in rxn_cols:
            pathways |= _resolve(row[col], rxn_to_pathways)
        if cpd_col is not None:
            pathways |= _resolve(row[cpd_col], cpd_to_pathways)
        return sep.join(sorted(pathways)) if pathways else missing_token

    out = df.copy()
    out[out_col] = out.apply(_row_pathway, axis=1)

    n_mapped = (out[out_col] != missing_token).sum()

    n_tx_mapped = (
        (~out[rxn_cols].apply(lambda s: s.apply(_is_missing)).all(axis=1) & (out[out_col] != missing_token)).sum()
        if rxn_cols else 0
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
    identifier_column: str = None,
    prefix: str = "tx",
) -> pd.DataFrame:
    """
    Generate a merged gene annotation table from multiple annotation files.

    The same annotation files/formats serve tx and px data; ``prefix`` selects the
    feature-ID prefix ("tx" or "px") used to match annotations to the data.

    Args:
        raw_data (pd.DataFrame): Raw transcriptomics data with gene IDs as index
        raw_data_dir (str): Directory containing annotation table files
        genome_type (str): Type of genome - "microbe", "algal", "metagenome", or "plant"
        output_dir (str): Output directory for saving the merged annotation table
        output_filename (str): Filename for the output annotation table

    Returns:
        pd.DataFrame: Merged annotation table with transcriptome_id and annotation columns
    """

    identifier_column = identifier_column or raw_data.columns[0]

    if genome_type == "microbe":
        annotation_df = _process_microbe_annotations(
            raw_data_dir, output_dir, output_filename, identifier_column
        )
    elif genome_type == "algal":
        annotation_df = _process_algal_annotations(
            raw_data_dir,
            output_dir,
            output_filename,
            identifier_column=identifier_column,
        )
    elif genome_type == "metagenome":
        log.info(f"Annotation processing for '{genome_type}' genome type is not yet implemented.")
        empty_df = pd.DataFrame(columns=['transcriptome_id'])
        write_integration_file(empty_df, output_dir, output_filename, indexing=False)
        return empty_df
    elif genome_type == "plant":
        annotation_df = _process_plant_annotations(
            raw_data_dir, output_dir, output_filename, identifier_column
        )
    else:
        raise ValueError(f"Invalid genome_type '{genome_type}'. Must be one of: 'microbe', 'algal', 'metagenome', 'plant'")

    if not raw_data.empty and not annotation_df.empty:
        _validate_annotation_gene_ids(annotation_df, raw_data, prefix)
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
    cache_dir: str = None,
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
        final_annotation_df, cache_path=Path(cache_dir) / "modelseed_reactions.tsv"
    )

    # Add KEGG pathway columns (mx_kegg_id, kegg_pathway, kegg_pathway_name),
    # auto-detecting dataset-prefixed InChIKey/EC/KO columns, with API
    # responses cached under output_dir so re-runs don't hit KEGG/MW again.
    final_annotation_df = add_kegg_pathway_annotations(
        final_annotation_df,
        cache_dir=Path(cache_dir) if cache_dir else None,
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
                        ~final_annotation_df.loc[dataset_feature_indices, dataset_annotation_cols].isin(["Unassigned", "", None])
                    ).any(axis=1)
                    dataset_annotated_count = dataset_annotated_mask.sum()
                else:
                    dataset_annotated_count = 0

                any_annotation_mask = (
                    ~final_annotation_df.loc[dataset_feature_indices, final_annotation_df.columns[1:]].isin(["Unassigned", "", None])
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

def split_list_cell(
    cell,
    delimiter: str = ";",
    drop_sentinel: Optional[str] = None,
    keep_empty: bool = True,
) -> List[str]:
    """
    Split a delimited cell into a list of tokens.

    - delimiter: character to split on (';' or ',').
    - drop_sentinel: if set (e.g. 'Unassigned'), tokens matching it
      (case-insensitive) are dropped.
    - keep_empty: if True, preserves empty slots in the output (needed
      for structure-preserving mappings like InChIKey -> KEGG ID -> pathway,
      where position across columns must line up). Set False for columns
      like tx_kegg_ec where we just want a deduplicated set.
    """
    if pd.isna(cell):
        return []
    tokens = [part.strip() for part in str(cell).split(delimiter)]
    if drop_sentinel is not None:
        tokens = [t for t in tokens if t.lower() != drop_sentinel.lower()]
    if not keep_empty:
        tokens = [t for t in tokens if t]
    return tokens

MW_COMPOUND_URL = "https://www.metabolomicsworkbench.org/rest/compound/inchi_key/{}/all/"


def fetch_kegg_id(
    inchikey: str,
    session: requests.Session,
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.0,
) -> Optional[str]:
    """Query MW REST API for one InChIKey and return its kegg_id (or None)."""
    url = MW_COMPOUND_URL.format(inchikey)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            # API can return a dict, a list of dicts, or an "error" payload
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                return None
            kegg_id = data.get("kegg_id")
            return kegg_id if kegg_id else None
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                print(f"Warning: failed to fetch '{inchikey}' after {retries} attempts: {exc}")
                return None
            time.sleep(backoff * attempt)
    return None


def build_inchikey_kegg_lookup(
    inchikeys: List[str],
    delay: float = 0.2,
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[str]]:
    """Query MW API for each InChIKey not already in cache; return {inchikey: kegg_id}."""

    def fetch_missing(missing_keys: List[str]) -> Dict[str, Optional[str]]:
        result: Dict[str, Optional[str]] = {}
        with requests.Session() as session:
            session.headers.update({"User-Agent": "inchikey-to-kegg-lookup/1.0"})
            for key in tqdm(missing_keys, desc="Fetching KEGG IDs", unit="InChIKey"):
                result[key] = fetch_kegg_id(key, session)
                time.sleep(delay)
        return result

    return get_cached_lookup(inchikeys, cache_path, fetch_missing)


def add_kegg_compound_id_column(
    df: pd.DataFrame,
    inchikey_col: str = "mx_InChiKey",
    new_col: str = "mx_kegg_id",
    delay: float = 0.2,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    df = df.copy()
    key_lists = df[inchikey_col].apply(
        lambda cell: split_list_cell(cell, delimiter=";", keep_empty=True)
    )
    all_keys = sorted({k for keys in key_lists for k in keys if k})
    lookup = build_inchikey_kegg_lookup(all_keys, delay=delay, cache_path=cache_path)

    def map_row(keys: List[str]) -> Optional[str]:
        if not keys or all(not k for k in keys):
            return None
        return ";".join((lookup.get(k) or "") if k else "" for k in keys)

    df[new_col] = key_lists.apply(map_row)
    return df


def fetch_kegg_links_batch(
    ids: List[str],
    prefix: str,
    session: requests.Session,
    target: str = "pathway",
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.0,
) -> Dict[str, List[str]]:
    """
    Query KEGG REST 'link' API for a batch (<=10) of KEGG IDs sharing the
    same db prefix (e.g. 'cpd:', 'ec:', 'ko:') and return
    {id_without_prefix: [target_id, ...]}.
    """
    result: Dict[str, List[str]] = {i: [] for i in ids}
    query = "+".join(f"{prefix}{i}" for i in ids)
    
    KEGG_LINK_URL = "https://rest.kegg.jp/link/{target}/{query}"
    url = KEGG_LINK_URL.format(target=target, query=query)

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                # No links found for any ID in this batch
                return result
            resp.raise_for_status()
            for line in resp.text.strip().splitlines():
                if not line:
                    continue
                source_entry, target_entry = line.split("\t")
                key = source_entry.split(":", 1)[1]
                result.setdefault(key, []).append(target_entry)
            return result
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                print(f"Warning: failed to fetch '{target}' links for batch {ids} after {retries} attempts: {exc}")
                return result
            time.sleep(backoff * attempt)
    return result


def build_kegg_link_lookup(
    ids: List[str],
    prefix: str,
    target: str = "pathway",
    delay: float = 0.2,
    batch_size: int = 10,
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """Batch-query the KEGG link API for IDs not already in cache."""

    def fetch_missing(missing_ids: List[str]) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        with requests.Session() as session:
            session.headers.update({"User-Agent": "kegg-link-lookup/1.0"})
            desc = f"Fetching KEGG {target} links ({prefix.rstrip(':')})"
            for i in tqdm(range(0, len(missing_ids), batch_size), desc=desc, unit="batch"):
                chunk = missing_ids[i : i + batch_size]
                lookup.update(fetch_kegg_links_batch(chunk, prefix, session, target=target))
                time.sleep(delay)
        return lookup

    return get_cached_lookup(ids, cache_path, fetch_missing)

def fetch_kegg_pathway_names_cached(
    cache_path: Optional[Union[str, Path]] = None,
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.0,
) -> Dict[str, str]:
    """
    Fetch the full KEGG reference pathway ID -> name list, using a cached
    copy if available (this is a single bulk API call, so caching is
    all-or-nothing rather than per-key).
    """
    cache = load_json_cache(cache_path)
    if cache:
        return cache

    KEGG_LIST_URL = "https://rest.kegg.jp/list/{db}"
    url = KEGG_LIST_URL.format(db="pathway")
    with requests.Session() as session:
        session.headers.update({"User-Agent": "kegg-pathway-name-lookup/1.0"})
        for attempt in range(1, retries + 1):
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                names: Dict[str, str] = {}
                for line in resp.text.strip().splitlines():
                    if not line:
                        continue
                    pathway_id, name = line.split("\t")
                    names[pathway_id.replace("path:", "")] = name
                save_json_cache(cache_path, names)
                return names
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries:
                    log.warning(f"Failed to fetch KEGG pathway names after {retries} attempts: {exc}")
                    return {}
                time.sleep(backoff * attempt)
    return {}

def add_compound_pathway_column(
    df: pd.DataFrame,
    kegg_id_col: str = "mx_kegg_id",
    pathway_col: str = "kegg_pathway",
    pathway_sep: str = ";",
    delay: float = 0.2,
    batch_size: int = 10,
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    df = df.copy()
    if pathway_col not in df.columns:
        df[pathway_col] = None

    id_lists = df[kegg_id_col].apply(
        lambda cell: split_list_cell(cell, delimiter=";", keep_empty=True)
    )
    all_ids = sorted({k for ids in id_lists for k in ids if k})
    lookup = build_kegg_link_lookup(
        all_ids, prefix="cpd:", target="pathway", delay=delay, batch_size=batch_size, cache_path=cache_path
    )

    def map_row(ids: List[str]) -> Optional[str]:
        if not ids or all(not i for i in ids):
            return None
        return ";".join(pathway_sep.join(lookup.get(cid, []) or []) if cid else "" for cid in ids)

    df[pathway_col] = id_lists.apply(map_row)
    return df


def get_ec_or_ko_ids(
    row: pd.Series,
    ec_col: Union[str, List[str]] = "tx_kegg_ec",
    ko_col: Union[str, List[str]] = "tx_ko_acc",
) -> Tuple[List[str], Optional[str]]:
    """
    Return (ids, source) for a row, using the EC column(s) if present, falling back
    to the KO column(s). source is 'ec', 'ko', or None if neither has values.
    Each argument may be one column name or a list (e.g. tx_ and px_ columns).
    """
    def _collect(cols) -> List[str]:
        cols = [cols] if isinstance(cols, str) else list(cols or [])
        ids: List[str] = []
        for c in cols:
            ids += split_list_cell(row.get(c), delimiter=",", drop_sentinel="Unassigned", keep_empty=False)
        return list(dict.fromkeys(ids))

    ec_ids = _collect(ec_col)
    if ec_ids:
        return ec_ids, "ec"
    ko_ids = _collect(ko_col)
    if ko_ids:
        return ko_ids, "ko"
    return [], None


def add_ec_ko_pathway_column(
    df: pd.DataFrame,
    ec_col: Union[str, List[str]] = "tx_kegg_ec",
    ko_col: Union[str, List[str]] = "tx_ko_acc",
    pathway_col: str = "kegg_pathway",
    pathway_sep: str = ";",
    delay: float = 0.2,
    batch_size: int = 10,
    ec_cache_path: Optional[Union[str, Path]] = None,
    ko_cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    df = df.copy()
    if pathway_col not in df.columns:
        df[pathway_col] = None

    id_source_pairs = df.apply(lambda row: get_ec_or_ko_ids(row, ec_col, ko_col), axis=1)
    id_lists = id_source_pairs.apply(lambda pair: pair[0])
    source_types = id_source_pairs.apply(lambda pair: pair[1])

    ec_ids_all = sorted({i for ids, src in zip(id_lists, source_types) if src == "ec" for i in ids})
    ko_ids_all = sorted({i for ids, src in zip(id_lists, source_types) if src == "ko" for i in ids})

    ec_lookup = (
        build_kegg_link_lookup(ec_ids_all, prefix="ec:", target="pathway",
                                 delay=delay, batch_size=batch_size, cache_path=ec_cache_path)
        if ec_ids_all else {}
    )
    ko_lookup = (
        build_kegg_link_lookup(ko_ids_all, prefix="ko:", target="pathway",
                                 delay=delay, batch_size=batch_size, cache_path=ko_cache_path)
        if ko_ids_all else {}
    )

    def merge_row(existing, ids: List[str], source: Optional[str]) -> Optional[str]:
        if not ids or source is None:
            return existing
        lookup = ec_lookup if source == "ec" else ko_lookup
        new_pathways = list(dict.fromkeys(p for i in ids for p in (lookup.get(i) or [])))
        if not new_pathways:
            return existing
        existing_pathways = []
        if pd.notna(existing) and existing:
            existing_pathways = [t.strip() for t in str(existing).replace(";", pathway_sep).split(pathway_sep) if t.strip()]
        combined = list(dict.fromkeys(existing_pathways + new_pathways))
        return pathway_sep.join(combined) if combined else None

    df[pathway_col] = [
        merge_row(existing, ids, src)
        for existing, ids, src in zip(df[pathway_col], id_lists, source_types)
    ]
    return df

def fetch_kegg_pathway_names(
    session: requests.Session,
    timeout: int = 10,
    retries: int = 3,
    backoff: float = 1.0,
) -> Dict[str, str]:
    """
    Fetch the full KEGG reference pathway list and return
    {pathway_id: pathway_name}, e.g. {'map00010': 'Glycolysis / Gluconeogenesis'}.

    This is a single bulk request — KEGG only has ~550 reference pathways,
    so there's no need to look up names one at a time.
    """
    KEGG_LIST_URL = "https://rest.kegg.jp/list/{db}"
    url = KEGG_LIST_URL.format(db="pathway")

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            names: Dict[str, str] = {}
            for line in resp.text.strip().splitlines():
                if not line:
                    continue
                pathway_id, name = line.split("\t")
                pathway_id = pathway_id.replace("path:", "")
                names[pathway_id] = name
            return names
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                print(f"Warning: failed to fetch KEGG pathway names after {retries} attempts: {exc}")
                return {}
            time.sleep(backoff * attempt)
    return {}


def add_kegg_pathway_name_column(
    df: pd.DataFrame,
    pathway_col: str = "kegg_pathway",
    new_col: str = "kegg_pathway_name",
    pathway_sep: str = ";",
    cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    df = df.copy()
    name_lookup = fetch_kegg_pathway_names_cached(cache_path=cache_path)

    def resolve_pathway_id(pid: str) -> str:
        clean_id = pid.replace("path:", "").strip()
        if not clean_id:
            return ""
        name = name_lookup.get(clean_id)
        if name:
            return name
        m = re.match(r"^[a-z]+(\d+)$", clean_id)
        if m:
            return name_lookup.get(f"map{m.group(1)}", "")
        return ""

    def map_cell(cell) -> Optional[str]:
        if pd.isna(cell) or not cell:
            return None
        resolved_slots = []
        for slot in str(cell).split(";"):
            if not slot:
                resolved_slots.append("")
                continue
            names = [resolve_pathway_id(pid) for pid in slot.split(pathway_sep)]
            resolved_slots.append(pathway_sep.join(names))
        return ";".join(resolved_slots)

    df[new_col] = df[pathway_col].apply(map_cell)
    return df

def _find_columns(df: pd.DataFrame, suffixes: List[str]) -> List[str]:
    """All columns whose name ends with any of the given suffixes (case-insensitive)."""
    return [c for c in df.columns if any(c.lower().endswith(s.lower()) for s in suffixes)]

def _find_column(df: pd.DataFrame, suffixes: List[str]) -> Optional[str]:
    """Find the first column whose name ends with any of the given suffixes (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        for suffix in suffixes:
            if col_lower.endswith(suffix.lower()):
                return col
    return None

def add_kegg_pathway_annotations(
    df: pd.DataFrame,
    cache_dir: Optional[Union[str, Path]] = None,
    inchikey_suffixes: List[str] = ("mx_InChiKey", "InChiKey"),
    ec_suffixes: List[str] = ("tx_kegg_ec", "kegg_ec"),
    ko_suffixes: List[str] = ("tx_ko_acc", "ko_acc"),
    kegg_id_col: str = "mx_kegg_id",
    pathway_col: str = "kegg_pathway",
    pathway_name_col: str = "kegg_pathway_name",
    delay: float = 0.2,
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    Full KEGG pathway annotation pipeline, wired for cached API calls:
      1. InChIKey          -> mx_kegg_id        (MW API, cached)
      2. mx_kegg_id         -> kegg_pathway      (KEGG link API, cpd:, cached)
      3. EC/KO (fallback)   -> kegg_pathway      (KEGG link API, ec:/ko:, cached, merged)
      4. kegg_pathway       -> kegg_pathway_name (KEGG list API, cached)

    Auto-detects source columns by suffix match, since upstream joins may
    prefix columns with dataset names (e.g. 'metabolomics_mx_InChiKey').
    Silently skips steps whose required source column isn't found.
    """
    df = df.copy()
    cache_dir = Path(cache_dir) if cache_dir else None

    inchikey_col = _find_column(df, list(inchikey_suffixes))
    ec_col = _find_columns(df, list(ec_suffixes))
    ko_col = _find_columns(df, list(ko_suffixes))

    if inchikey_col:
        log.info(f"KEGG annotation: using InChIKey column '{inchikey_col}'")
        df = add_kegg_compound_id_column(
            df, inchikey_col=inchikey_col, new_col=kegg_id_col, delay=delay,
            cache_path=cache_dir / "kegg_inchikey_to_id.json" if cache_dir else None,
        )
        df = add_compound_pathway_column(
            df, kegg_id_col=kegg_id_col, pathway_col=pathway_col, delay=delay, batch_size=batch_size,
            cache_path=cache_dir / "kegg_cpd_to_pathway.json" if cache_dir else None,
        )
    else:
        log.warning("KEGG annotation: no InChIKey column found, skipping compound->pathway mapping")

    if ec_col or ko_col:
        log.info(f"KEGG annotation: using EC column '{ec_col}', KO column '{ko_col}'")
        df = add_ec_ko_pathway_column(
            df, ec_col=ec_col, ko_col=ko_col, pathway_col=pathway_col, delay=delay, batch_size=batch_size,
            ec_cache_path=cache_dir / "kegg_ec_to_pathway.json" if cache_dir else None,
            ko_cache_path=cache_dir / "kegg_ko_to_pathway.json" if cache_dir else None,
        )
    else:
        log.warning("KEGG annotation: no EC or KO column found, skipping EC/KO->pathway mapping")

    if pathway_col in df.columns:
        df = add_kegg_pathway_name_column(
            df, pathway_col=pathway_col, new_col=pathway_name_col,
            cache_path=cache_dir / "kegg_pathway_names.json" if cache_dir else None,
        )

    return df

# ===============================
# Helper scripts for annotation caching
# ===============================

def load_json_cache(cache_path: Optional[Union[str, Path]]) -> Dict:
    """Load a JSON cache file if it exists, else return an empty dict."""
    if cache_path is None:
        return {}
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"Failed to load cache {cache_path}: {exc}")
            return {}
    return {}


def save_json_cache(cache_path: Optional[Union[str, Path]], data: Dict) -> None:
    """Write a dict to a JSON cache file, creating parent dirs as needed."""
    if cache_path is None:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)


def get_cached_lookup(
    keys: List[str],
    cache_path: Optional[Union[str, Path]],
    fetch_fn: Callable[[List[str]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generic disk-cached key -> value lookup.

    Loads an existing JSON cache, determines which `keys` are missing,
    calls `fetch_fn(missing_keys)` to fetch only those, merges the result
    into the cache, writes it back to disk, and returns the subset of the
    (now up-to-date) cache relevant to `keys`.
    """
    cache = load_json_cache(cache_path)
    missing = [k for k in keys if k not in cache]

    if missing:
        log.info(f"KEGG cache: fetching {len(missing)} new entries "
                  f"({len(keys) - len(missing)} already cached) -> {cache_path}")
        new_data = fetch_fn(missing)
        cache.update(new_data)
        save_json_cache(cache_path, cache)
    else:
        log.info(f"KEGG cache: all {len(keys)} entries found in cache, no API calls needed ({cache_path})")

    return {k: cache.get(k) for k in keys}


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


def load_link_table_metadata(
    datasets: list,
    link_table_path: Union[str, Path],
    return_link_table: bool = False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load per-dataset metadata and raw-sample mappings from a link table.

    The link table must contain one raw sample-name column for every dataset
    name. Blank cells mean that a shared sample is absent from that datatype.
    One shared-name column is required (``unique_group``, ``shared_sample``,
    ``shared_name``, or ``sample``); all remaining columns are categories.
    """
    path = Path(link_table_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata link table not found: {path}")

    link_table = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    link_table.columns = link_table.columns.astype(str).str.strip()
    dataset_names = [ds.dataset_name for ds in datasets]
    missing_columns = [name for name in dataset_names if name not in link_table.columns]
    if missing_columns:
        raise ValueError(
            f"Metadata link table is missing dataset columns: {missing_columns}. "
            f"Expected columns for {dataset_names}."
        )

    shared_candidates = [
        column for column in ("unique_group", "shared_sample", "shared_name", "sample")
        if column in link_table.columns and column not in dataset_names
    ]
    if not shared_candidates:
        raise ValueError(
            "Metadata link table must contain a shared sample-name column: "
            "unique_group, shared_sample, shared_name, or sample."
        )
    shared_column = shared_candidates[0]
    if shared_column != "unique_group":
        link_table = link_table.rename(columns={shared_column: "unique_group"})
    link_table["unique_group"] = link_table["unique_group"].astype(str).str.strip()
    if (link_table["unique_group"] == "").any() or link_table["unique_group"].eq("nan").any():
        raise ValueError("Metadata link table contains a blank shared sample name.")

    linked_metadata = {}
    for ds in datasets:
        sample_column = ds.dataset_name
        table = link_table.copy()
        table[sample_column] = table[sample_column].replace(r"^\s*$", pd.NA, regex=True)
        table = table.dropna(subset=[sample_column]).copy()
        table[sample_column] = table[sample_column].astype(str)

        if table[sample_column].duplicated().any():
            duplicates = table.loc[
                table[sample_column].duplicated(keep=False), sample_column
            ].unique().tolist()
            raise ValueError(
                f"Metadata link table has duplicate {ds.dataset_name} sample names: "
                f"{duplicates}"
            )

        table = table.set_index("unique_group", drop=False)
        linked_metadata[ds.dataset_name] = table
        write_integration_file(
            data=table,
            output_dir=ds.output_dir,
            filename=ds._linked_metadata_filename,
            indexing=True,
        )
        log.info(
            "Linked %d %s samples from %s",
            len(table), ds.dataset_name, path,
        )

    if return_link_table:
        return linked_metadata, link_table
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
    overlap_only: bool = True,
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

        # Link-table columns are named after dataset names and contain the raw
        # quantitative-table sample names.
        unifying_col = "unique_group"
        sample_col = ds.dataset_name
        if sample_col not in ds.linked_metadata.columns:
            raise ValueError(
                f"Link table metadata is missing raw sample column '{sample_col}'."
            )
        
        # Get library names and create data subset
        library_names = ds.linked_metadata[sample_col].dropna().tolist()
        data_subset = ds.raw_data[
            [ds.raw_data.columns[0]]
            + [col for col in ds.raw_data.columns if col in library_names]
        ].copy()
        
        # Create mapping from library names to unified group names
        mapping = dict(zip(ds.linked_metadata[sample_col], ds.linked_metadata[unifying_col]))
        data_subset.columns = [data_subset.columns[0]] + [mapping.get(col, col) for col in data_subset.columns[1:]]
        
        # Handle duplicate columns (occurs in metabolomics datasets with multiple polarities)
        if data_subset.columns.duplicated().any():
            data_subset = data_subset.T.groupby(data_subset.columns).sum().T
        
        unified_data[ds.dataset_name] = data_subset
        sample_sets[ds.dataset_name] = set(data_subset.columns[1:])

    # Restrict to overlapping samples if requested
    if overlap_only and len(unified_data) > 1:
        log.info("\tRestricting matching samples to only those present in all datasets...")
        overlapping_columns = set.intersection(*sample_sets.values())
        
        if not overlapping_columns:
            raise ValueError("No overlapping samples found across datasets.")
        
        for name, data_subset in unified_data.items():
            feature_id_column = data_subset.columns[0]
            cols = [feature_id_column] + [
                col for col in overlapping_columns if col in data_subset.columns
            ]
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

def integrate_data(
    datasets: list,
    overlap_only: bool = True,
    output_filename: str = "integrated_data",
    output_dir: str = None,
    data_attr: str = "scaled_data",
) -> pd.DataFrame:
    """
    Concatenate per-dataset feature matrices into a single integrated matrix.

    Used by both tracks after per-dataset scaling is complete:

    * **sample_resolution** — concatenates ``ds.scaled_data`` (features x samples)
      across datasets.  Columns are sample IDs; overlapping samples are enforced
      when ``overlap_only=True``.
    * **condition_resolution** — concatenates ``ds.scaled_data`` (features x contrasts)
      across datasets.  Columns are contrast names (``A_vs_B``); shared contrasts
      are enforced when ``overlap_only=True``.

    In both cases the function is purely a concatenation step — all transformation
    logic lives in ``scale_all_datasets()`` / ``scale_data_lfc()``.

    Parameters
    ----------
    datasets : list
        Dataset objects.  Each must have a non-empty ``data_attr`` attribute.
    overlap_only : bool
        If True, restrict columns to those present in every dataset.
    output_filename : str
        Output CSV filename (without extension).
    output_dir : str
        Directory to write the output file.
    data_attr : str
        Per-dataset attribute to read.  Always ``"scaled_data"`` — both tracks
        write their output to ``ds.scaled_data``.

    Returns
    -------
    pd.DataFrame
        Integrated features x columns matrix.
    """
    log.info(f"Concatenating per-dataset matrices (data_attr='{data_attr}')...")

    dataset_data: Dict[str, pd.DataFrame] = {}
    sample_sets: Dict[str, set] = {}

    for ds in datasets:
        source_df = getattr(ds, data_attr, None)
        if source_df is None or source_df.empty:
            raise ValueError(
                f"Dataset '{ds.dataset_name}' missing attribute '{data_attr}'. "
                "Run scale_all_datasets() before integrate_data() in sample_resolution track."
            )
        data_copy = source_df.copy()
        if not data_copy.index.astype(str).str.startswith(f"{ds.dataset_name}_").all():
            data_copy.index = [f"{ds.dataset_name}_{idx}" for idx in data_copy.index]
        dataset_data[ds.dataset_name] = data_copy
        sample_sets[ds.dataset_name] = set(data_copy.columns.astype(str))
        log.info(f"\t{ds.dataset_name}: {data_copy.shape[0]} features")

    if overlap_only and len(dataset_data) > 1:
        log.info("\tRestricting to overlapping columns across all datasets...")
        overlapping_cols = set.intersection(*sample_sets.values())
        if not overlapping_cols:
            raise ValueError(
                f"No overlapping columns found across datasets for attr='{data_attr}'."
            )
        shared_cols = sorted(list(overlapping_cols))
        for ds_name, data in dataset_data.items():
            data.columns = data.columns.astype(str)
            dataset_data[ds_name] = data[shared_cols]
        log.info(f"\t{len(shared_cols)} overlapping columns retained.")

    integrated_data = pd.concat(dataset_data.values(), axis=0)
    integrated_data.index.name = "features"
    integrated_data = integrated_data.fillna(0)

    log.info(
        f"Integrated dataset: {integrated_data.shape[0]} features x "
        f"{integrated_data.shape[1]} columns"
    )

    if output_dir:
        log.info("Writing integrated data table...")
        write_integration_file(integrated_data, output_dir, output_filename, indexing=True)

    return integrated_data


# def scale_data_lfc(
#     data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     dataset_name: str,
#     output_filename: str,
#     output_dir: str,
#     group_col: str = "group",
#     sample_col: str = "unique_group",
# ) -> pd.DataFrame:
#     """
#     Per-dataset LFC scaling for the **condition_resolution** track.

#     Transforms a raw feature x sample matrix into a feature x contrasts matrix
#     by applying the following steps in order:

#     1. **log2 transform** — ``log2(x + 1)``
#     2. **Low-variance filter** — drop features with zero variance across all
#        samples (prevents noise amplification).
#     3. **Collapse replicates to per-condition medians** — group samples by
#        condition label from ``metadata`` and compute the median per condition.
#     4. **All-pairwise LFC** — for every unique pair of conditions (A, B),
#        compute ``log2_median[A] − log2_median[B]``.  Each pair becomes one
#        column named ``A_vs_B``.

#     The result is written to disk and returned.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         Raw feature x sample matrix (``replicate_filtered_data``).
#     metadata : pd.DataFrame
#         Sample metadata with at least ``sample_col`` and ``group_col`` columns.
#     dataset_name : str
#         Dataset name prefix (used for logging and feature-name prefixing).
#     output_filename : str
#         Output CSV filename (without extension).
#     output_dir : str
#         Directory to write the output file.
#     group_col : str
#         Metadata column containing condition/group labels.
#     sample_col : str
#         Metadata column containing sample identifiers matching data columns.

#     Returns
#     -------
#     pd.DataFrame
#         Feature x contrasts matrix.  Columns are contrast names of the form
#         ``"groupA_vs_groupB"``.
#     """
#     # ── Step 1: log2 transform ────────────────────────────────────────────────
#     log.info(f"  [{dataset_name}] Applying log2(x+1) transform...")
#     log2_df = np.log2(data.astype(float) + 1)

#     # ── Step 2: low-variance filter ───────────────────────────────────────────
#     row_var = log2_df.var(axis=1)
#     low_var_mask = row_var > 0.0
#     n_dropped = (~low_var_mask).sum()
#     if n_dropped:
#         log.info(f"  [{dataset_name}] Dropped {n_dropped} zero-variance features after log2.")
#     log2_df = log2_df.loc[low_var_mask]

#     # ── Step 3: collapse replicates to per-condition medians ──────────────────
#     if sample_col in metadata.columns:
#         sample_to_group = metadata.set_index(sample_col)[group_col].to_dict()
#     else:
#         sample_to_group = metadata[group_col].to_dict()

#     common_samples = [c for c in log2_df.columns if c in sample_to_group]
#     if not common_samples:
#         raise ValueError(
#             f"Dataset '{dataset_name}': no overlap between data columns and "
#             f"metadata '{sample_col}' values."
#         )
#     log2_df = log2_df[common_samples]

#     condition_labels = pd.Series(
#         [sample_to_group[s] for s in common_samples],
#         index=common_samples,
#         name=group_col,
#     )
#     condition_medians = log2_df.T.groupby(condition_labels).median().T  # features x conditions

#     n_conditions = condition_medians.shape[1]
#     log.info(
#         f"  [{dataset_name}] Collapsed {len(common_samples)} samples → "
#         f"{n_conditions} condition medians."
#     )
#     if n_conditions < 2:
#         raise ValueError(
#             f"Dataset '{dataset_name}': need at least 2 conditions for pairwise LFC, "
#             f"found {n_conditions}: {condition_medians.columns.tolist()}"
#         )

#     # ── Step 4: all-pairwise LFC ──────────────────────────────────────────────
#     pairs = list(combinations(condition_medians.columns.tolist(), 2))
#     log.info(f"  [{dataset_name}] Computing {len(pairs)} pairwise LFC contrasts...")

#     lfc_df = pd.DataFrame(index=condition_medians.index)
#     for a, b in pairs:
#         lfc_df[f"{a}_vs_{b}"] = condition_medians[a] - condition_medians[b]

#     # Prefix feature names with dataset name
#     if not lfc_df.index.astype(str).str.startswith(f"{dataset_name}_").all():
#         lfc_df.index = [f"{dataset_name}_{idx}" for idx in lfc_df.index]

#     log.info(
#         f"  [{dataset_name}]: {lfc_df.shape[0]} features x {lfc_df.shape[1]} contrasts"
#     )

#     write_integration_file(lfc_df, output_dir, output_filename, indexing=True)
#     return lfc_df


def scale_data_moderated_lfc(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    dataset_name: str,
    output_filename: str,
    output_dir: str,
    group_col: str = "group",
    sample_col: str = "unique_group",
    min_reps_for_se: int = 2,
) -> pd.DataFrame:
    """
    Per-dataset **moderated** LFC scaling for the **condition_resolution** track.

    Applies the same log2 → group-median → pairwise-LFC pipeline as
    ``scale_data_lfc()``, but then shrinks each LFC estimate toward zero using
    an empirical-Bayes normal-means model (James–Stein / limma-style shrinkage).

    **Shrinkage rationale**
    For each pairwise contrast ``A_vs_B`` and each feature, the raw LFC is
    ``d = log2_median[A] - log2_median[B]``.  The standard error of ``d`` is
    estimated from the within-group variance of the log2-transformed replicates:

    .. code-block:: text

        SE²(d) = Var_A / n_A + Var_B / n_B

    where ``Var_k`` is the sample variance of log2(x+1) values within group k
    and ``n_k`` is the number of replicates.  Features with fewer than
    ``min_reps_for_se`` replicates in either group fall back to the global
    median SE for that contrast.

    The posterior (shrunken) LFC is computed as:

    .. code-block:: text

        d_shrunk = d x (τ² / (τ² + SE²))

    where ``τ²`` is the empirical prior variance estimated as
    ``max(0, mean(d²) − mean(SE²))`` across all features for that contrast
    (method-of-moments estimator of the signal variance).  When ``τ² ≈ 0``
    (all LFCs are noise), estimates are shrunk to zero.  When ``τ² >> SE²``
    (strong signal), estimates are barely shrunk.

    Parameters
    ----------
    data : pd.DataFrame
        Raw feature x sample matrix (``replicate_filtered_data``).
    metadata : pd.DataFrame
        Sample metadata with at least ``sample_col`` and ``group_col`` columns.
    dataset_name : str
        Dataset name prefix (used for logging and feature-name prefixing).
    output_filename : str
        Output CSV filename (without extension).
    output_dir : str
        Directory to write the output file.
    group_col : str
        Metadata column containing condition/group labels.
    sample_col : str
        Metadata column containing sample identifiers matching data columns.
    min_reps_for_se : int, default 2
        Minimum replicates required in a group to compute a per-feature SE.
        Features/groups below this threshold use the global median SE for
        that contrast instead.

    Returns
    -------
    pd.DataFrame
        Feature x contrasts matrix of shrunken LFC values.  Columns are
        contrast names of the form ``"groupA_vs_groupB"``.
    """
    # ── Step 1: log2 transform ────────────────────────────────────────────────
    log.info(f"  [{dataset_name}] Applying log2(x+1) transform...")
    log2_df = np.log2(data.astype(float) + 1)

    # ── Step 2: low-variance filter ───────────────────────────────────────────
    row_var = log2_df.var(axis=1)
    low_var_mask = row_var > 0.0
    n_dropped = (~low_var_mask).sum()
    if n_dropped:
        log.info(f"  [{dataset_name}] Dropped {n_dropped} zero-variance features after log2.")
    log2_df = log2_df.loc[low_var_mask]

    # ── Step 3: build group → sample mapping ─────────────────────────────────
    if sample_col in metadata.columns:
        sample_to_group = metadata.set_index(sample_col)[group_col].to_dict()
    else:
        sample_to_group = metadata[group_col].to_dict()

    common_samples = [c for c in log2_df.columns if c in sample_to_group]
    if not common_samples:
        raise ValueError(
            f"Dataset '{dataset_name}': no overlap between data columns and "
            f"metadata '{sample_col}' values."
        )
    log2_df = log2_df[common_samples]

    condition_labels = pd.Series(
        [sample_to_group[s] for s in common_samples],
        index=common_samples,
        name=group_col,
    )

    # Build per-group sample lists
    group_to_samples: Dict[str, List[str]] = {}
    for s, g in condition_labels.items():
        group_to_samples.setdefault(g, []).append(s)

    conditions = sorted(group_to_samples.keys())
    n_conditions = len(conditions)
    if n_conditions < 2:
        raise ValueError(
            f"Dataset '{dataset_name}': need at least 2 conditions for pairwise LFC, "
            f"found {n_conditions}: {conditions}"
        )

    # ── Step 4: per-group log2 medians and within-group variances ────────────
    log.info(
        f"  [{dataset_name}] Computing per-group medians and within-group variances "
        f"({n_conditions} conditions)..."
    )
    group_medians: Dict[str, pd.Series] = {}
    group_vars: Dict[str, pd.Series] = {}
    group_n: Dict[str, int] = {}

    for g, samples in group_to_samples.items():
        g_data = log2_df[samples]
        group_medians[g] = g_data.median(axis=1)
        group_vars[g] = g_data.var(axis=1, ddof=1).fillna(0.0)
        group_n[g] = len(samples)

    # ── Step 5: all-pairwise moderated LFC ───────────────────────────────────
    pairs = list(combinations(conditions, 2))
    log.info(
        f"  [{dataset_name}] Computing {len(pairs)} moderated pairwise LFC contrasts..."
    )

    lfc_df = pd.DataFrame(index=log2_df.index)

    for a, b in pairs:
        contrast = f"{a}_vs_{b}"

        # Raw LFC
        raw_lfc = group_medians[a] - group_medians[b]

        # Per-feature SE²: Var_A/n_A + Var_B/n_B
        n_a, n_b = group_n[a], group_n[b]
        se2: pd.Series

        if n_a >= min_reps_for_se and n_b >= min_reps_for_se:
            se2 = group_vars[a] / n_a + group_vars[b] / n_b
        elif n_a >= min_reps_for_se:
            se2 = group_vars[a] / n_a
        elif n_b >= min_reps_for_se:
            se2 = group_vars[b] / n_b
        else:
            # No replicates in either group — SE is undefined; use raw LFC
            log.warning(
                f"  [{dataset_name}] Contrast {contrast}: both groups have < "
                f"{min_reps_for_se} replicates. Using raw LFC (no shrinkage)."
            )
            lfc_df[contrast] = raw_lfc
            continue

        # Replace any negative SE² (numerical noise) with 0
        se2 = se2.clip(lower=0.0)

        # Fall back to global median SE² for features with 0 variance
        median_se2 = float(se2[se2 > 0].median()) if (se2 > 0).any() else 0.0
        se2 = se2.where(se2 > 0, other=median_se2)

        # Empirical prior variance τ² (method-of-moments)
        # τ² = max(0, E[d²] − E[SE²])
        mean_d2 = float((raw_lfc ** 2).mean())
        mean_se2 = float(se2.mean())
        tau2 = max(0.0, mean_d2 - mean_se2)

        # Posterior shrinkage factor: τ² / (τ² + SE²)
        if tau2 == 0.0:
            # All signal is noise — shrink everything to zero
            shrinkage = pd.Series(0.0, index=raw_lfc.index)
        else:
            shrinkage = tau2 / (tau2 + se2)

        shrunken_lfc = raw_lfc * shrinkage

        log.info(
            f"    {contrast}: τ²={tau2:.4f}, mean SE²={mean_se2:.4f}, "
            f"mean shrinkage={float(shrinkage.mean()):.3f}"
        )
        lfc_df[contrast] = shrunken_lfc

    # ── Step 6: prefix feature names ─────────────────────────────────────────
    if not lfc_df.index.astype(str).str.startswith(f"{dataset_name}_").all():
        lfc_df.index = [f"{dataset_name}_{idx}" for idx in lfc_df.index]

    log.info(
        f"  [{dataset_name}]: {lfc_df.shape[0]} features x {lfc_df.shape[1]} contrasts "
        "(moderated LFC)"
    )

    write_integration_file(lfc_df, output_dir, output_filename, indexing=True)
    return lfc_df


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


# def filter_data(
#     data: pd.DataFrame,
#     dataset_name: str,
#     data_type: str,
#     output_dir: str,
#     output_filename: str,
#     filter_method: str,
#     filter_value: float,
# ) -> pd.DataFrame:
#     """
#     Filter features from a dataset based on a minimum average obs value or proportion of samples with feature.

#     Args:
#         data (pd.DataFrame): Feature matrix (features x samples).
#         dataset_name (str): Name of the dataset (used for output).
#         data_type (str): Data type ('counts', 'abundance', etc.).
#         output_dir (str): Output directory.
#         filter_method (str): Filtering method ('minimum', 'proportion', or 'none').
#         filter_value (float): Threshold value for filtering.

#     Returns:
#         pd.DataFrame: Filtered feature matrix.
#     """

#     # Filter data based on the specified method
#     if filter_method == "minimum":
#         row_means = data.mean(axis=1)
#         log.info(f"Filtering out features with {filter_method} method in {dataset_name} that have an average {data_type} value less than {filter_value} across samples...")
#         filtered_data = data[row_means >= filter_value]
#         log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
#     elif filter_method == "proportion":
#         log.info(f"Filtering out features with {filter_method} method in {dataset_name} that were observed in fewer than {filter_value}% samples...")
#         global_min = data.values.min()
#         min_count = (data <= global_min).sum(axis=1)
#         min_proportion = min_count / data.shape[1]
#         filtered_data = data[min_proportion < filter_value / 100]
#         log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
#     elif filter_method == "none":
#         log.info("Not filtering any features.")
#         log.info(f"Keeping all {data.shape[0]} features.")
#         filtered_data = data
#     else:
#         log.info(f"Invalid filter method '{filter_method}'. Please choose 'minimum', 'proportion', or 'none'.")
#         return None

#     log.info(f"Saving filtered data for {dataset_name}...")
#     write_integration_file(filtered_data, output_dir, output_filename, indexing=True)
#     return filtered_data

# def devariance_data(
#     data: pd.DataFrame,
#     filter_value: float,
#     dataset_name: str,
#     output_filename: str,
#     output_dir: str,
#     devariance_mode: str = "none",
# ) -> pd.DataFrame | None:
#     """
#     Remove low-variance features from a dataset using various strategies.

#     Args:
#         data (pd.DataFrame): Feature matrix (features x samples).
#         filter_value (float): Uniform percent of features to remove.
#         dataset_name (str): Name for output file.
#         output_dir (str): Output directory.
#         devariance_mode (str): Devariance method ('percent', 'none').

#     Returns:
#         pd.DataFrame or None: Filtered feature matrix, or None if variance is too low.
#     """

#     if devariance_mode == "percent":
#         log.info(f"Removing {filter_value}% of features with the lowest variance in {dataset_name}...")
#         data_filtered = remove_uniform_percent_low_variance_features(data=data, filter_percent=filter_value)
#     elif devariance_mode == "none":
#         log.info(f"\tNot removing any features based on variance in {dataset_name}. Retaining all {data.shape[0]} features.")
#         log.info(f"Saving devarianced data for {dataset_name}...")
#         data_filtered = data
#     else:
#         log.info(f"Invalid devariance mode '{devariance_mode}'. Please choose 'percent' or 'none'.")
#         return None
    
#     log.info(f"Saving devarianced data for {dataset_name}...")
#     write_integration_file(data_filtered, output_dir, output_filename, indexing=True)
#     return data_filtered

# def scale_data(
#     df: pd.DataFrame,
#     output_filename: str = None,
#     output_dir: str = None,
#     dataset_name: str = None,
#     log2: bool = True,
#     norm_method: str = "modified_zscore"
# ) -> pd.DataFrame:
#     """
#     Normalize and scale feature matrix with multiple methods.
    
#     New norm_methods:
#     - 'quantile': Force identical distributions across all features
#     - 'rank_normal': Rank-based inverse normal transformation
#     - 'vst': Variance stabilizing transformation + z-score
#     - 'vsn': Variance Stabilizing Normalization
#     """
    
#     if norm_method == "none":
#         log.info("Not scaling data.")
#         return df
    
#     # Convert to numeric
#     df = df.apply(pd.to_numeric, errors='coerce')
    
#     # Quantile normalization (forces identical distribution)
#     if norm_method == "quantile":
#         log.info(f"Applying quantile normalization to {dataset_name}...")
        
#         # Get reference distribution (sorted values of all data)
#         reference = np.sort(df.values.flatten())
        
#         # Apply to each column
#         scaled_values = np.zeros_like(df.values)
#         for j in range(df.shape[1]):
#             ranks = rankdata(df.iloc[:, j], method='average')
#             for i in range(df.shape[0]):
#                 idx = int(ranks[i]) - 1
#                 scaled_values[i, j] = reference[idx]
        
#         scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    
#     # Rank-based inverse normal transformation
#     elif norm_method == "rank_normal":
#         log.info(f"Applying rank-based inverse normal transformation to {dataset_name}...")
        
#         def transform_col(col):
#             ranks = rankdata(col, method='average')
#             quantiles = ranks / (len(ranks) + 1)
#             return norm.ppf(quantiles)
        
#         scaled_df = df.apply(transform_col, axis=0)
    
#     # VST + z-score (good for count data)
#     elif norm_method == "vst":
#         log.info(f"Applying VST transformation to {dataset_name}...")
#         vst_df = np.arcsinh(np.sqrt(df))
#         scaled_df = vst_df.sub(vst_df.mean(axis=1), axis=0).div(vst_df.std(axis=1), axis=0)
    
#     # VSN - Variance Stabilizing Normalization
#     elif norm_method == "vsn":
#         log.info(f"Applying VSN transformation to {dataset_name}...")
        
#         # Convert to numpy array
#         X = df.values.astype(float)
        
#         # VSN transformation function
#         def _vsn_transform(x: np.ndarray, lam: float) -> np.ndarray:
#             """Element-wise VSN transform: g_λ(x) = log2[(x + sqrt(x² + λ)) / 2]"""
#             x = np.where(x < 0, 0.0, x)
#             return np.log2((x + np.sqrt(x * x + lam)) / 2.0)
        
#         # Estimate lambda by minimizing variance of transformed data
#         def _objective(lam_candidate: float) -> float:
#             if lam_candidate <= 0:
#                 return np.inf
#             Y = _vsn_transform(X, lam_candidate)
#             return np.nanvar(Y)
        
#         # Use scipy's bounded minimizer to find optimal lambda
#         res = minimize_scalar(
#             _objective,
#             bounds=(1e-6, 1e6),
#             method='bounded',
#             options={'xatol': 1e-8}
#         )
        
#         if not res.success:
#             raise RuntimeError(
#                 f"λ estimation failed for VSN: {res.message}. "
#                 "Try a different normalization method."
#             )
        
#         lam = res.x
#         log.info(f"  Estimated λ = {lam:.6f} for VSN transformation")
        
#         # Apply transformation with estimated lambda
#         Y = _vsn_transform(X, lam)
#         scaled_df = pd.DataFrame(Y, index=df.index, columns=df.columns)
    
#     elif norm_method in ["zscore", "modified_zscore", "logfc_mean", "logfc_median", "logfc_geometric_mean"]:
#         if log2 and norm_method not in ["logfc_mean", "logfc_median", "logfc_geometric_mean"]:
#             df = np.log2(df + 1)
#         if norm_method == "zscore":
#             scaled_df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
#         elif norm_method == "modified_zscore":
#             med = df.median(axis=1)
#             centered = df.sub(med, axis=0)
#             mad = centered.abs().median(axis=1)
#             scaled_df = centered.multiply(0.6745).div(mad, axis=0)
#         elif norm_method == "logfc_mean":
#             log.info(f"Scaling {dataset_name} data using log2 fold-change relative to mean...")
#             df_pseudocount = df + 1
#             log_values = np.log2(df_pseudocount)
#             mean_log = log_values.mean(axis=1, skipna=True)
#             scaled_df = log_values.subtract(mean_log, axis=0)
#         elif norm_method == "logfc_median":
#             log.info(f"Scaling {dataset_name} data using log2 fold-change relative to median...")
#             df_pseudocount = df + 1
#             log_values = np.log2(df_pseudocount)
#             median_log = log_values.median(axis=1, skipna=True)
#             scaled_df = log_values.subtract(median_log, axis=0)
#         elif norm_method == "logfc_geometric_mean":
#             log.info(f"Scaling {dataset_name} data using log2 fold-change relative to geometric mean...")
#             df_pseudocount = df + 1
#             geometric_means = df_pseudocount.apply(lambda row: gmean(row.dropna()), axis=1)
#             scaled_df = np.log2(df_pseudocount.divide(geometric_means, axis=0))
#         else:
#             raise ValueError("Please select a valid norm_method: 'zscore', 'modified_zscore', 'logfc_mean', 'logfc_median', or 'logfc_geometric_mean'.")
#     else:
#         raise ValueError(f"Unknown norm_method: {norm_method}")

#     # Ensure output is float, and replace NA/inf/-inf with 0
#     scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

#     if output_dir:
#         log.info(f"Saving scaled data for {dataset_name}...")
#         write_integration_file(scaled_df, output_dir, output_filename, indexing=True)

#     return scaled_df

# def remove_low_replicable_features(
#     data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     dataset_name: str,
#     output_filename: str,
#     output_dir: str,
#     method: str = "variance",
#     group_col: str = "group",
#     threshold: float = 0.5,
#     normalize: bool = True,
#     normalization_scale: float = 1_000_000.0,
#     min_replicates: int = 2,
# ):
#     """
#     Remove features (rows) from `data` with high within-group variability
#     across replicates.

#     Variability is assessed on a column-normalized copy of `data` (each
#     sample column scaled to sum to `normalization_scale`, e.g. CPM-style
#     normalization) so that differences in sample magnitude/depth don't
#     drive apparent within-group variability. The *original* (unnormalized)
#     values are what get filtered and returned/written.

#     Parameters
#     ----------
#     method : {"none", "variance"}
#         - "none": no filtering, keep all features.
#         - "variance": flag a feature if its within-group coefficient of
#           variation (CV = std / mean) exceeds `threshold` in ANY group.
#     normalize : bool
#         If True (default), column-normalize samples before computing CV
#         (recommended when samples differ in scale / depth). Filtering is
#         still applied to the original `data`.
#     normalization_scale : float
#         Target sum for each sample column after normalization (default
#         1,000,000; use 1.0 for fractional/proportion scale, 100.0 for %).
#     min_replicates : int
#         Minimum number of samples a group must have to be evaluated
#         (groups with fewer are skipped / not used to flag features).
#     """

#     def _normalize_columns(df: pd.DataFrame, scale: float) -> pd.DataFrame:
#         col_sums = df.sum(axis=0, skipna=True)
#         col_sums_safe = col_sums.replace(0, np.nan)
#         return df.divide(col_sums_safe, axis=1) * scale

#     def _build_group_sample_map(meta: pd.DataFrame, cols: pd.Index) -> dict:
#         groups = meta[group_col].unique()
#         return {
#             group: [
#                 s for s in meta.loc[meta[group_col] == group, "unique_group"].tolist()
#                 if s in cols
#             ]
#             for group in groups
#         }

#     if method == "none":
#         log.info(
#             f"\tNot removing any features based on replicability in {dataset_name}. "
#             f"Retaining all {data.shape[0]} features."
#         )
#         replicable_data = data
#         log.info(f"Saving replicable data for {dataset_name}...")
#         write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
#         return replicable_data

#     if method != "variance":
#         raise ValueError(
#             "Currently only 'variance' or 'none' methods are supported for "
#             "removing low replicable features."
#         )

#     if group_col not in metadata.columns:
#         raise ValueError(f"Column '{group_col}' not found in metadata.")

#     log.info(
#         f"Removing features with high within-group variability "
#         f"(threshold={threshold}, normalize={normalize})..."
#     )

#     group_sample_map = _build_group_sample_map(metadata, data.columns)
#     valid_groups = {k: v for k, v in group_sample_map.items() if len(v) >= min_replicates}

#     if not valid_groups:
#         log.info(
#             f"No groups with sufficient replicates (>={min_replicates}) found. "
#             f"Keeping all features."
#         )
#         replicable_data = data
#     else:
#         # Compute CV on normalized data so sample-level scale/depth
#         # differences don't drive apparent within-group variance.
#         stats_data = _normalize_columns(data, normalization_scale) if normalize else data

#         keep_mask = pd.Series(True, index=data.index)

#         for group, samples in valid_groups.items():
#             group_data = stats_data[samples]
#             group_mean = group_data.mean(axis=1, skipna=True)
#             group_std = group_data.std(axis=1, skipna=True, ddof=1)
#             group_mean_safe = group_mean.replace(0, np.nan)

#             cv = (group_std / group_mean_safe).abs()
#             high_var_in_group = cv > threshold

#             keep_mask = keep_mask & (~high_var_in_group)

#         replicable_data = data.loc[keep_mask]

#     log.info(
#         f"Started with {data.shape[0]} features; filtered out "
#         f"{data.shape[0] - replicable_data.shape[0]} to keep {replicable_data.shape[0]}."
#     )

#     log.info(f"Saving replicable data for {dataset_name}...")
#     write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
#     return replicable_data


# def normalize_integrated_data(
#     data: pd.DataFrame,
#     method: str,
#     metadata: pd.DataFrame,
#     output_filename: str,
#     output_dir: str,
#     log2: bool = True,
#     group_col: str = "group",
#     sample_col: str = "unique_group",
#     pseudocount: float = 1.0,
#     lfc_pairs: list = None,
# ) -> pd.DataFrame:
#     """
#     Normalize the integrated feature matrix (features x samples, all datasets combined)
#     after integration.

#     This is the post-integration normalization step that replaces the old per-dataset
#     ``scale_data()`` step. It operates on the combined matrix so that all features
#     are normalized on the same scale.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         Integrated feature matrix (features x samples). Rows = features, columns = samples.
#         Should be the replicate-filtered devarianced data concatenated across datasets.
#     method : str
#         Normalization method:
#         - ``"lfc"``            : log2 fold-change of group medians vs. all-group median.
#                                  Produces a features x contrasts matrix where each column
#                                  is a pairwise group comparison (groupA_vs_groupB).
#         - ``"vst"``            : variance-stabilizing transformation (arcsinh(sqrt(x))) + z-score.
#         - ``"zscore"``         : log2(x+1) then z-score per sample (column-wise).
#         - ``"modified_zscore"``: log2(x+1) then modified z-score (median-based) per sample.
#         - ``"rank_normal"``    : rank-based inverse normal transformation per sample.
#     metadata : pd.DataFrame
#         Sample metadata. Required for ``"lfc"`` method (needs ``group_col`` and ``sample_col``).
#         For other methods, pass the integrated metadata for reference.
#     output_filename : str
#         Output filename (without extension).
#     output_dir : str
#         Output directory.
#     log2 : bool, default True
#         Whether to log2-transform before scaling. Used by vst/zscore/modified_zscore methods.
#         Ignored for lfc (which computes its own log2 ratios).
#     group_col : str, default "group"
#         Metadata column containing group labels. Used by ``"lfc"`` method.
#     sample_col : str, default "unique_group"
#         Metadata column containing sample identifiers matching data columns. Used by ``"lfc"``.
#     pseudocount : float, default 1.0
#         Pseudocount added before log2 transformation in lfc mode to avoid log(0).
#     lfc_pairs : list of [str, str], optional
#         Specific pairwise contrasts to compute for ``"lfc"`` method.
#         Each element is [groupA, groupB] and the LFC is log2(median_A / median_B).
#         If None, all pairwise combinations are computed.

#     Returns
#     -------
#     pd.DataFrame
#         Normalized feature matrix. For ``"lfc"``, columns are contrast names
#         (e.g. ``"groupA_vs_groupB"``). For all other methods, columns are sample names.
#     """
#     valid_methods = {"lfc", "vst", "zscore", "modified_zscore", "rank_normal"}
#     if method not in valid_methods:
#         raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

#     log.info(f"Normalizing integrated data using method='{method}' ({data.shape[0]} features x {data.shape[1]} samples)...")

#     if method == "lfc":
#         # Build group medians from the integrated matrix
#         if group_col not in metadata.columns:
#             raise ValueError(f"group_col '{group_col}' not found in metadata columns: {metadata.columns.tolist()}")

#         # Map sample → group
#         if sample_col in metadata.columns:
#             sample_to_group = metadata.set_index(sample_col)[group_col].to_dict()
#         else:
#             sample_to_group = metadata[group_col].to_dict()

#         # Restrict to samples present in data
#         common_samples = [s for s in data.columns if s in sample_to_group]
#         if not common_samples:
#             raise ValueError("No samples in data match the metadata index/sample_col.")

#         data_common = data[common_samples]
#         groups = sorted(set(sample_to_group[s] for s in common_samples))

#         # Compute per-group medians (features x groups)
#         group_medians = pd.DataFrame(index=data.index)
#         for grp in groups:
#             grp_samples = [s for s in common_samples if sample_to_group[s] == grp]
#             if grp_samples:
#                 group_medians[grp] = data_common[grp_samples].median(axis=1)

#         # Determine contrasts
#         if lfc_pairs is None:
#             contrasts = list(combinations(groups, 2))
#         else:
#             contrasts = [tuple(p) for p in lfc_pairs]

#         # Compute log2 fold-changes
#         lfc_df = pd.DataFrame(index=data.index)
#         for grp_a, grp_b in contrasts:
#             col_name = f"{grp_a}_vs_{grp_b}"
#             med_a = group_medians[grp_a] + pseudocount
#             med_b = group_medians[grp_b] + pseudocount
#             lfc_df[col_name] = np.log2(med_a / med_b)

#         normalized = lfc_df
#         log.info(f"LFC normalization complete: {normalized.shape[0]} features x {normalized.shape[1]} contrasts")

#     else:
#         # Sample-wise scaling methods
#         df = data.copy().astype(float)

#         if log2 and method in {"vst", "zscore", "modified_zscore"}:
#             df = np.log2(df + 1)

#         if method == "vst":
#             # Variance-stabilizing transformation: arcsinh(sqrt(x)) then z-score per sample
#             df = np.arcsinh(np.sqrt(df.clip(lower=0)))
#             normalized = df.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col, axis=0)

#         elif method == "zscore":
#             # Z-score per sample (column-wise)
#             normalized = df.apply(lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col, axis=0)

#         elif method == "modified_zscore":
#             # Modified z-score (median-based) per sample
#             def _modified_zscore(col):
#                 med = col.median()
#                 mad = (col - med).abs().median()
#                 if mad > 0:
#                     return 0.6745 * (col - med) / mad
#                 return col - med
#             normalized = df.apply(_modified_zscore, axis=0)

#         elif method == "rank_normal":
#             # Rank-based inverse normal transformation per sample
#             arr = quantile_transform(df.values, n_quantiles=min(df.shape[0], 1000),
#                                      output_distribution='normal', random_state=0)
#             normalized = pd.DataFrame(arr, index=df.index, columns=df.columns)

#         log.info(f"{method} normalization complete: {normalized.shape[0]} features x {normalized.shape[1]} samples")

#     write_integration_file(normalized, output_dir, output_filename, indexing=True)
#     return normalized


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

    plt.xlabel('Abundance')
    if xlog:
        plt.xscale('log')
        plt.xlabel('Abundance (log)')

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

    plt.title(f'Histogram of Integrated Datasets')

    plt.legend()

    # Save the plot if output_dir is specified
    filename = f"distribution_of_integrated_datasets.pdf"
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





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from scipy.cluster.hierarchy import linkage
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm
import matplotlib.patheffects as pe
import pathlib

# ==========================================
# 1. DATA PIPELINE
# ==========================================

def prepare_universe(node_table, pathway_col, group_col, pe_cfg, missing_token='Unassigned'):
    df = node_table[[group_col, pathway_col]].copy()

    df[pathway_col] = df[pathway_col].fillna(missing_token).astype(str)
    df[pathway_col] = df[pathway_col].str.split(';')
    exploded = df.explode(pathway_col)
    exploded[pathway_col] = exploded[pathway_col].str.strip()

    exploded = exploded[
        (exploded[pathway_col].notna()) & (exploded[pathway_col] != "")
    ].rename(columns={pathway_col: "pathway", group_col: "group"})

    if pe_cfg.get('exclude_nopathway'):
        exploded = exploded[~exploded['pathway'].str.startswith("NOPATHWAY_")]

    is_missing = (exploded['pathway'] == missing_token).to_numpy()

    if pe_cfg.get('bipartite_only'):
        exploded['dtype'] = exploded.index.astype(str).str.split('_').str[0]
        total_dtypes = node_table.index.astype(str).str.split('_').str[0].nunique()
        real = exploded.loc[~is_missing]
        counts = real.groupby('pathway')['dtype'].nunique()
        keep_pathways = counts[counts >= total_dtypes].index
        keep_mask = exploded['pathway'].isin(keep_pathways).to_numpy() | is_missing
        exploded = exploded[keep_mask]
        is_missing = (exploded['pathway'] == missing_token).to_numpy()

    real = exploded.loc[~is_missing]
    p_counts = real['pathway'].value_counts()
    keep_pathways = p_counts[p_counts >= pe_cfg['min_features_per_pathway']].index
    keep_mask = exploded['pathway'].isin(keep_pathways).to_numpy() | is_missing
    exploded = exploded[keep_mask]

    node_dedup = node_table[~node_table.index.duplicated(keep='first')]
    g_counts = node_dedup[group_col].value_counts()
    keep_groups = g_counts[g_counts >= pe_cfg['min_features_per_group']].index
    exploded = exploded[exploded['group'].isin(keep_groups)]

    return exploded

def compute_enrichment_matrix(exploded, alpha=0.05, fdr_method='fdr_bh', missing_token='Unassigned'):
    # Full-universe background -- includes unannotated ('Unassigned') features,
    # so M/N reflect the true population, not just annotated features.
    M = exploded.index.nunique()
    N = exploded.index.to_series().groupby(exploded['group']).nunique()

    # Drop 'Unassigned' from what's actually TESTED -- see prior message.
    testable = exploded[exploded['pathway'] != missing_token]
    overlap_mat = pd.crosstab(testable['pathway'], testable['group'])
    overlap_mat = overlap_mat.reindex(columns=N.index, fill_value=0)

    n = testable.index.to_series().groupby(testable['pathway']).nunique().reindex(overlap_mat.index)
    N_aligned = N.reindex(overlap_mat.columns)

    pval_mat = np.array([[hypergeom.sf(k - 1, M, n[p], N_aligned[g]) for g, k in row.items()]
                        for p, row in overlap_mat.iterrows()])
    pval_df = pd.DataFrame(pval_mat, index=overlap_mat.index, columns=overlap_mat.columns)

    flat_reject, flat_padj, _, _ = multipletests(
        pval_df.values.ravel(), alpha=alpha, method=fdr_method
    )
    padj_df = pd.DataFrame(
        flat_padj.reshape(pval_df.shape), index=pval_df.index, columns=pval_df.columns
    )

    expected = np.outer(n.values, N_aligned.values) / M
    fold_df = pd.DataFrame(overlap_mat.values / expected, index=overlap_mat.index, columns=overlap_mat.columns)

    return overlap_mat, fold_df, pval_df, padj_df

def get_trend_values(quant_df, mapping_df, group_col, metadata_df, sort_by, collapse_by, agg='median'):
    if collapse_by and metadata_df is not None:
        meta = metadata_df.reindex(quant_df.columns)
        if meta[collapse_by].isna().all():
            raise ValueError(
                f"metadata_df.index has no overlap with quant_df.columns — "
                f"e.g. quant_df cols: {list(quant_df.columns[:3])} vs "
                f"metadata_df index: {list(metadata_df.index[:3])}"
            )
        collapsed_samples = quant_df.T.groupby(meta[collapse_by]).agg(agg).T

        if sort_by and sort_by in metadata_df.columns:
            sort_order = metadata_df.groupby(collapse_by)[sort_by].sort_values().index
            sort_order = [c for c in sort_order if c in collapsed_samples.columns]
            collapsed_samples = collapsed_samples.reindex(columns=sort_order)
    else:
        collapsed_samples = quant_df

    labels = mapping_df[group_col].unique()
    final_values = []
    for label in labels:
        feats = mapping_df[mapping_df[group_col] == label].index.unique()
        valid_feats = [f for f in feats if f in collapsed_samples.index]
        val = collapsed_samples.loc[valid_feats].agg(agg, axis=0) if valid_feats else pd.Series(np.nan, index=collapsed_samples.columns)
        final_values.append(val)

    return pd.DataFrame(final_values, index=labels)

# ==========================================
# 2. VISUAL ENGINE
# ==========================================

def _compute_fixed_layout(n_rows, n_cols, row_dtype_cnt, col_dtype_cnt, row_trends, col_trends,
                           square_cells=False):
    
    DENDRO_W_IN = 1.5            # row dendrogram width / col dendrogram height
    COMPOSITION_UNIT_IN = 0.18   # width/height per dtype column in the composition strip
    TREND_UNIT_IN = 0.25         # width/height per metadata category in a trend track
    TREND_PAD_IN = 0.25          # extra room for trend-track title/labels
    HEATMAP_PER_COL_IN = 0.1
    HEATMAP_PER_ROW_IN = 0.1
    MIN_HEATMAP_W_IN = 1.0
    MIN_HEATMAP_H_IN = 6.0
    LABEL_MARGIN_W_IN = 3.0      # room for heatmap y-tick labels + cbar
    LABEL_MARGIN_H_IN = 2.0      # room for heatmap x-tick labels + title
    
    n_row_dtypes = max(row_dtype_cnt.shape[1], 1) if row_dtype_cnt is not None else 1
    n_col_dtypes = max(col_dtype_cnt.shape[1], 1) if col_dtype_cnt is not None else 1
    n_row_trend_cats = row_trends.shape[1] if row_trends is not None and not row_trends.empty else 0
    n_col_trend_cats = col_trends.shape[1] if col_trends is not None and not col_trends.empty else 0

    heatmap_h_in = max(MIN_HEATMAP_H_IN, n_rows * HEATMAP_PER_ROW_IN)
    if square_cells:
        # Force each cell to be square: derive width from height-per-row
        cell_in = heatmap_h_in / n_rows
        heatmap_w_in = cell_in * n_cols
    else:
        heatmap_w_in = max(MIN_HEATMAP_W_IN, n_cols * HEATMAP_PER_COL_IN)

    dendro_w_in = DENDRO_W_IN
    dendro_h_in = DENDRO_W_IN
    colors_w_in = COMPOSITION_UNIT_IN * n_row_dtypes
    colors_h_in = COMPOSITION_UNIT_IN * n_col_dtypes
    row_trend_w_in = (TREND_UNIT_IN * n_row_trend_cats + TREND_PAD_IN) if n_row_trend_cats else 0.0
    col_trend_h_in = (TREND_UNIT_IN * n_col_trend_cats + TREND_PAD_IN) if n_col_trend_cats else 0.0

    fig_w_in = dendro_w_in + colors_w_in + heatmap_w_in + row_trend_w_in + LABEL_MARGIN_W_IN
    fig_h_in = dendro_h_in + colors_h_in + heatmap_h_in + col_trend_h_in + LABEL_MARGIN_H_IN

    return {
        "figsize": (fig_w_in, fig_h_in),
        "dendrogram_ratio": (dendro_w_in / fig_w_in, dendro_h_in / fig_h_in),
        "colors_ratio": (colors_w_in / fig_w_in, colors_h_in / fig_h_in),
        "row_trend_frac_w": row_trend_w_in / fig_w_in,
        "col_trend_frac_h": col_trend_h_in / fig_h_in,
    }

def _get_dtype_composition(exploded, id_col, id_order):
    df = exploded.copy()
    df['dtype'] = df.index.astype(str).str.split('_').str[0]

    feat_col = exploded.index.name or "feature_id"
    df = df.reset_index(names=feat_col)

    counts = df.groupby([id_col, 'dtype'])[feat_col].nunique().unstack(fill_value=0)
    counts = counts.reindex(index=id_order, fill_value=0)

    prefixes = sorted(counts.columns)
    cmap = plt.get_cmap("Greys")
    colors, norms = pd.DataFrame(index=counts.index), pd.DataFrame(index=counts.index)
    for dtype in prefixes:
        vals = counts[dtype].values.astype(float)
        vmax = vals.max() if vals.max() > 0 else 1.0
        normed = vals / vmax
        colors[dtype] = [cm.colors.to_hex(cmap(v)) if v > 0 else "#ffffff" for v in normed]
        norms[dtype] = normed
    return colors, counts, norms

def _annotate_dtype_counts(g, counts_df, norm_df, order, axis='row'):
    ax = g.ax_row_colors if axis == 'row' else g.ax_col_colors
    if ax is None: return
    dtypes = list(counts_df.columns)
    for pos, id_ in enumerate(order):
        if id_ not in counts_df.index: continue
        for level, dtype in enumerate(dtypes):
            val = int(counts_df.loc[id_, dtype])
            if val == 0: continue
            txt_color = "white" if norm_df.loc[id_, dtype] > 0.5 else "black"
            x, y = (level, pos) if axis == 'row' else (pos, level)
            ax.text(x + 0.5, y + 0.5, str(val), ha="center", va="center", fontsize=6, color=txt_color)

    # Dtype column-name ticks centred on each cell — seaborn uses pcolormesh so
    # cell centres are at 0.5, 1.5, ... in data coordinates.
    if axis == 'row':
        ax.set_xticks(np.arange(len(dtypes)) + 0.5)
        ax.set_xticklabels(dtypes, rotation=90, fontsize=5)
        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.set_ylabel("")
    else:
        ax.set_yticks(np.arange(len(dtypes)) + 0.5)
        ax.set_yticklabels(dtypes, fontsize=5)
        ax.set_xticks([])
        ax.set_xlabel("")

def _measure_ticklabel_right_edge(ax):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [t.get_window_extent(renderer=renderer) for t in ax.get_yticklabels() if t.get_text()]
    if not extents: return ax.get_position().x1
    return fig.transFigure.inverted().transform((max(e.x1 for e in extents), 0))[0]

def _measure_ticklabel_bottom_edge(ax):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = [t.get_window_extent(renderer=renderer) for t in ax.get_xticklabels() if t.get_text()]
    if not extents: return ax.get_position().y0
    return fig.transFigure.inverted().transform((0, min(e.y0 for e in extents)))[1]

def _values_to_rgba_zscored(value_mat, cmap='RdBu_r', vmax=None):
    """Convert a value matrix to RGBA using a symmetric diverging norm centred at 0.

    vmax : if provided, use this as the colour scale maximum (allows a shared
           scale across multiple calls). If None, derived from the data.
    """
    if value_mat.size == 0 or np.all(np.isnan(value_mat)):
        return np.full((*value_mat.shape, 4), cm.get_cmap(cmap)(0.5))
    if vmax is None:
        vmax = max(np.nanmax(np.abs(value_mat)), 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    rgba = cm.get_cmap(cmap)(norm(np.nan_to_num(value_mat, nan=0.0)))
    rgba[np.isnan(value_mat), 3] = 0.0
    return rgba

def _overlay_significance(g, pval_df, row_order, col_order, alpha_stars={0.001: "***", 0.01: "**", 0.05: "*"}):
    ax = g.ax_heatmap
    stroke = [pe.withStroke(linewidth=1.5, foreground="black")]
    for i, r_lab in enumerate(row_order):
        for j, c_lab in enumerate(col_order):
            try:
                val = pval_df.loc[r_lab, c_lab]
            except KeyError:
                continue
            for thresh, stars in sorted(alpha_stars.items()):
                if val < thresh:
                    ax.text(j + 0.5, i + 0.7, stars, ha="center", va="center",
                            color="white", fontsize=8, fontweight="bold", path_effects=stroke)
                    break

def _add_trend_track(g, rgba, value_mat, category_labels, tick_labels,
                     axis='row', size_frac=0.1, gap_frac=0.005):
    """
    Place a trend-heatmap track directly adjacent to the main heatmap.

    gap_frac : small figure-fraction gap between the heatmap edge and the track
               (prevents tick labels from overlapping the heatmap border).

    For axis='row':
      - track sits to the right of the heatmap (+ gap)
      - category labels (metadata columns) go on top as rotated xticklabels
      - group-name tick labels appear on the RIGHT side of this track
    For axis='col':
      - track sits below the heatmap (+ gap)
      - category labels (metadata rows) go on the left as yticklabels
      - pathway-name tick labels appear on the BOTTOM of this track
    """
    fig = g.figure
    heat_pos = g.ax_heatmap.get_position()

    if axis == 'row':
        n_rows, n_cats = rgba.shape[0], rgba.shape[1]

        ax = fig.add_axes([heat_pos.x1 + gap_frac, heat_pos.y0, size_frac, heat_pos.height])
        ax.imshow(rgba, aspect='auto',
                  extent=[-0.5, n_cats - 0.5, n_rows - 0.5, -0.5])

        # Category labels on top — centred on each column
        ax.set_xticks(np.arange(n_cats))
        ax.set_xticklabels(category_labels, rotation=90, fontsize=5, ha="center")
        ax.xaxis.tick_top()
        ax.set_title("")   # no label

        # Group-name tick labels on the RIGHT — centred on each row
        ax.set_yticks(np.arange(n_rows))
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_yticklabels(tick_labels, fontsize=7, ha='left')
        ax.set_ylim(n_rows - 0.5, -0.5)   # top-to-bottom, matching imshow

        # 3 sig-fig cell annotations
        for r in range(value_mat.shape[0]):
            for c in range(value_mat.shape[1]):
                v = value_mat[r, c]
                if not np.isnan(v):
                    lum = 0.299*rgba[r,c,0] + 0.587*rgba[r,c,1] + 0.114*rgba[r,c,2]
                    txt_col = 'black' if lum > 0.5 else 'white'
                    ax.text(c, r, f"{v:.3g}", ha='center', va='center',
                            fontsize=4, color=txt_col)

    else:
        n_cats, n_cols = rgba.shape[0], rgba.shape[1]

        ax = fig.add_axes([heat_pos.x0, heat_pos.y0 - gap_frac - size_frac,
                           heat_pos.width, size_frac])
        ax.imshow(rgba, aspect='auto',
                  extent=[-0.5, n_cols - 0.5, n_cats - 0.5, -0.5])

        # Category labels on the left — centred on each row
        ax.set_yticks(np.arange(n_cats))
        ax.set_yticklabels(category_labels, fontsize=5)
        ax.set_ylabel("")   # no label
        ax.set_ylim(n_cats - 0.5, -0.5)

        # Pathway-name tick labels on the BOTTOM — centred on each column
        ax.set_xticks(np.arange(n_cols))
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7,
                           ha='right', rotation_mode='anchor')
        ax.set_xlim(-0.5, n_cols - 0.5)

        # 3 sig-fig cell annotations
        for r in range(value_mat.shape[0]):
            for c in range(value_mat.shape[1]):
                v = value_mat[r, c]
                if not np.isnan(v):
                    lum = 0.299*rgba[r,c,0] + 0.587*rgba[r,c,1] + 0.114*rgba[r,c,2]
                    txt_col = 'black' if lum > 0.5 else 'white'
                    ax.text(c, r, f"{v:.3g}", ha='center', va='center',
                            fontsize=4, color=txt_col)

    for spine in ax.spines.values(): spine.set_visible(False)

def plot_group_pathway_heatmaps(counts_df, row_normalized_df, enrichment_matrix_df, pval_df,
                                row_trends, col_trends, row_dtype_cols, row_dtype_cnt, row_dtype_norm,
                                col_dtype_cols, col_dtype_cnt, col_dtype_norm,
                                trend_cmap='RdBu_r', output_dir=None, **kwargs):

    n_rows, n_cols = len(row_normalized_df), len(row_normalized_df.columns)
    square_cells = kwargs.get('square_cells', False)
    layout = _compute_fixed_layout(n_rows, n_cols, row_dtype_cnt, col_dtype_cnt, row_trends, col_trends,
                                   square_cells=square_cells)

    row_link = linkage(row_normalized_df.values, method='average')
    col_link = linkage(row_normalized_df.values.T, method='average')

    has_row_trend = row_trends is not None and not row_trends.empty
    has_col_trend = col_trends is not None and not col_trends.empty
    n_row_trend_cats = row_trends.shape[1] if has_row_trend else 0
    n_col_trend_cats = col_trends.shape[1] if has_col_trend else 0

    trend_gap = kwargs.get('trend_gap', 0.025)   # configurable gap (figure-fraction)

    common = dict(
        row_linkage=row_link, col_linkage=col_link, row_cluster=True, col_cluster=True,
        figsize=layout["figsize"],
        dendrogram_ratio=layout["dendrogram_ratio"],
        colors_ratio=layout["colors_ratio"],
        # Colorbar: top-left, aligned with the figure title
        cbar_pos=(0.02, 0.88, 0.015, 0.08),
    )

    panels = [("counts", counts_df, "viridis", "Feature Count"),
              ("normalized", row_normalized_df, "magma", "Fraction of Pathway"),
              ("enrichment", enrichment_matrix_df, "RdBu_r", "Fold Enrichment")]

    figs = {}
    for name, df, cmap, title in panels:
        g = sns.clustermap(df, cmap=cmap, row_colors=row_dtype_cols, col_colors=col_dtype_cols, **common)

        curr_row_order = df.index[g.dendrogram_row.reordered_ind].tolist()
        curr_col_order = df.columns[g.dendrogram_col.reordered_ind].tolist()

        # Suppress heatmap tick labels that will be re-drawn on the trend tracks,
        # and remove the axis labels ("group" / "pathway").
        ax_hm = g.ax_heatmap
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel("")
        if has_row_trend:
            # y-tick labels (group names) will appear on the right of the row trend track
            ax_hm.set_yticks(ax_hm.get_yticks())
            ax_hm.set_yticklabels([])
        else:
            ax_hm.tick_params(axis='y', labelsize=7)
        if has_col_trend:
            # x-tick labels (pathway names) will appear on the bottom of the col trend track
            ax_hm.set_xticks(ax_hm.get_xticks())
            ax_hm.set_xticklabels([])
        else:
            ax_hm.tick_params(axis='x', labelsize=7)

        if name == "enrichment":
            _overlay_significance(g, pval_df, curr_row_order, curr_col_order)
            g.figure.suptitle(
                title + "\n* padj<0.05  ** padj<0.01  *** padj<0.001 (hypergeometric, BH-FDR)",
                y=1.02)
        else:
            g.figure.suptitle(title, y=1.02)

        _annotate_dtype_counts(g, row_dtype_cnt, row_dtype_norm, curr_row_order, axis='row')
        _annotate_dtype_counts(g, col_dtype_cnt, col_dtype_norm, curr_col_order, axis='col')

        # ── Trend track sizing ────────────────────────────────────────────────
        # Row trend track: each column is just wide enough to fit the widest
        #   3-sig-fig annotation text (measured via a temporary Text artist).
        # Col trend track: each row is cell_h_px tall (matches heatmap row height).
        if has_row_trend or has_col_trend:
            fig = g.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            heat_pos = ax_hm.get_position()
            fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi
            heat_w_px = heat_pos.width  * fig_w_px
            heat_h_px = heat_pos.height * fig_h_px

            cell_h_px = heat_h_px / n_rows   # height of one heatmap row

            if has_row_trend:
                # Find the widest 3-sig-fig label across all row-trend values
                row_vals = row_trends.values.ravel()
                row_vals = row_vals[~np.isnan(row_vals)]
                widest_str = max(
                    (f"{v:.3g}" for v in row_vals),
                    key=len,
                    default="0.0"
                )
                # Measure pixel width of that string at fontsize=4
                tmp = fig.text(0, 0, widest_str, fontsize=4, visible=False)
                fig.canvas.draw()
                txt_w_px = tmp.get_window_extent(renderer=renderer).width
                tmp.remove()
                TEXT_PAD_PX = 12   # padding on each side of the text
                col_cell_w_px = txt_w_px + 2 * TEXT_PAD_PX
                row_trend_frac_w = (col_cell_w_px * n_row_trend_cats) / fig_w_px
            else:
                row_trend_frac_w = 0.0

            # Col trend: n_col_trend_cats rows, each cell_h_px tall
            col_trend_frac_h = (cell_h_px * n_col_trend_cats) / fig_h_px if has_col_trend else 0.0
        else:
            row_trend_frac_w = layout["row_trend_frac_w"]
            col_trend_frac_h = layout["col_trend_frac_h"]

        # Each trend track is colour-scaled independently from its own data.
        if has_row_trend:
            row_trend_aligned = row_trends.reindex(curr_row_order)
            rgba = _values_to_rgba_zscored(row_trend_aligned.values, cmap=trend_cmap)
            _add_trend_track(g, rgba, value_mat=row_trend_aligned.values,
                              category_labels=row_trend_aligned.columns,
                              tick_labels=curr_row_order,
                              axis='row', size_frac=row_trend_frac_w,
                              gap_frac=trend_gap)

        if has_col_trend:
            col_trend_aligned = col_trends.reindex(curr_col_order)
            # rgba shape: (n_cats, n_pathways, 4)  — transpose values for imshow
            rgba = _values_to_rgba_zscored(col_trend_aligned.values.T, cmap=trend_cmap)
            _add_trend_track(g, rgba, value_mat=col_trend_aligned.values.T,
                              category_labels=col_trend_aligned.columns,
                              tick_labels=curr_col_order,
                              axis='col', size_frac=col_trend_frac_h,
                              gap_frac=trend_gap)

        figs[name] = g.figure

    if output_dir:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(out / f"heatmap_{name}.png", bbox_inches='tight', dpi=300)
    return figs

# ==========================================
# 3. INTEGRATION
# ==========================================

def compare_groups_to_pathways(
    node_table: pd.DataFrame,
    quant_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str = None,
    pe: dict = None,
) -> dict:

    exploded = prepare_universe(node_table, pe['pathway_col'], 'group', pe)
    overlap_mat, fold_mat, pval_mat, padj_mat = compute_enrichment_matrix(
        exploded, alpha=pe['alpha'], fdr_method=pe.get('fdr_method', 'fdr_bh')
    )
    missing_token = pe.get('missing_token', 'Unassigned')

    # get selected_pathways from pe (as a comma-separated list of strings in quotes)
    selected_pathways = pe.get('selected_pathways', None)
    if selected_pathways:
        # User-specified subset takes priority over ranking entirely.
        available = set(padj_mat.index)
        requested = list(dict.fromkeys(selected_pathways))  # de-dup, preserve order
        missing = [p for p in requested if p not in available]
        if missing:
            print(f"[compare_groups_to_pathways] Warning: {len(missing)} requested "
                  f"pathway(s) not found after universe filtering (dropped by "
                  f"min_features_per_pathway/bipartite_only, or misspelled): {missing}")
        chosen_pathways = [p for p in requested if p in available]
        if not chosen_pathways:
            raise ValueError(
                "None of the requested `selected_pathways` are present in the "
                "tested/filtered pathway universe."
            )
        # Groups restricted to those with >=1 feature in the chosen pathways.
        kept_groups = sorted(
            exploded.loc[exploded['pathway'].isin(chosen_pathways), 'group'].unique()
        )
    else:
        if pe.get('rank_by') == 'summed_significance':
            rank_scores = -np.log10(padj_mat.clip(lower=1e-300)).sum(axis=1).sort_values(ascending=False)
        elif pe.get('rank_by') == 'feature_count':
            rank_scores = (
                exploded[exploded['pathway'] != missing_token]
                .groupby('pathway').size().sort_values(ascending=False)
            )
        else:
            raise ValueError(f"Unknown rank_by method: {pe.get('rank_by')}")
        chosen_pathways = rank_scores.head(pe['top_n']).index.tolist()
        kept_groups = sorted(exploded['group'].unique())

    counts_df = overlap_mat.T.loc[kept_groups, chosen_pathways]
    enrich_df = fold_mat.T.loc[kept_groups, chosen_pathways]
    padj_df   = padj_mat.T.loc[kept_groups, chosen_pathways]
    norm_df   = counts_df.div(counts_df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)

    row_dtype_cols, row_dtype_cnt, row_dtype_norm = _get_dtype_composition(exploded, 'group', kept_groups)
    col_dtype_cols, col_dtype_cnt, col_dtype_norm = _get_dtype_composition(exploded, 'pathway', chosen_pathways)

    row_trends = get_trend_values(quant_df, node_table, 'group', metadata_df, pe['trend_sort_by'], pe['trend_collapse_by'], pe['trend_agg'])
    col_trends = get_trend_values(quant_df, exploded, 'pathway', metadata_df, pe['trend_sort_by'], pe['trend_collapse_by'], pe['trend_agg'])

    figs = plot_group_pathway_heatmaps(
        counts_df=counts_df, row_normalized_df=norm_df, enrichment_matrix_df=enrich_df,
        pval_df=padj_df, row_trends=row_trends, col_trends=col_trends,
        row_dtype_cols=row_dtype_cols, row_dtype_cnt=row_dtype_cnt, row_dtype_norm=row_dtype_norm,
        col_dtype_cols=col_dtype_cols, col_dtype_cnt=col_dtype_cnt, col_dtype_norm=col_dtype_norm,
        trend_cmap=pe.get('trend_cmap', 'RdBu_r'), output_dir=output_dir
    )

    # ---- Export everything
    if output_dir:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        exploded.to_csv(out / "exploded_universe.csv")

        overlap_mat.to_csv(out / "full_overlap_matrix.csv")
        fold_mat.to_csv(out / "full_fold_enrichment_matrix.csv")
        pval_mat.to_csv(out / "full_pvalue_matrix.csv")
        padj_mat.to_csv(out / "full_padj_matrix.csv")

        row_trends.to_csv(out / "group_trend_values.csv")
        col_trends.to_csv(out / "pathway_trend_values.csv")

        with open(out / "run_config.json", "w") as f:
            json.dump({
                "pe": {k: v for k, v in pe.items()},
                "selected_pathways_requested": selected_pathways,
                "selected_pathways_used": chosen_pathways,
                "kept_groups": kept_groups,
            }, f, indent=2, default=str)

    return {"figs": figs, "padj_matrix": padj_df, "selected_pathways": chosen_pathways, "kept_groups": kept_groups}