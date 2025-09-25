# --- Standard library imports ---
import glob
import gzip
import importlib.util
import io
import itertools
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import yaml
from tqdm import tqdm

# --- Display and plotting ---
from IPython.display import display
import fitz  # PyMuPDF

# --- Typing ---
from typing import List

# --- Scientific computing & data analysis ---
import numpy as np
import pandas as pd
import scipy.stats as stats

# --- Plotting ---
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import viridis
from matplotlib.colors import to_hex

# --- Machine learning & statistics ---
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# --- MOFA2 & related ---
import mofax
from mofapy2.run.entry_point import entry_point

# --- Bioinformatics ---
import gff3_parser
from Bio import SeqIO

# --- PDF and Excel handling ---
import fitz
from openpyxl import load_workbook
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# --- Network analysis ---
import networkx as nx
from community import community_louvain

# --- Cheminformatics ---
from rdkit import Chem

import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None

# ====================================
# Helper functions for various tasks
# ====================================

def clear_directory(dir_path: str) -> None:
    # Wipe out all contents of dir_path if generating new outputs
    if os.path.exists(dir_path):
        #print(f"Clearing existing contents of directory: {dir_path}")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')  

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
        print(f"\tData saved to {fname}\n")
    else:
        print("Not saving data to disk.")

# ====================================
# Analysis step functions
# ====================================

def glm_for_differential_abundance(
    count_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    category_column: str,
    reference_group: str,
    output_dir: str,
    threshold: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform differential abundance analysis using Generalized Linear Models (GLM) with a Gaussian family.

    Args:
        count_df (pd.DataFrame): Features x samples count matrix.
        metadata_df (pd.DataFrame): Samples x metadata DataFrame.
        category_column (str): Metadata column for group comparison.
        reference_group (str): Reference group for GLM.
        output_dir (str): Output directory for results.
        threshold (float): Adjusted p-value threshold.

    Returns:
        tuple: (coefficients_df, corrected_p_values_df) DataFrames.
    """

    # Check if outputs already exist and skip if they do
    coefficient_file = os.path.join(output_dir, f"glm_coefficients_{reference_group}_against_{category_column}")
    pvalue_file = os.path.join(output_dir, f"glm_corrected_pvalues_{reference_group}_against_{category_column}")
    if os.path.exists(f"{coefficient_file}.csv") and os.path.exists(f"{pvalue_file}.csv"):
        print(f"GLM results for {category_column} already exist. Skipping analysis and returning existing data.\n")
        return pd.read_csv(f"{coefficient_file}.csv", index_col=0), pd.read_csv(f"{pvalue_file}.csv", index_col=0)

    # Transpose the count DataFrame to have samples as rows and features as columns
    count_df_transposed = count_df.T

    # Merge count DataFrame and metadata on sample identifiers
    linked_data = count_df_transposed.join(metadata_df)

    # Check for NaNs and infinite values in the linked data
    if linked_data.isnull().values.any():
        raise ValueError("NaN values detected in the linked data. Please clean the data.")

    # Ensure the reference group exists in the category column
    unique_categories = metadata_df[category_column].unique()
    if reference_group not in unique_categories:
        raise ValueError(f"Reference group '{reference_group}' not found in the '{category_column}' column.")

    # Set the reference group for the categorical variable
    linked_data[category_column] = pd.Categorical(
        linked_data[category_column],
        categories=[reference_group] + [cat for cat in unique_categories if cat != reference_group],
        ordered=True
    )

    print(f"Running GLM on all features against {category_column} with reference group '{reference_group}'...")
    results = {}
    coefficients = {}

    # Loop through each feature
    for feature in count_df.index:
        # Create a formula for the GLM
        formula = f'Q("{feature}") ~ C({category_column})'
        # Fit the GLM with a Gaussian family
        try:
            model = smf.glm(formula=formula, data=linked_data, family=sm.families.Gaussian())
            result = model.fit()
            results[feature] = result
            coefficients[feature] = result.params.to_dict()
        except ValueError as e:
            # Handle cases where the model fitting fails
            print(f"Model fitting failed for feature {feature}: {e}")
            results[feature] = None
            coefficients[feature] = None

    # Extract and correct p-values
    print(f"\tExtracting and correcting p-values for {category_column} model")
    all_p_values_dict = {}
    for feature, result in results.items():
        if result is not None:
            all_p_values_dict[feature] = result.pvalues.to_dict()

    all_p_values_list = []
    feature_keys = []
    term_keys = []
    for feature, terms in all_p_values_dict.items():
        for term, p_value in terms.items():
            all_p_values_list.append(p_value)
            feature_keys.append(feature)
            term_keys.append(term)
    corrected_p_values = multipletests(all_p_values_list, alpha=threshold, method='fdr_bh')[1]

    corrected_p_values_dict = {}
    for feature, term, corrected_p_value in zip(feature_keys, term_keys, corrected_p_values):
        if feature not in corrected_p_values_dict:
            corrected_p_values_dict[feature] = {}
        corrected_p_values_dict[feature][term] = float(corrected_p_value)

    # Create DataFrames for coefficients and corrected p-values
    coefficients_df = pd.DataFrame.from_dict(coefficients, orient='index')
    coefficients_df['reference'] = reference_group
    corrected_p_values_df = pd.DataFrame.from_dict(corrected_p_values_dict, orient='index')
    corrected_p_values_df['reference'] = reference_group
    
    # Save results to output directory
    print(f"\tExporting results for {category_column} model")
    write_integration_file(coefficients_df, output_dir, os.path.basename(coefficient_file))
    write_integration_file(corrected_p_values_df, output_dir, os.path.basename(pvalue_file))

    return coefficients_df, corrected_p_values_df

def run_kruskalwallis_test(
    data: pd.DataFrame,
    integrated_metadata: pd.DataFrame,
    attribute: str = None,
    significance_level: float = 0.05,
    lfc_level: float = 0.5,
    test_type: str = 'kruskalwallis',
    output_dir: str = None,
    stats_results_filename: str = None
) -> pd.DataFrame:
    """
    Subset features based on statistical significance and log fold change using a specified test.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        integrated_metadata (pd.DataFrame): Metadata DataFrame with sample information.
        attribute (str): Metadata column to test for group differences.
        significance_level (float): FDR-corrected p-value cutoff.
        lfc_level (float): Minimum absolute log2 fold change to consider significant.
        test_type (str): Statistical test to use ('kruskalwallis').
        output_dir (str): Directory to write results.
        stats_results_filename (str): Filename for summary statistics.

    Returns:
        pd.DataFrame: Subset of features passing significance and LFC thresholds.
    """
    if attribute not in integrated_metadata.columns:
        raise ValueError(f"Attribute '{attribute}' not found in metadata")
    
    # Calculate test statistic and p-value for each feature
    data = data.fillna(0)
    integrated_metadata_subset = integrated_metadata[integrated_metadata.index.isin(data.columns)]
    groups = integrated_metadata_subset[attribute].unique()
    p_values = []
    log_fold_changes = pd.DataFrame(index=data.index)
    for feature in data.index:
        samples = [data.loc[feature, integrated_metadata_subset[integrated_metadata_subset[attribute] == group].index].values for group in groups]
        if test_type == 'kruskalwallis':
            if all(np.all(sample == sample[0]) for sample in samples):
                p_val = 1.0  # Assign a non-significant p-value if all samples are identical
            else:
                _, p_val = stats.kruskal(*samples)
        else:
            raise ValueError("Invalid test_type")
        p_values.append(p_val)
        # Calculate log fold change between each pair of groups for each feature
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                mean_i = np.mean(samples[i])
                mean_j = np.mean(samples[j])
                log_fold_change = np.log2(mean_i / mean_j)
                column_name = f"LFC_{groups[i]}_vs_{groups[j]}"
                log_fold_changes.loc[feature, column_name] = log_fold_change

    corrected_p_values = multipletests(p_values, alpha=significance_level, method='fdr_bh')[1]
    
    # Create the final dataframe with feature name, p value, corrected p value, and log-fold-change columns
    results_df = pd.DataFrame(index=data.index)
    results_df['p_value'] = p_values
    results_df['corrected_p_value'] = corrected_p_values
    results_df = results_df.join(log_fold_changes)

    # Subset the dataframe by significant features and log-fold-change
    significant_features = data[(results_df['corrected_p_value'] <= significance_level) & (results_df.iloc[:, 2:].abs() >= lfc_level).any(axis=1)]
    significant_features['padj'] = results_df.loc[significant_features.index, 'corrected_p_value']
    print(f"Found {significant_features.shape[0]} significant features (out of {data.shape[0]}) based on '{attribute}' attribute using {test_type} test. Keeping only these features.\n")

    print("Writing summary stats table...")
    write_integration_file(data=results_df, output_dir=output_dir, filename=stats_results_filename, indexing=True)

    return significant_features

def perform_feature_selection(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    output_filename: str = "feature_selection_table.csv",
    subset_method: str = "variance",
    significance_level: float = 0.05,
    lfc_level: float = 0.5,
    category_column: str = None,
    reference_group: str = None,
    output_dir: str = None,
    feature_cutoff: int = 5000,
    overwrite: bool = False,
    feature_list_file: str = None
) -> pd.DataFrame:
    """
    Subset features prior to network analysis using various methods.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        metadata (pd.DataFrame): Metadata DataFrame.
        output_filename (str): Filename for the feature selection table.
        subset_method (str): Method for subsetting ('variance', 'glm', 'kruskalwallis', 'feature_list', 'none').
        significance_level (float): FDR-corrected p-value cutoff.
        lfc_level (float): Minimum absolute log2 fold change.
        category_column (str): Metadata column for group comparison.
        reference_group (str): Reference group for GLM.
        output_dir (str): Directory to write results.
        feature_cutoff (int): Maximum number of features to retain.
        overwrite (bool): Whether to overwrite existing results.
        feature_list_file (str): Path to file with list of features to keep.

    Returns:
        pd.DataFrame: Subsetted feature matrix.
    """

    # Always use this filename for the output table
    selection_table_path = f"{output_dir}/{output_filename}"

    if os.path.exists(selection_table_path) and not overwrite:
        print(f"Feature selection table already exists: {selection_table_path}. \nReturning and skipping calculation...")
        return pd.read_csv(selection_table_path, index_col=0)

    if subset_method == "none":
        print(f"\nNot subsetting dataset by any statistical methods and printing/returning data...\n")
        write_integration_file(data=data, output_dir=output_dir, filename=output_filename, indexing=True)
        return data

    elif subset_method == "glm":
        print(f"\nRunning GLM model on {data.shape[0]} features in integrated dataset to find significant features across samples...\n")
        lfc_table, pval_table = glm_for_differential_abundance(data, metadata, category_column, reference_group, output_dir, threshold=significance_level)
        summary_stats_table = pd.concat([lfc_table, pval_table], axis=1)
        summary_stats_table.index.name = "features"
        pval_table['psum'] = pval_table.iloc[:, 1:-1].sum(axis=1)
        pval_table.sort_values('psum', inplace=True)
        pval_table.drop(columns=['psum'], inplace=True)
        sig_indexes_pval = (pval_table.iloc[:, 1:-1] <= significance_level).any(axis=1)
        sig_indexes_lfc = (lfc_table.iloc[:, 1:-1] >= lfc_level).any(axis=1)
        sig_indexes = sig_indexes_pval & sig_indexes_lfc
        combined_df_subset = data[sig_indexes]
        combined_df_subset = combined_df_subset.reindex(pval_table.index).dropna()
        if combined_df_subset.shape[0] == 0:
            print(f"No features were found after running GLM model.")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
            return None
        else:
            print(f"{combined_df_subset.shape[0]} features with GLM model p-value<{significance_level} and LFC>{lfc_level} were selected for network analysis.\n")
            if combined_df_subset.shape[0] > feature_cutoff:
                print(f"\tNote! More than {feature_cutoff} features were output from GLM model. Need to subset to top {feature_cutoff} for network analysis.\n")
                combined_df_subset = combined_df_subset.iloc[:feature_cutoff]
            print("Writing significant features table...")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
        return combined_df_subset

    elif subset_method == "kruskalwallis":
        print(f"\nRunning {subset_method} test on {data.shape[0]} features in integrated dataset to subset by those significantly associated with '{category_column}' variable...\n")
        combined_df_subset = run_kruskalwallis_test(data=data, integrated_metadata=metadata, attribute=category_column, lfc_level=lfc_level, \
                                                    test_type=subset_method, significance_level=significance_level, output_dir=output_dir,
                                                    stats_results_filename=f"{subset_method}_summary_stats_on_{category_column}")
        if combined_df_subset is not None and 'padj' in combined_df_subset.columns:
            combined_df_subset = combined_df_subset.sort_values(by='padj').drop(columns=['padj'])
        if combined_df_subset is None or combined_df_subset.shape[0] == 0:
            print(f"No features were found to be significantly associated with '{category_column}' group using a {subset_method} test at alpha={significance_level} and lfc={lfc_level}.\n")
            write_integration_file(data=pd.DataFrame(), output_dir=output_dir, filename=output_filename, indexing=True)
            return None
        else:
            print(f"""{combined_df_subset.shape[0]} features are significantly (alpha={significance_level}) associated with '{category_column}' variable.""")
            if combined_df_subset.shape[0] > feature_cutoff:
                print(f"\tNote! More features than allowed by feature cutoff were identified. Need to subset to top {feature_cutoff} for network analysis.\n")
                combined_df_subset = combined_df_subset.iloc[:feature_cutoff]
            print("Writing significant features table...")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
        return combined_df_subset

    elif subset_method == "variance":
        print(f"\nRunning variance test on {data.shape[0]} features in integrated dataset to subset by those with highest variance across samples...\n")
        combined_df_subset = data.loc[data.var(axis=1).sort_values(ascending=False).head(feature_cutoff).index]
        if combined_df_subset.shape[0] == 0:
            print(f"""No features were found after {subset_method} sorting.""")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
            return None
        else:
            print(f"""{combined_df_subset.shape[0]} features with the highest variance were selected for network analysis.""")
            print("\nWriting significant features table...")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
        return combined_df_subset

    elif subset_method == "feature_list":
        if os.path.exists(os.path.join(output_dir, feature_list_file)) is False:
            print(f"Feature list file does not exist: {feature_list_file}. Please provide a valid file path.")
            write_integration_file(data=pd.DataFrame(), output_dir=output_dir, filename=output_filename, indexing=True)
            return None
        print(f"\nSubsetting {data.shape[0]} features in integrated dataset by a specific feature list...\n")
        feature_list = pd.read_csv(os.path.join(output_dir, feature_list_file), header=None).iloc[:, 0].tolist()
        combined_df_subset = data.loc[data.index.isin(feature_list)]
        if combined_df_subset.shape[0] == 0:
            print(f"No features from feature list were found in data")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
            return None
        else:
            print(f"{combined_df_subset.shape[0]} features from the input list were selected for network analysis.")
            print("Writing selected features table...")
            write_integration_file(data=combined_df_subset, output_dir=output_dir, filename=output_filename, indexing=True)
        return combined_df_subset

    else:
        print("Please select a valid subsetting method: 'none', 'glm', 'kruskalwallis', 'variance', or 'feature_list'")
        write_integration_file(data=pd.DataFrame(), output_dir=output_dir, filename=output_filename, indexing=True)
        return None

def calculate_correlated_features(
    data: pd.DataFrame,
    output_filename: str,
    output_dir: str,
    corr_method: str = "pearson",
    corr_cutoff: float = 0.5,
    overwrite: bool = False,
    keep_negative: bool = False,
    only_bipartite: bool = False,
    save_corr_matrix: bool = False
) -> pd.DataFrame:
    """
    Memory-efficient calculation and filtering of feature-feature correlation matrix.
    Only stores pairs above cutoff in memory.

    Args:
        data (pd.DataFrame): Feature matrix (features x samples).
        output_dir (str): Output directory for results.
        corr_method (str): Correlation method.
        corr_cutoff (float): Correlation threshold.
        overwrite (bool): Overwrite existing results.
        keep_negative (bool): Keep negative correlations if True.
        only_bipartite (bool): Only keep bipartite correlations.
        save_corr_matrix (bool): Save correlation matrix to disk.

    Returns:
        pd.DataFrame: Filtered correlation matrix.
    """

    output_subdir = output_dir
    os.makedirs(output_subdir, exist_ok=True)
    existing_output = f"{output_subdir}/{output_filename}"
    if os.path.exists(existing_output) and not overwrite:
        print(f"Correlation table already exists at {existing_output}. Returning existing matrix.")
        return pd.read_csv(existing_output, sep=',')

    data_input = data.T
    features = data_input.columns.tolist()
    n = len(features)
    print(f"Calculating feature correlations for {n} features...")

    # Use numpy for speed and memory
    arr = data_input.values
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    arr_centered = arr - means

    # Only store pairs above cutoff
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = pd.Series(arr[:, i]).corr(pd.Series(arr[:, j]), method=corr_method)
            if keep_negative:
                if abs(corr) >= corr_cutoff and abs(corr) != 1:
                    results.append((features[i], features[j], corr))
            else:
                if corr >= corr_cutoff and corr != 1:
                    results.append((features[i], features[j], corr))

    print(f"\tFiltered to {len(results)} pairs above cutoff.")

    # Only bipartite
    if only_bipartite:
        print("Filtering for only bipartite correlations...")
        results = [r for r in results if r[0][:3] != r[1][:3]]
        print(f"\tCorrelation matrix filtered to {len(results)} pairs.")

    # Build DataFrame
    melted_corr = pd.DataFrame(results, columns=['feature_1', 'feature_2', 'correlation'])

    if save_corr_matrix:
        print(f"Saving feature correlation table to disk...")
        write_integration_file(melted_corr, output_subdir, output_filename, indexing=False)

    return melted_corr

def plot_correlation_network(
    correlation_matrix: pd.DataFrame,
    feature_prefixes: list,
    integrated_data: pd.DataFrame,
    integrated_metadata: pd.DataFrame,
    output_filenames: dict,
    annotation_df: str = None,
    network_mode: str = "bipartite",
    submodule_mode: str = "community",
    extract_submodules: bool = True,
    show_plot_in_notebook: bool = False,
    corr_cutoff: float = 0.5
) -> None:
    """
    Plot and export a correlation network from a correlation matrix, with optional submodule extraction.

    Args:
        correlation_matrix (pd.DataFrame): DataFrame with columns ['feature_1', 'feature_2', 'correlation'].
        feature_prefixes (list): List of feature prefixes (e.g., ['tx', 'mx']).
        integrated_data (pd.DataFrame): Feature matrix.
        integrated_metadata (pd.DataFrame): Metadata DataFrame.
        output_filenames (dict): Dictionary with keys 'graph', 'node_table', 'edge_table' for output filenames.
        annotation_df (pd.DataFrame): DataFrame with feature annotations.
        network_mode (str): 'bipartite' or 'full'.
        submodule_mode (str): 'community' or 'subgraphs'.
        extract_submodules (bool): Whether to extract submodules.
        show_plot_in_notebook (bool): Whether to display the plot.
        corr_cutoff (float): Correlation threshold for edges.

    Returns:
        None
    """

    # Create a correlation matrix
    if network_mode == "bipartite":
        print("Filtering correlation matrix to bipartite relationships...")
        correlation_matrix = correlation_matrix[~correlation_matrix.apply(lambda row: row['feature_1'][:3] == row['feature_2'][:3], axis=1)]
        
    print("Pivoting data...")
    correlation_df = correlation_matrix.pivot(index='feature_2', columns='feature_1', values='correlation')

    # Create a graph from the correlation matrix
    print("Creating graph...")
    G = nx.Graph()

    # Use vectorized operations to add nodes and edges based on the threshold
    print(f"Filtering edges based on correlation threshold of {corr_cutoff}...")
    mask = np.abs(correlation_df.values) >= corr_cutoff
    np.fill_diagonal(mask, False)  # Exclude self-loops
    edges = np.column_stack(np.where(mask))
    #print(f"\t{len(edges)} edges added to the graph.")
    if network_mode == "full":
        for i, j in edges:
            G.add_edge(correlation_df.index[i], correlation_df.columns[j], weight=correlation_df.iloc[i, j])
    elif network_mode == "bipartite":
        for i, j in edges:
            node_i = correlation_df.index[i]
            node_j = correlation_df.columns[j]
            if (any(prefix in node_i for prefix in feature_prefixes) and 
                any(prefix in node_j for prefix in feature_prefixes) and 
                node_i[:2] != node_j[:2]):  # Ensure nodes are from different groups
                G.add_edge(node_i, node_j, weight=float((correlation_df.iloc[i, j]*100)**2))
    else:
        raise ValueError("Invalid network_mode. Please select 'full' or 'bipartite'.")
    print(f"\t{G.number_of_edges()} edges added to the graph.")
    print(f"\t{G.number_of_nodes()} nodes added to the graph.")

    # Define node colors based on datatype
    print("Adding node aesthetics...")
    viridis_colors = viridis(np.linspace(0, 1, 3))  # Generate 3 distinct colors from viridis colormap
    viridis_colors_hex = [to_hex(color) for color in viridis_colors]  # Convert to hex color strings
    for node in G.nodes():
        if any(keyword in node for keyword in feature_prefixes):
            for i, keyword in enumerate(feature_prefixes):
                if keyword in node:
                    G.nodes[node]['datatype_color'] = viridis_colors_hex[i % len(viridis_colors_hex)]
                    G.nodes[node]['datatype_shape'] = "Round Rectangle" if keyword == "tx" else "Diamond"
        else:
            G.nodes[node]['datatype_color'] = "gray"
            G.nodes[node]['datatype_shape'] = "Rectangle" 

    # Annotate nodes and edges with functional information if available
    if annotation_df is not None and not annotation_df.empty:
        print("Annotating nodes and edges with functional information...")
        annotation_df['node_id'] = annotation_df.index
        node_annotations = annotation_df.set_index('node_id')['annotation'].fillna("Unassigned").to_dict()
        for node in G.nodes():
            if node in node_annotations:
                annotation_string = str(node_annotations[node])
                if annotation_string == "Unassigned":
                    node_shape = "Round Rectangle"
                else:
                    node_shape = "Diamond"
                G.nodes[node]['annotation'] = annotation_string
                G.nodes[node]['annotation_shape'] = node_shape
        for edge in G.edges():
            if edge[0] in node_annotations:
                G.edges[edge]['source_annotation'] = str(node_annotations[edge[0]])
            if edge[1] in node_annotations:
                G.edges[edge]['target_annotation'] = str(node_annotations[edge[1]])

    # Remove isolated groups of exactly 2 nodes
    small_components = [c for c in nx.connected_components(G) if len(c) < 3]
    for comp in small_components:
        G.remove_nodes_from(comp)    

    print(f"Exporting node/edge tables and correlation network...")
    nx.write_graphml(G, output_filenames['graph'])
    edge_table = nx.to_pandas_edgelist(G)
    edge_table.index.name = 'edge_index'
    node_table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    node_table.index.name = 'node_id'
    node_table.to_csv(output_filenames['node_table'], index=True, index_label='node_index')
    edge_table.to_csv(output_filenames['edge_table'], index=True, index_label='edge_index')

    print("Drawing graph...")
    if show_plot_in_notebook is True:
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        node_colors = [G.nodes[node]['datatype_color'] for node in G.nodes()]
        nx.draw(G, pos, with_labels=True, node_size=200, node_color=node_colors, font_size=1)
        plt.show()

    if extract_submodules is True and network_mode == "bipartite":
        print("Extracting submodules from the bipartite network...")
        G = extract_submodules_from_bipartite_network(graph=G, integrated_data=integrated_data, metadata=integrated_metadata,
                                                      output_filenames=output_filenames, submodule_mode=submodule_mode)
        print(f"Updating main graphml file and edge/node tables with submodule information...")
        nx.write_graphml(G, output_filenames['graph'])
        edge_table = nx.to_pandas_edgelist(G)
        edge_table.index.name = 'edge_index'
        node_table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
        node_table.index.name = 'node_id'
        node_table.to_csv(output_filenames['node_table'], index=True, index_label='node_index')
        edge_table.to_csv(output_filenames['edge_table'], index=True, index_label='edge_index')

def extract_submodules_from_bipartite_network(
    graph: 'nx.Graph',
    integrated_data: pd.DataFrame,
    metadata: pd.DataFrame,
    output_filenames: dict,
    submodule_mode: str,
    save_plots: bool = True,
) -> 'nx.Graph':
    """
    Extract submodules (connected components or communities) from a bipartite network and save results.

    Args:
        graph (nx.Graph): Bipartite network graph.
        integrated_data (pd.DataFrame): Feature matrix.
        metadata (pd.DataFrame): Metadata DataFrame.
        output_filenames (dict): Dictionary with keys 'graph', 'node_table', 'edge_table' for output filenames.
        submodule_mode (str): 'subgraphs' or 'community'.
        save_plots (bool): Whether to save abundance plots.

    Returns:
        nx.Graph: Annotated main graph with submodule information.
    """

    if submodule_mode == "subgraphs": # Find connected components in the graph
        submodules = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    elif submodule_mode == "community": # Detect communities in the graph and group nodes into submodules by their community
        partition = community_louvain.best_partition(graph)
        submodules = []
        for community_id in set(partition.values()):
            nodes_in_community = [node for node, community in partition.items() if community == community_id]
            submodules.append(graph.subgraph(nodes_in_community).copy())
    else:
        raise ValueError("Invalid submodule_mode. Please select 'subgraphs' or 'community'.")

    # Define module colors based on index text and add as node attribute
    submodule_num = len(submodules)
    viridis_colors = viridis(np.linspace(0, 1, submodule_num))  # Generate distinct colors from viridis colormap
    viridis_colors_hex = [to_hex(color) for color in viridis_colors]  # Convert to hex color strings
    random.shuffle(viridis_colors_hex)

    for idx, submodule in enumerate(submodules):

        # Annotate nodes in submodules and main graph with submodule IDs
        for node in submodule.nodes():
            if node in graph.nodes():
                graph.nodes[node]['submodule'] = f"submodule_{idx+1}"
                graph.nodes[node]['submodule_color'] = viridis_colors_hex[idx % len(viridis_colors_hex)]
            submodule.nodes[node]['submodule'] = f"submodule_{idx+1}"
            submodule.nodes[node]['submodule_color'] = viridis_colors_hex[idx % len(viridis_colors_hex)]

        # Save each submodule as a GraphML file
        graph_filename = os.path.basename(output_filenames['graph'])
        os.makedirs(output_filenames['submodule_path'], exist_ok=True)
        submodule_graph_name = f"{output_filenames['submodule_path']}/{graph_filename.replace('.graphml', '_submodule')}{idx+1}.graphml"
        nx.write_graphml(submodule, submodule_graph_name)
        print(f"\tSubmodule {idx + 1} saved to {submodule_graph_name}")

        # Write node and edge tables for each submodule
        node_table = pd.DataFrame.from_dict(dict(submodule.nodes(data=True)), orient='index')
        node_table.index.name = 'node_id'
        edge_table = nx.to_pandas_edgelist(submodule)
        edge_table.index.name = 'node_id'
        node_table_filename = os.path.basename(output_filenames['node_table'])
        edge_table_filename = os.path.basename(output_filenames['edge_table'])
        submodule_node_table_name = f"{output_filenames['submodule_path']}/{node_table_filename.replace('.csv', '_submodule')}{idx+1}.csv"
        submodule_edge_table_name = f"{output_filenames['submodule_path']}/{edge_table_filename.replace('.csv', '_submodule')}{idx+1}.csv"
        node_table.to_csv(submodule_node_table_name, index=True)
        edge_table.to_csv(submodule_edge_table_name, index=True)

        # Graph the abundance of features in each submodule across metadata
        if save_plots is True:
            nodes_df = pd.DataFrame(node_table.index, columns=['node_id'])
            nodes_df = nodes_df.set_index('node_id')
            abundance_df = integrated_data.apply(pd.to_numeric, errors='coerce')
            linked_df = nodes_df.join(abundance_df, how='inner')
            if linked_df.empty:
                print(f"\tNo data available for plotting submodule {idx+1}. Skipping...")
                continue
            else:
                data_series = linked_df.mean(axis=0)
            
            final_df = pd.DataFrame(data_series, columns=['abundance'])
            plot_df = final_df.join(metadata, how='inner')

            plt.figure(figsize=(10, 10))
            sns.violinplot(x='group', y='abundance', data=plot_df, inner=None, palette='viridis')
            sns.stripplot(x='group', y='abundance', data=plot_df, color='k', alpha=0.5)
            plt.title(f'Mean abundance of nodes in submodule{idx+1} across group')
            plt.xlabel('group')
            plt.ylabel(f'Mean abundance')
            plt.xticks(rotation=90)
            plt.tight_layout()

            submodule_boxplot_name = f"{output_filenames['submodule_path']}/submodule_{idx+1}_abundance_boxplot.pdf"
            plt.savefig(submodule_boxplot_name, dpi=300, format='pdf')
            plt.close()

    if save_plots is True:
        print(f"\n\tSubmodule node abundance boxplots plots saved.")

    return graph

# ====================================
# Dataset acquisition functions
# ====================================     

def find_mx_parent_folder(
    pid: str,
    pi_name: str,
    script_dir: str,
    mx_dir: str,
    polarity: str,
    datatype: str,
    chromatography: str,
    filtered_mx: bool = False,
    overwrite: bool = False
) -> str:
    """
    Find the parent folder for metabolomics (MX) data on Google Drive using rclone.

    Args:
        pid (str): Proposal ID.
        pi_name (str): PI name.
        script_dir (str): Directory for scripts.
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
        print("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None
    
    if polarity == "multipolarity":
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        print(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
        return None

    if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
        print("MX folder already downloaded and linked.")
        return None
    elif glob.glob(os.path.expanduser(mx_data_pattern)) and overwrite:
        raise ValueError("You are not currently authorized to download or overwrite metabolomics data from source. Please contact your JGI project manager for access.")
    elif not glob.glob(os.path.expanduser(mx_data_pattern)):
        raise ValueError("MX folder not found locally. Exiting...")

    # Find project folder
    cmd = f"rclone lsd JGI_Metabolomics_Projects: | grep -E '{pid}|{pi_name}'"
    print("Finding MX parent folders...\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        data = [line.split()[:5] for line in result.stdout.strip().split('\n')]
        mx_parent = pd.DataFrame(data, columns=["dir", "date", "time", "size", "folder"])
        mx_parent = mx_parent[["date", "time", "folder"]]
        mx_parent["ix"] = range(1, len(mx_parent) + 1)
        mx_parent = mx_parent[["ix", "date", "time", "folder"]]
        mx_final_folders = []
        # For each possible project folder (some will not be the right "final" folder)
        print("Finding MX final folders...\n")
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
            print("Warning! Multiple untargeted MX final folders found:")
            print(untargeted_mx_final)
            return None
        elif untargeted_mx_final.shape[0] == 0:
            print("Warning! No untargeted MX final folders found.")
            return None
        else:
            final_results_folder = f"{untargeted_mx_final['parent_folder'].values[0]}/{untargeted_mx_final['folder'].values[0]}"

        script_name = f"{script_dir}/find_mx_files.sh"
        with open(script_name, "w") as script_file:
            script_file.write(f"cd {script_dir}/\n")
            script_file.write(f"rclone lsd --max-depth 2 JGI_Metabolomics_Projects:{untargeted_mx_final['parent_folder'].values[0]}")
        
        print("Using the following metabolomics final results folder for further analysis:\n")
        print(untargeted_mx_final)
        return final_results_folder
    else:
        print(f"Warning! No folders could be found with rclone lsd command: {cmd}")
        return None

def gather_mx_files(
    mx_untargeted_remote: str,
    script_dir: str,
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
        script_dir (str): Directory for scripts.
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
        print("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None

    if polarity == "multipolarity":
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{mx_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        print(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
        return None

    if glob.glob(os.path.expanduser(mx_data_pattern)) and not overwrite:
        print("MX data already linked.")
        return "Archive already linked", "Archive already extracted"
    else:
        raise ValueError("You are not currently authorized to download metabolomics data from source. Please contact your JGI project manager for access.")
    
    script_name = f"{script_dir}/gather_mx_files.sh"
    if chromatography == "C18":
        chromatography = "C18_" # This (hopefully) removes C18-Lipid

    # Create the script to link MX files
    with open(script_name, "w") as script_file:
        script_file.write(f"cd {script_dir}/\n")
        script_file.write(f"rclone copy --include '*{chromatography}*.zip' --stats-one-line -v --max-depth 1 JGI_Metabolomics_Projects:{mx_untargeted_remote} {mx_dir};")
    
    print("Linking MX files...\n")
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
        print(f"Warning! No files could be found with rclone copy command in {script_name}")
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
    print("Extracting the following archive to be used for MX data input:\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        data = [line.split() for line in result.stdout.strip().split('\n')]
        df = pd.DataFrame(data)
        df = df[df.iloc[:, 0].str.contains('Archive', na=False)]
        df.iloc[:, 1] = df.iloc[:, 1].str.replace(f"{mx_dir}/", "", regex=False)
        return df
    else:
        print(f"No archives could be decompressed with unzip command: {cmd}")
        return None


def find_tx_files(
    pid: str,
    script_dir: str,
    tx_dir: str,
    tx_index: int,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Find TX files for a project using JAMO report select.

    Args:
        pid (str): Proposal ID.
        script_dir (str): Directory for scripts.
        tx_dir (str): TX data directory.
        tx_index (int): Index of analysis project to use.
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: DataFrame of TX file information.
    """
    
    if os.path.exists(f"{tx_dir}/tx_files.txt") and overwrite is False:
        print("TX files already found.")
        return pd.DataFrame()
    else:
        raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")
    
    file_list = f"{tx_dir}/tx_files.txt"
    script_name = f"{script_dir}/find_tx_files.sh"

    if not os.path.exists(os.path.dirname(file_list)):
        os.makedirs(os.path.dirname(file_list))
    if not os.path.exists(os.path.dirname(script_name)):
        os.makedirs(os.path.dirname(script_name))
    
    print("Creating script to find TX files...\n")
    script_content = (
        f"jamo report select _id,metadata.analysis_project.analysis_project_id,metadata.library_name,metadata.analysis_project.status_name where "
        f"metadata.proposal_id={pid} file_name=counts.txt "
        f"| sed 's/\\[//g' | sed 's/\\]//g' | sed 's/u'\\''//g' | sed 's/'\\''//g' | sed 's/ //g' > {file_list}"
    )

    print("Finding TX files...\n")
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
            print(f"Error fetching refs for {apid}: {e}")
            return pd.DataFrame({"ref_name": [np.nan], "ref_genome": [np.nan], "ref_transcriptome": [np.nan], "ref_gff": [np.nan], "ref_protein_kegg": [np.nan], "APID": [apid]})
    
    refs = pd.concat([fetch_refs(apid) for apid in files["APID"].unique()], ignore_index=True)
    files = files.merge(refs, on="APID", how="left").sort_values(by=["ref_name", "APID"]).reset_index(drop=True)
    files["ix"] = files.index + 1
    # Move "ix" column to the beginning
    cols = ["ix"] + [col for col in files.columns if col != "ix"]
    files = files[cols]
    files.reset_index(drop=True)
    
    if files.shape[0] > 0:
        print(f"Using the value of 'tx_index' ({tx_index}) from the config file to choose the correct 'ix' column (change if incorrect): \n")
        files.to_csv(f"{tx_dir}/jamo_report_tx_files.txt", sep="\t", index=False)
        display(files)
        return files
    else:
        print("No files found.")
        return None

def gather_tx_files(
    file_list: pd.DataFrame,
    tx_index: int = None,
    script_dir: str = None,
    tx_dir: str = None,
    overwrite: bool = False
) -> str:
    """
    Link TX files to the working directory using JAMO.

    Args:
        file_list (pd.DataFrame): DataFrame of TX file information.
        tx_index (int): Index of analysis project to use.
        script_dir (str): Directory for scripts.
        tx_dir (str): TX data directory.
        overwrite (bool): Overwrite existing results.

    Returns:
        str: Analysis project ID (APID).
    """

    if glob.glob(f"{tx_dir}/*counts.txt") and os.path.exists(f"{tx_dir}/tx_files.txt") and overwrite is False:
        print("TX files already linked.")
        tx_files = pd.read_csv(f"{tx_dir}/jamo_report_tx_files.txt", sep="\t")
        apid = int(tx_files.iloc[tx_index-1:,2].values)
        return apid
    else:
        raise ValueError("You are not currently authorized to download transcriptomics data from source. Please contact your JGI project manager for access.")


    if tx_index is None:
        print("There may be multiple APIDS or analyses for a given PI/Proposal ID and you have not specified which one to use!")
        print("Please set 'tx_index' in the project config file by choosing the correct row from the table above.\n")
        sys.exit(1)

    script_name = f"{script_dir}/gather_tx_files.sh"
    print("Linking TX files...\n")

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
                    print(f"\tError linking {file_type} file for APID {apid}. File does not exist or you may need to wait for files to be restored.")
                    continue

    subprocess.run(f"chmod +x {script_name} && {script_name}", shell=True, check=True)
    
    if apid:
        with open(f"{tx_dir}/apid.txt", "w") as f:
            f.write(str(apid))
        print(f"\nWorking with APID: {apid} from tx_index {tx_index}.\n")
        return apid
    else:
        print("Warning: Did not find APID. Check the index you selected from the tx_files object.")
        return None

def get_mx_data(
    input_dir: str,
    output_dir: str,
    output_filename: str,
    chromatography: str,
    polarity: str,
    datatype: str = "peak-height",
    filtered_mx: bool = False,
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
        print("Quant (peak area) data is not filtered. Please use peak-height as 'datatype'.")
        return None
    if polarity == "multipolarity":
        mx_data_pattern = f"{input_dir}/*{chromatography}*/*_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{input_dir}/*{chromatography}*/*_{datatype}.csv"
    elif polarity in ["positive", "negative"]:
        mx_data_pattern = f"{input_dir}/*{chromatography}*/*{polarity}_{datatype}-filtered-3x-exctrl.csv" if filtered_mx else f"{input_dir}/*{chromatography}*/*{polarity}_{datatype}.csv"
    else:
        print(f"Polarity '{polarity}' is not recognized. Please use 'positive', 'negative', or 'multipolarity'.")
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
            print(f"MX data loaded from {mx_data_files}:\n")
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
            print(f"MX data loaded from {mx_data_filename}:\n")
            write_integration_file(data=mx_data, output_dir=output_dir, filename=output_filename, indexing=False)
            #display(mx_data.head())
            return mx_data
        else:
            print("No MX data files found.")
            return None

    else:
        print(f"MX data file matching pattern {mx_data_pattern} not found.")
        return None

def get_mx_metadata(
    output_filename: str,
    output_dir: str,
    input_dir: str,
    chromatography: str,
    polarity: str
) -> pd.DataFrame:
    """
    Load MX metadata from extracted files.

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
            print(f"MX metadata loaded from {mx_metadata_files}\n")
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
            print(f"MX metadata loaded from {mx_metadata_filename}\n")
            print("Writing MX metadata to file...")
            write_integration_file(data=mx_metadata, output_dir=output_dir, filename=output_filename, indexing=False)
            #display(mx_metadata.head())
            return mx_metadata
    else:
        print(f"MX data file matching pattern {mx_metadata_pattern} not found.")
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
            print(f"Multiple TX data files found matching pattern {tx_data_pattern}.")
            print("Please specify the correct file.")
            return None
        tx_data_filename = tx_data_files[0]  # Assuming you want the first (and only) match
        tx_data = pd.read_csv(tx_data_filename, sep='\t')
        tx_data = tx_data.rename(columns={tx_data.columns[0]: 'GeneID'})
        
        # Add prefix 'tx_' if not already present
        tx_data['GeneID'] = tx_data['GeneID'].apply(lambda x: x if str(x).startswith('tx_') else f'tx_{x}')
        
        print(f"TX data loaded from {tx_data_filename} and processing...\n")
        write_integration_file(data=tx_data, output_dir=output_dir, filename=output_filename, indexing=False)
        #display(tx_data.head())
        return tx_data
    else:
        print(f"TX data file matching pattern {tx_data_pattern} not found.")
        return None
    
def get_tx_metadata(
    tx_files: pd.DataFrame,
    output_dir: str,
    proposal_ID: str,
    apid: str,
    overwrite: bool = False
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

    if os.path.exists(f"{output_dir}/portal_metadata.csv") and overwrite is False:
        print("TX metadata already extracted.")
        tx_metadata = pd.read_csv(f"{output_dir}/portal_metadata.csv")
        return tx_metadata
    else:
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

    print("Saving TX metadata...")
    write_integration_file(data=tx_metadata, output_dir=output_dir, filename="portal_metadata", indexing=False)
    #display(tx_metadata.head())
    return tx_metadata

# ====================================
# Feature annotation functions
# ====================================

def parse_gff3_proteinid_to_id(gff3_path: str) -> dict:
    """
    Parse a GFF3 file to map protein IDs to gene IDs.

    Args:
        gff3_path (str): Path to GFF3 file.

    Returns:
        dict: Mapping from proteinId to geneId.
    """

    open_func = gzip.open if gff3_path.endswith('.gz') else open
    proteinid_to_id = {}
    with open_func(gff3_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            attrs = {kv.split('=')[0]: kv.split('=')[1] for kv in fields[8].split(';') if '=' in kv}
            protein_id = attrs.get('proteinId')
            gene_id = attrs.get('ID')
            if protein_id and gene_id:
                proteinid_to_id[protein_id.strip()] = gene_id.strip()
    return proteinid_to_id

def parse_annotation_file(annotation_path: str, tx_id_col: str, tx_annotation_col: str) -> dict:
    """
    Parse an annotation file to map IDs to annotation values.

    Args:
        annotation_path (str): Path to annotation file.
        tx_id_col (str): Column name for IDs.
        tx_annotation_col (str): Column name for annotation.

    Returns:
        dict: Mapping from ID to annotation.
    """
    
    df = pd.read_csv(annotation_path)
    if tx_id_col not in df.columns or tx_annotation_col not in df.columns:
        raise ValueError(f"Required columns {tx_id_col} or {tx_annotation_col} not found in annotation file.")
    df = df.dropna(subset=[tx_id_col, tx_annotation_col])
    annotation_map = dict(zip(df[tx_id_col].astype(str).str.strip(), df[tx_annotation_col].astype(int).astype(str).str.strip()))
    return annotation_map

def map_ids_to_annotations(
    gff3_path: str,
    annotation_path: str,
    tx_id_col: str,
    tx_annotation_col: str
) -> dict:
    """
    Map gene IDs from a GFF3 file to annotation values using an annotation file.

    Args:
        gff3_path (str): Path to GFF3 file.
        annotation_path (str): Path to annotation file.
        tx_id_col (str): ID column name.
        tx_annotation_col (str): Annotation column name.

    Returns:
        dict: Mapping from gene ID to annotation.
    """
    
    proteinid_to_id = parse_gff3_proteinid_to_id(gff3_path)
    annotation_map = parse_annotation_file(annotation_path, tx_id_col, tx_annotation_col)
    id_to_annotation = {}
    for protein_id, gene_id in proteinid_to_id.items():
        for annot_key in annotation_map:
            if protein_id in annot_key:
                id_to_annotation[gene_id] = annotation_map[annot_key]
                break
    return id_to_annotation

def annotate_genes_to_rhea(
    gff3_path: str,
    annotation_path: str,
    tx_id_col: str,
    tx_annotation_col: str
) -> pd.DataFrame:
    """
    Annotate gene IDs with Rhea IDs using GFF3 and annotation files.

    Args:
        gff3_path (str): Path to GFF3 file.
        annotation_path (str): Path to annotation file.
        tx_id_col (str): ID column name.
        tx_annotation_col (str): Annotation column name.

    Returns:
        pd.DataFrame: DataFrame mapping gene IDs to Rhea IDs.
    """

    id_to_annotation = map_ids_to_annotations(gff3_path, annotation_path, tx_id_col, tx_annotation_col)
    id_to_annotation = pd.DataFrame.from_dict(id_to_annotation, orient='index')
    id_to_annotation.index.name = tx_id_col
    id_to_annotation.rename(columns={0: tx_annotation_col}, inplace=True)
    id_to_annotation[tx_id_col] = id_to_annotation.index
    return id_to_annotation

def check_annotation_ids(
    annotated_data: pd.DataFrame,
    dtype: str,
    id_col: str,
    method: str
) -> pd.DataFrame:
    """
    Ensure annotation DataFrame has correct index format and matches integrated data.

    Args:
        annotated_data (pd.DataFrame): Annotation DataFrame.
        dtype (str): Data type prefix ('tx' or 'mx').
        id_col (str): ID column name.
        method (str): Annotation method.

    Returns:
        pd.DataFrame: Checked annotation DataFrame.
    """

    if dtype+"_" in annotated_data[id_col].astype(str).values[0]:
        annotated_data.index = annotated_data[id_col].astype(str)
    else:
        annotated_data.index = dtype+"_" + annotated_data[id_col].astype(str)
    if len(annotated_data.index.intersection(annotated_data.index)) == 0:
        print(f"Warning! No matching indexes between integrated_data and annotation done with {method}. Please check the annotation file.")
        return None
    return annotated_data

def annotate_integrated_data(
    integrated_data: pd.DataFrame,
    output_dir: str,
    project_name: str,
    tx_dir: str,
    mx_dir: str,
    magi2_dir: str,
    tx_annotation_file: str,
    mx_annotation_file: str,
    tx_annotation_method: str,
    mx_annotation_method: str,
    tx_id_col: str,
    mx_id_col: str,
    tx_annotation_col: str,
    mx_annotation_col: str,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Annotate integrated data features with functional information from transcriptomics and metabolomics.

    Args:
        integrated_data (pd.DataFrame): Integrated feature matrix.
        output_dir (str): Output directory.
        project_name (str): Project name.
        tx_dir (str): Transcriptomics data directory.
        mx_dir (str): Metabolomics data directory.
        magi2_dir (str): MAGI2 results directory.
        tx_annotation_file (str): Transcriptomics annotation file.
        mx_annotation_file (str): Metabolomics annotation file.
        tx_annotation_method (str): Annotation method for transcriptomics.
        mx_annotation_method (str): Annotation method for metabolomics.
        tx_id_col (str): Transcriptomics ID column.
        mx_id_col (str): Metabolomics ID column.
        tx_annotation_col (str): Transcriptomics annotation column.
        mx_annotation_col (str): Metabolomics annotation column.
        overwrite (bool): Overwrite existing results.

    Returns:
        pd.DataFrame: DataFrame with feature annotations.
    """
    
    annotated_data_filename = f"integrated_data_annotated"
    if os.path.exists(f"{output_dir}/{annotated_data_filename}.csv") and overwrite is False:
        print(f"Integrated data already annotated: {output_dir}/{annotated_data_filename}.csv")
        annotated_data = pd.read_csv(f"{output_dir}/{annotated_data_filename}.csv", index_col=0)
        return annotated_data

    if tx_annotation_method == "custom":
        if tx_id_col is None or tx_annotation_col is None:
            raise ValueError("Please provide tx_id_col and tx_annotation_col for custom annotation.")
        sep = "\t" if tx_annotation_file.endswith((".txt", ".tsv", ".tab")) else ","
        tx_annotation = pd.read_csv(glob.glob(os.path.expanduser(tx_annotation_file))[0], sep=sep)
        tx_annotation = tx_annotation.drop_duplicates(subset=[tx_id_col])
        tx_annotation[tx_annotation_col] = tx_annotation[tx_annotation_col].astype(str)
        tx_annotation = check_annotation_ids(tx_annotation, "tx", tx_id_col, tx_annotation_method)

    elif tx_annotation_method == "jgi":
        print("JGI annotation method not yet implemented!!")
        return None

    elif tx_annotation_method == "magi2":
        tx_id_col = "gene_ID"
        tx_annotation_col = "rhea_ID_g2r"

        gff_filename = glob.glob(f"{tx_dir}/*gff3*")
        if len(gff_filename) == 0:
            raise ValueError(f"No GFF3 file found in {tx_dir}.")
        elif len(gff_filename) > 1:
            raise ValueError(f"Multiple GFF3 files found in the specified directory: {gff_filename}. \nPlease specify one.")
        elif len(gff_filename) == 1:
            gff_filename = gff_filename[0]

        magi_genes = f"{magi2_dir}/{project_name}/output_{project_name}/magi_gene_results.csv"
        
        tx_annotation = annotate_genes_to_rhea(gff_filename, magi_genes, tx_id_col, tx_annotation_col)
        tx_annotation = check_annotation_ids(tx_annotation, "tx", tx_id_col, tx_annotation_method)

    else:
        raise ValueError("Please select a valid tx_annotation_method: custom, jgi, or magi2.")

    if mx_annotation_method == "fbmn":
        mx_id_col = "#Scan#"
        mx_annotation_col = "Compound_Name"
        mx_annotation = pd.read_csv(f"{mx_dir}/fbmn_compounds.csv")
        mx_annotation = mx_annotation.drop_duplicates(subset=[mx_id_col], keep='first')
        mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)

    elif mx_annotation_method == "custom":
        if mx_id_col is None or mx_annotation_col is None:
            raise ValueError("Please provide mx_id_col and mx_annotation_col for custom annotation.")
        sep = "\t" if mx_annotation_file.endswith((".txt", ".tsv", ".tab")) else ","
        mx_annotation = pd.read_csv(glob.glob(os.path.expanduser(mx_annotation_file))[0], sep=sep)
        mx_annotation = mx_annotation.drop_duplicates(subset=[mx_id_col])
        mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)
        
    elif mx_annotation_method == "magi2":
        mx_id_col = "#Scan#"
        mx_annotation_col = "rhea_ID_r2g"
        magi_compounds = pd.read_csv(f"{magi2_dir}/{project_name}/output_{project_name}/magi_compound_results.csv")
        fbmn_compounds = pd.read_csv(f"{mx_dir}/fbmn_compounds.csv")
        mx_annotation = fbmn_compounds.merge(magi_compounds[['original_compound', mx_annotation_col, 'MAGI_score']],on='original_compound',how='left')
        mx_annotation = mx_annotation.sort_values(by='MAGI_score', ascending=False).drop_duplicates(subset=[mx_id_col], keep='first')
        mx_annotation = check_annotation_ids(mx_annotation, "mx", mx_id_col, mx_annotation_method)

    else:
        raise ValueError("Please select a valid mx_annotation_method: fbmn, custom, or magi2.")

    print("Annotating features in integrated data...")
    tx_annotation = tx_annotation.dropna(subset=[tx_id_col])
    mx_annotation = mx_annotation.dropna(subset=[mx_id_col])
    tx_annotation[tx_annotation_col] = tx_annotation[tx_annotation_col].replace("<NA>","Unassigned").fillna("Unassigned")
    mx_annotation[mx_annotation_col] = mx_annotation[mx_annotation_col].replace("<NA>","Unassigned").fillna("Unassigned")

    annotated_data = integrated_data[[]]
    annotated_data['tx_annotation'] = integrated_data.index.map(tx_annotation[tx_annotation_col])
    annotated_data['mx_annotation'] = integrated_data.index.map(mx_annotation[mx_annotation_col])
    
    annotated_data['annotation'] = annotated_data['tx_annotation'].combine_first(annotated_data['mx_annotation'])
    
    result = annotated_data[['annotation']]
    print("Writing annotated data to file...")
    write_integration_file(data=result, output_dir=output_dir, filename=annotated_data_filename, indexing=True)
    return result

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
        print(f"Saving linked metadata for {ds.dataset_name}...")
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
        print(f"\nProcessing {ds.dataset_name} metadata and data...")

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
        print("\tRestricting matching samples to only those present in all datasets...")
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
        print(f"Saving linked data for {ds.dataset_name}...")
        df = unified_data[ds.dataset_name]
        df = df.set_index(df.columns[0])
        write_integration_file(df, ds.output_dir, ds._linked_data_filename, indexing=True)
        unified_data[ds.dataset_name] = df

    return unified_data

def integrate_datasets(
    datasets: dict,
    output_dir: str
) -> pd.DataFrame:
    """
    Integrate multiple normalized datasets into a single DataFrame.

    Args:
        datasets (dict): Dictionary of DataFrames to integrate.
        output_dir (str): Output directory.

    Returns:
        pd.DataFrame: Integrated dataset.
    """

    integrated_data = pd.concat(datasets.values(), join='inner', ignore_index=False)
    integrated_data.index.name = 'features'
    integrated_data = integrated_data.fillna(0)
    print("Saving integrated dataset to disk...")
    write_integration_file(integrated_data, output_dir, "integrated_data", indexing=True)
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
        print("Low variance detected. Is data already autoscaled?")
        return None
    
    # Determine the variance threshold
    var_thresh = np.quantile(row_variances, filter_percent / 100)
    
    # Filter rows with variance below the threshold
    feat_keep = row_variances >= var_thresh
    filtered_df = data[feat_keep]

    print(f"Started with {data.shape[0]} features; filtered out {filter_percent}% ({data.shape[0] - filtered_df.shape[0]}) to keep {filtered_df.shape[0]}.")

    return filtered_df


def filter_data(
    data: pd.DataFrame,
    dataset_name: str,
    data_type: str,
    output_filename: str,
    output_dir: str,
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
        print(f"Filtering out features with {filter_method} method in {dataset_name} that have an average {data_type} value less than {filter_value} across samples...")
        filtered_data = data[row_means >= filter_value]
        print(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
    elif filter_method == "proportion":
        print(f"Filtering out features with {filter_method} method in {dataset_name} that were observed in fewer than {filter_value}% samples...")
        row_min = max(data.min(axis=1), 0)
        min_count = (data.eq(row_min, axis=0)).sum(axis=1)
        min_proportion = min_count / data.shape[1]
        filtered_data = data[min_proportion <= filter_value / 100]
        print(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - filtered_data.shape[0]} to keep {filtered_data.shape[0]}.")
    elif filter_method == "none":
        print("Not filtering any features.")
        print(f"Keeping all {data.shape[0]} features.")
        filtered_data = data
    else:
        print(f"Invalid filter method '{filter_method}'. Please choose 'minimum', 'proportion', or 'none'.")
        return None

    print(f"Saving filtered data for {dataset_name}...")
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
        print(f"Removing {filter_value}% of features with the lowest variance in {dataset_name}...")
        data_filtered = remove_uniform_percent_low_variance_features(data=data, filter_percent=filter_value)
    elif devariance_mode == "none":
        print(f"\tNot removing any features based on variance in {dataset_name}. Retaining all {data.shape[0]} features.")
        print(f"Saving devarianced data for {dataset_name}...")
        data_filtered = data
    else:
        print(f"Invalid devariance mode '{devariance_mode}'. Please choose 'percent' or 'none'.")
        return None
    
    print(f"Saving devarianced data for {dataset_name}...")
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
    Normalize and scale a feature matrix using log2 and z-score or modified z-score.

    Args:
        df (pd.DataFrame): Feature matrix.
        output_dir (str): Output directory.
        dataset_name (str): Dataset name (for stdout)
        log2 (bool): Apply log2 transformation.
        norm_method (str): Normalization method ('zscore', 'modified_zscore').

    Returns:
        pd.DataFrame: Scaled feature matrix.
    """

    if norm_method == "none":
        print("Not scaling data.")
        scaled_df = df
    if log2 is True:
        print(f"Transforming {dataset_name} data by log2 before z-scoring...")
        unscaled_df = df.copy()
        df = unscaled_df.apply(pd.to_numeric, errors='coerce')
        df = np.log2(df + 1)
    if norm_method == "zscore":
        print(f"Scaling {dataset_name} data to z-scores...")
        scaled_df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    elif norm_method == "modified_zscore":
        print(f"Scaling {dataset_name} data to modified z-scores...")
        scaled_df = df.apply(lambda x: ((x - x.median()) * 0.6745) / (x - x.median()).abs().median(), axis=1)
    else:
        raise ValueError("Please select a valid norm_method: 'zscore' or 'modified_zscore'.")

    print(f"Saving scaled data for {dataset_name}...")
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

    Args:
        data: DataFrame, features as rows, samples as columns.
        metadata: DataFrame, index are sample names, must contain group_col.
        dataset_name: str, Name of the dataset (used for output).
        output_dir: str, Directory to save the filtered data.
        output_filename: str, Name for output file.
        method: Method to assess variability ('variance' supported).
        group_col: Name of the column in metadata for group assignment.
        threshold: Maximum allowed within-group variability (default 0.6).

    Returns:
        Filtered DataFrame.
    """

    if method == "none":
        print(f"\tNot removing any features based on replicability in {dataset_name}. Retaining all {data.shape[0]} features.")
        replicable_data = data
    elif method != "variance":
        variability_func = lambda x: x.std(axis=1, skipna=True)

        if group_col not in metadata.columns:
            raise ValueError(f"Column '{group_col}' not found in metadata.")
        groups = metadata[group_col].unique()
        group_samples = {
            group: metadata[metadata[group_col] == group]['unique_group'].tolist()
            for group in groups
        }

        keep_mask = []
        print(f"Removing features with high within-group variability (threshold: {threshold})...")
        for idx, row in data.iterrows():
            high_var = False
            enough_replicates = False
            for group, samples in group_samples.items():
                samples_in_data = [s for s in samples if s in data.columns]
                if len(samples_in_data) < 2:
                    continue 
                enough_replicates = True
                vals = row[samples_in_data]
                var = variability_func(vals.to_frame().T)
                if var.values[0] > threshold:
                    high_var = True
                    break
            # If there are not enough replicates in any group, keep the feature
            keep_mask.append(not high_var or not enough_replicates)

        replicable_data = data.loc[keep_mask]
        print(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - replicable_data.shape[0]} to keep {replicable_data.shape[0]}.")
    else:
        raise ValueError("Currently only 'variance' or 'none' methods are supported for removing low replicable features.")

    print(f"Saving replicable data for {dataset_name}...")
    write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
    return replicable_data

# ====================================
# Plotting functions
# ====================================

def plot_pca(
    data: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    metadata_variables: List[str],
    alpha: float = 0.75,
    output_dir: str = None,
    output_filename: str = None,
    dataset_name: str = None,
) -> None:
    """
    Plot a PCA of the data colored by a metadata variable.

    Args:
        data (dict): A dictionary containing two DataFrames: {"linked": linked_data, "normalized": normalized_data}.
        metadata (pd.DataFrame): Metadata DataFrame.
        metadata_variables (List[str]): List of metadata variables to plot.
        alpha (float): Transparency for points.
        output_dir (str, optional): Output directory for plots.
        output_filename (str, optional): Name for output file.
        dataset_name (str, optional): Name for output file.

    Returns:
        None
    """

    clear_directory(output_dir)
    plot_paths = {}
    for df_type, df in data.items():
        plot_paths[df_type] = []
        df_samples = df.columns.tolist()
        metadata_samples = metadata['unique_group'].tolist()
        if not set(df_samples).intersection(set(metadata_samples)):
            print(f"Warning: No matching samples between {df_type} data and metadata. Skipping PCA plot for {df_type}.")
            continue
        common_samples = list(set(df_samples).intersection(set(metadata_samples)))
        metadata_input = metadata[metadata['unique_group'].isin(common_samples)]
        data_input = df.T
        data_input = data_input.loc[common_samples]
        data_input = data_input.fillna(0)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_input)
        
        # Merge PCA results with metadata
        pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'], index=data_input.index)
        pca_df = pca_df.reset_index().rename(columns={'index': 'unique_group'})
        pca_df = pca_df.merge(metadata_input, on='unique_group', how='left')
        
        # Plot PCA using seaborn
        for metadata_variable in metadata_variables:
            plt.figure(figsize=(8, 6))
            sns.kdeplot(
                data=pca_df, 
                x='PCA1', 
                y='PCA2', 
                hue=metadata_variable, 
                fill=True, 
                alpha=alpha, 
                palette='viridis',
                bw_adjust=2
            )
            sns.scatterplot(
                x='PCA1', y='PCA2', 
                hue=metadata_variable, 
                palette='viridis', 
                data=pca_df, 
                alpha=alpha,
                s=100,
                edgecolor='w',
                linewidth=0.5
            )
            plt.xlabel('PCA1')
            plt.ylabel('PCA2')
            plt.title(f"{dataset_name} {df_type} data PCA by {metadata_variable}")
            plt.legend(title=metadata_variable)
            
            # Save the plot if output_dir is specified
            if output_dir:
                filename = f"PCA_of_{df_type}_data_by_{metadata_variable}_for_{dataset_name}.pdf"
                output_plot = f"{output_dir}/{filename}"
                print(f"Saving plot to {output_plot}")
                plt.savefig(output_plot)
                plot_paths[df_type].append(output_plot)
                plt.close()
            else:
                print("Not saving plot to disk.")

    plot_pdf_grids(plot_paths, output_dir, metadata_variables, output_filename)

    return

def plot_pdf_grids(plot_paths: dict[str, list[str]], output_dir: str, variables: List[str], output_filename: str) -> None:
    """
    Combine PDF plots into a grid PDF (rows: data types, columns: metadata variables).

    Args:
        plot_paths (dict): {"linked": [...], "normalized": [...]}, each a list of PDF paths.
        variables (List[str]): Metadata variable names (columns).
        output_dir (str): Output directory for combined PDF.
        output_filename (str): Name for output PDF.

    Returns:
        None
    """
    # Build grid: each row is a data type, each column is a variable
    grid = []
    for data_type, paths in plot_paths.items():
        grid_row = []
        for var in variables:
            match = next((f for f in paths if f and f"_by_{var}_" in f), None)
            grid_row.append(match)
        grid.append(grid_row)

    # Set page size based on grid
    n_rows = len(grid)
    n_cols = len(variables)
    page_height = 250 * n_rows
    page_width = 250 * n_cols

    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)

    img_w = page_width / n_cols
    img_h = page_height / n_rows

    for i, grid_row in enumerate(grid):
        for j, pdf in enumerate(grid_row):
            if pdf and os.path.exists(pdf):
                try:
                    src = fitz.open(pdf)
                    src_page = src[0]
                    scale_x = img_w / src_page.rect.width
                    scale_y = img_h / src_page.rect.height
                    scale = min(scale_x, scale_y)
                    mat = fitz.Matrix(scale, scale)
                    rect = fitz.Rect(
                        j * img_w,
                        i * img_h,
                        (j + 1) * img_w,
                        (i + 1) * img_h
                    )
                    page.show_pdf_page(rect, src, 0, mat)
                    src.close()
                except Exception as e:
                    print(f"Error processing {pdf}: {e}")

    output_pdf = os.path.join(output_dir, output_filename)
    doc.save(output_pdf)
    doc.close()
    print(f"Combined PDF saved as {output_pdf}")
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
        print(f"Saving plot to {output_subdir}/{filename}")
        plt.savefig(f"{output_subdir}/{filename}")
    
    plt.show()

def plot_data_variance_histogram(
    dataframes: dict[str, pd.DataFrame],
    bins: int = 50,
    transparency: float = 0.8,
    xlog: bool = False,
    dataset_name: str = None,
    output_dir: str = None
) -> None:
    """
    Plot histograms of values for multiple datasets on the same plot.

    Args:
        dataframes (dict of str: pd.DataFrame): Dictionary mapping labels to feature matrices.
        bins (int): Number of histogram bins.
        transparency (float): Alpha for bars.
        xlog (bool): Use log scale for x-axis.
        dataset_name (str, optional): Name for output file.
        output_dir (str, optional): Output directory for plots.

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
    output_subdir = f"{output_dir}/data_distributions"
    os.makedirs(output_subdir, exist_ok=True)
    filename = f"distribution_of_{dataset_name}.pdf"
    print(f"Saving plot to {output_subdir}/{filename}...")
    plt.savefig(f"{output_subdir}/{filename}")

    plt.show()

def plot_feature_abundance_by_metadata(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    feature: str,
    metadata_group: str | list[str]
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
    row_data = data.loc[feature]
    
    # Merge row data with metadata
    linked_data = pd.merge(row_data.to_frame(name='abundance'), metadata, left_index=True, right_index=True)
    
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
        print(f"Replicate heatmap plot already exists: {output_plot}. Not overwriting.")
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
        print(f"Saving plot to {output_plot}")
        g.figure.savefig(output_plot)
        plt.close(g.figure)
    else:
        print("Not saving plot to disk.")
    
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
        print(f"Heatmap plot already exists: {output_plot}. Not overwriting.")
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
        print(f"Saving plot to {output_plot}")
        g.savefig(output_plot)
        plt.close(g.figure)
    else:
        print("Not saving plot to disk.")

    plt.show()
    plt.close()

def plot_correlated_features(
    data: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: str,
    overwrite: bool = False,
    only_bipartite: bool = True,
    top_n_features: int = 10
) -> None:
    """
    Plot scatterplots for the top correlated feature pairs colored by metadata.

    Args:
        data (pd.DataFrame): Feature matrix.
        correlation_matrix (pd.DataFrame): Correlation matrix DataFrame.
        metadata (pd.DataFrame): Metadata DataFrame.
        output_dir (str): Output directory for plots.
        overwrite (bool): Overwrite existing plots.
        only_bipartite (bool): Exclude pairs from the same dataset prefix.
        top_n_features (int): Number of top pairs to plot.

    Returns:
        None
    """

    output_subdir = f"{output_dir}/feature_correlation"
    os.makedirs(output_subdir, exist_ok=True)
    output_subdir2 = f"{output_subdir}/correlation_plots"
    
    if os.path.exists(output_subdir2) and overwrite is False:
        print(f"Correlation plots already exist! Not overwriting plots.")
        return
    elif not os.path.exists(output_subdir2):
        os.makedirs(output_subdir2, exist_ok=True)

    if only_bipartite is True:
        print("Filtering out pairs with the same prefix (same dataset)...")
        correlation_matrix = correlation_matrix[~correlation_matrix.apply(lambda row: row['feature_1'][:3] == row['feature_2'][:3], axis=1)]

    # Get the top-X most highly correlated pairs
    top_pairs = correlation_matrix.nlargest(top_n_features, 'correlation')
    print(f"Plotting top {top_n_features} most highly correlated features:\n")
    print(top_pairs)
    print("\n")

    for i, row in top_pairs.iterrows():
        feature1 = row['feature_1']
        feature2 = row['feature_2']
        plt.figure(figsize=(7, 4))
        plot = sns.scatterplot(x=data.loc[feature1], y=data.loc[feature2], hue=metadata["group"])
        plt.title(f'Correlation between {feature1} and {feature2}: {row["correlation"]:.3f}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True)
        plt.tight_layout()
        plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        os.makedirs(output_subdir2, exist_ok=True)
        output_plot = f"{output_subdir2}/{feature1}_vs_{feature2}_correlation.pdf"
        plt.savefig(output_plot, bbox_inches='tight')
        plt.close()

    print(f"Saved all plots of best feature pair correlation in {output_subdir2}")
    return

def plot_specific_correlated_features(
    correlation_matrix: pd.DataFrame,
    feature_list: list[str],
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_group: str,
    output_dir: str = None
) -> None:
    """
    Plot scatterplots for specific correlated feature pairs.

    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix DataFrame.
        feature_list (list of str): List of features to plot.
        data (pd.DataFrame): Feature matrix.
        metadata (pd.DataFrame): Metadata DataFrame.
        metadata_group (str): Metadata variable to color by.
        output_dir (str, optional): Output directory for plots.

    Returns:
        None
    """
    
    # Get the top-X most highly correlated pairs
    correlation_matrix_subset = correlation_matrix[correlation_matrix['feature_1'].isin(feature_list) & correlation_matrix['feature_2'].isin(feature_list)]

    for _, row in correlation_matrix_subset.iterrows():
        feature1 = row['feature_1']
        feature2 = row['feature_2']
        plt.figure(figsize=(7, 4))
        plot = sns.scatterplot(x=data.loc[feature1], y=data.loc[feature2], hue=metadata[metadata_group])
        plt.title(f'Correlation between {feature1} and {feature2}: {row["correlation"]:.3f}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True)
        plt.tight_layout()
        
        # Move the legend outside the plot
        plot.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if output_dir:
            plot_subdir = f"{output_dir}/correlated_features"
            os.makedirs(plot_subdir, exist_ok=True)
            plt.savefig(f"{plot_subdir}/{feature1}_vs_{feature2}_correlation.pdf", bbox_inches='tight')
        
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
        print(f"Submodule Sankey diagrams already exist in {os.path.join(submodules_dir, 'submodule_sankeys')}. Set overwrite=True to regenerate.")
        return
    elif os.path.exists(os.path.join(submodules_dir, "submodule_sankeys")) and overwrite is True:
        print(f"Overwriting existing submodule Sankey diagrams in {os.path.join(submodules_dir, 'submodule_sankeys')}.")
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
# MOFA functions
# ====================================

def run_full_mofa2_analysis(
    integrated_data: pd.DataFrame,
    mofa2_views: list[str],
    metadata: pd.DataFrame,
    output_dir: str,
    output_filename: str,
    num_factors: int = 5,
    num_features: int = 10,
    num_iterations: int = 100,
    training_seed: int = 555,
    overwrite: bool = False
) -> None:
    """
    Run a full MOFA2 analysis pipeline for multiple omics datasets and metadata groups.
    https://biofam.github.io/MOFA2/

    Args:
        integrated_data (pd.DataFrame): Integreated omics dataframe (features x samples).
        mofa2_views (list of str): Names for each omics view (e.g., ['tx', 'mx']).
        metadata (pd.DataFrame): Metadata DataFrame (samples x variables).
        output_dir (str): Output directory for results.
        output_filename (str): Name for model h5 file.
        num_factors (int): Number of MOFA factors to compute.
        num_features (int): Number of top features to plot per factor.
        num_iterations (int): Number of training iterations.
        training_seed (int): Random seed for reproducibility.
        overwrite (bool): Overwrite existing results if True.

    Returns:
        None
    """

    clear_directory(output_dir)

    melted_dfs = []
    for datatype in mofa2_views:
        datatype_df = integrated_data.loc[integrated_data.index.str.contains(datatype)]
        datatype_df = datatype_df.rename_axis('features').reset_index()
        datatype_df_melted = datatype_df.melt(id_vars=['features'], var_name='sample', value_name='value')
        datatype_df_melted['view'] = datatype
        melted_dfs.append(datatype_df_melted)
    mofa_data = pd.concat(melted_dfs, ignore_index=True)
    
    # Merge with metadata
    mofa_metadata = metadata.loc[metadata.index.isin(mofa_data['sample'].unique())]
    mofa_data = mofa_data.merge(mofa_metadata, left_on='sample', right_index=True, how='left')
    mofa_data = mofa_data[['sample', 'group', 'features', 'view', 'value']]
    mofa_data = mofa_data.rename(columns={'features': 'feature'})

    print("Converted omics data to mofa2 format:\n")

    # Run the model and load to memory
    model_file = run_mofa2_model(data=mofa_data, output_dir=output_dir, output_filename=output_filename,
                                num_factors=num_factors, num_iterations=num_iterations, 
                                training_seed=training_seed)
    
    model = load_mofa2_model(model_file)
    model.metadata = mofa_metadata
    model.metadata.index.name = 'sample'

    # Print and save output stats and info
    info_text = f"""\
    Information for model:\n
        Samples: {model.shape[0]}
        Features: {model.shape[1]}
        Groups: {', '.join(model.groups)}
        Datasets: {', '.join(model.views)}
    """
    info_file_path = os.path.join(output_dir, f"mofa2_model_summary.txt")
    with open(info_file_path, 'w') as f:
        f.write(info_text)
    print(info_text)

    # Calculate and save model stats
    calculate_mofa2_feature_weights_and_r2(model=model, output_dir=output_dir, num_factors=num_factors)        

    # Plot and save generic output graphs
    #plot_mofa2_factor_r2(model=model, output_dir=output_subdir, num_factors=num_factors)
    plot_mofa2_feature_weights_linear(model=model, num_features=num_features, output_dir=output_dir)
    for dataset in mofa2_views:
        plot_mofa2_feature_weights_scatter(model=model, data_type=dataset, factorX="Factor0", factorY="Factor1",
                                           num_features=num_features, output_dir=output_dir)
    # for dataset in mofa2_views:
    #     plot_mofa2_feature_importance_per_factor(model=model, num_features=num_features,
    #                                              data_type=dataset, output_dir=output_dir)
    return

def run_mofa2_model(
    data: pd.DataFrame,
    output_dir: str = None,
    output_filename: str = "mofa2_model.hdf5",
    num_factors: int = 1,
    num_iterations: int = 100,
    training_seed: int = 555
) -> str:
    """
    Run MOFA2 model training and save the model to disk.

    Args:
        data (pd.DataFrame): Melted MOFA2 input DataFrame.
        output_dir (str): Output directory for model file.
        num_factors (int): Number of factors to compute.
        num_iterations (int): Number of training iterations.
        training_seed (int): Random seed for reproducibility.

    Returns:
        str: Path to the saved MOFA2 model file.
    """

    outfile = os.path.join(output_dir, output_filename)
        
    ent = entry_point()
    ent.set_data_options(
        scale_views = False
    )

    ent.set_data_df(data)

    ent.set_model_options(
        factors = num_factors, 
        spikeslab_weights = True, 
        ard_weights = True,
        ard_factors = True
    )

    ent.set_train_options(
        iter = num_iterations,
        convergence_mode = "fast", 
        dropR2 = 0.001, 
        gpu_mode = False, 
        seed = training_seed
    )

    ent.build()

    ent.run()

    print(f"Exporting model to disk as {outfile}...\n")
    ent.save(outfile=outfile,save_data=True)
    return outfile

def load_mofa2_model(filename: str):
    """
    Load a MOFA2 model from file.

    Args:
        filename (str): Path to MOFA2 model file.

    Returns:
        MOFA2 model object.
    """

    model = mofax.mofa_model(filename)
    return model

def calculate_mofa2_feature_weights_and_r2(
    model,
    output_dir: str,
    num_factors: int
) -> None:
    """
    Calculate and save MOFA2 feature weights and R2 tables for each factor.

    Args:
        model: MOFA2 model object.
        output_dir (str): Output directory to save results.
        num_factors (int): Number of MOFA factors.

    Returns:
        None
    """

    feature_weight_per_factor = model.get_weights(df=True)
    feature_weight_per_factor.set_index(feature_weight_per_factor.columns[0], inplace=False)
    feature_weight_per_factor['abs_sum'] = feature_weight_per_factor.abs().sum(axis=1)
    feature_weight_per_factor.sort_values(by='abs_sum', ascending=False, inplace=True)

    r2_table = model.get_r2(factors=list(range(num_factors))).sort_values("R2", ascending=False)

    print(f"Saving mofa2 factor weights and R2 tables...\n")
    write_integration_file(feature_weight_per_factor, output_dir, f"mofa2_feature_weight_per_factor", index_label='Feature')
    write_integration_file(r2_table, output_dir, f"mofa2_r2_per_factor", indexing=False)

def plot_mofa2_factor_r2(
    model,
    output_dir: str,
    num_factors: int
) -> "plt.Figure":
    """
    Plot and save the R2 values for each MOFA2 factor.

    Args:
        model: MOFA2 model object.
        output_dir (str): Output directory to save the plot.
        num_factors (int): Number of MOFA factors.

    Returns:
        plt.Figure: The matplotlib figure object for the R2 plot.
    """

    r2_plot = mofax.plot_r2(model, factors=list(range(num_factors)), cmap="Blues")

    print(f"Printing and saving mofa2 factor R2 plot...\n")
    r2_plot.figure.savefig(f'{output_dir}/mofa2_r2_per_factor.pdf')

    return r2_plot

def plot_mofa2_feature_weights_linear(
    model,
    output_dir: str,
    num_features: int
) -> "plt.Figure":
    """
    Plot and save the linear feature weights for MOFA2 factors.

    Args:
        model: MOFA2 model object.
        output_dir (str): Output directory to save the plot.
        num_features (int): Number of top features to plot.

    Returns:
        plt.Figure: The matplotlib figure object for the feature weights plot.
    """

    feature_plot = mofax.plot_weights(model, n_features=num_features, label_size=7)
    feature_plot.figure.set_size_inches(16, 8)

    print(f"Printing and saving mofa2 feature weights linear plot...\n")
    feature_plot.figure.savefig(f'{output_dir}/mofa2_feature_weights_linear_plot_combined_data.pdf')

    return feature_plot

def plot_mofa2_feature_weights_scatter(
    model,
    data_type: str,
    factorX: str,
    factorY: str,
    num_features: int,
    output_dir: str
) -> "plt.Figure":
    """
    Plot and save a scatterplot of feature weights for two MOFA2 factors.

    Args:
        model: MOFA2 model object.
        data_type (str): Omics view (e.g., 'tx', 'mx').
        factorX (str): Name of the first factor (e.g., 'Factor0').
        factorY (str): Name of the second factor (e.g., 'Factor1').
        num_features (int): Number of top features to plot.
        output_dir (str): Output directory to save the plot.

    Returns:
        plt.Figure: The matplotlib figure object for the scatter plot.
    """

    feature_plot = mofax.plot_weights_scatter(
        model,
        view=data_type,
        x=factorX, 
        y=factorY, 
        hist=True,
        label_size=8,
        n_features=num_features, 
        linewidth=0, 
        alpha=0.5,
        height=10  # For sns.jointplot
    )
    feature_plot.figure.set_size_inches(10, 10)

    print(f"Printing and saving {data_type} mofa2 feature weights scatter plot...\n")
    feature_plot.figure.savefig(f'{output_dir}/mofa2_feature_weights_scatter_plot_{data_type}_data.pdf')

    return feature_plot

def plot_mofa2_feature_importance_per_factor(
    model,
    num_features: int,
    data_type: str,
    output_dir: str
) -> "plt.Figure":
    """
    Plot and save a dotplot of feature importance per MOFA2 factor.

    Args:
        model: MOFA2 model object.
        num_features (int): Number of top features to plot.
        data_type (str): Omics view (e.g., 'tx', 'mx').
        output_dir (str): Output directory to save the plot.

    Returns:
        plt.Figure: The matplotlib figure object for the dotplot.
    """

    plot = mofax.plot_weights_dotplot(model, 
                                        n_features=num_features, 
                                        view=data_type,
                                        w_abs=True, 
                                        yticklabels_size=8)

    print(f"Printing and saving {data_type} mofa2 feature importance per factor plot...\n")
    plot.figure.savefig(f'{output_dir}/mofa2_feature_importance_per_factor_for_{data_type}_data.pdf')

    return plot

# ====================================
# MAGI functions
# ====================================

def run_magi2(
    run_name: str,
    sequence_file: str,
    compound_file: str,
    output_dir: str,
    magi2_source_dir: str,
    overwrite: bool = False
) -> tuple[str, str]:
    """
    Run MAGI2 analysis by submitting a SLURM job on NERSC.

    Args:
        run_name (str): Name for the MAGI2 run.
        sequence_file (str): Path to the protein FASTA file.
        compound_file (str): Path to the compound input file.
        output_dir (str): Output directory for MAGI2 results.
        magi2_source_dir (str): Directory containing MAGI2 scripts.
        overwrite (bool): Overwrite existing results if True.

    Returns:
        tuple: (compound_results_file, gene_results_file) paths to MAGI2 output files.
    """

    if os.path.exists(f"{output_dir}/{run_name}/output_{run_name}") and overwrite is False:
        print(f"MAGI2 results directory already exists: {output_dir}/{run_name}/output_{run_name}. \nNot queueing new job.")
        print("Returning path to existing results files.")
        compound_results_file = f"{output_dir}/{run_name}/output_{run_name}/magi2_compounds.csv"
        gene_results_file = f"{output_dir}/{run_name}/output_{run_name}/magi2_gene_results.csv"
        return compound_results_file, gene_results_file
    elif os.path.exists(f"{output_dir}/{run_name}/output_{run_name}/") and overwrite is True:
        print(f"MAGI2 results directory already exists: {output_dir}/{run_name}/output_{run_name}. Overwriting...\n.")

    print("Queueing MAGI2 with sbatch...\n")

    SLURM_PERLMUTTER_HEADER = """#!/bin/bash

#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --error="slurm.err"
#SBATCH --output="slurm.out"
#SBATCH -A m2650
#SBATCH -C cpu
#SBATCH --qos=regular
#SBATCH -t 6:00:00

    """

    magi2_sbatch_filename = f"{output_dir}/magi2_slurm.sbatch"
    magi2_runner_filename = f"{output_dir}/magi2_kickoff.sh"
    cmd = f"{magi2_source_dir}/run_magi2.sh {run_name} {sequence_file} {compound_file} {output_dir} {magi2_source_dir}\n\necho MAGI2 completed."

    with open(magi2_sbatch_filename,'w') as fid:
        fid.write(f"{SLURM_PERLMUTTER_HEADER.replace('slurm', f'{output_dir}/magi2_slurm')}\n{cmd}")
    with open(magi2_runner_filename,'w') as fid:
        fid.write(f"sbatch {magi2_sbatch_filename}")

    with open(magi2_runner_filename, 'r') as fid:
        cmd = fid.read()
        sbatch_output = subprocess.run(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE)
        sbatch_output_str = sbatch_output.stdout.decode('utf-8').replace('\n', '')
        job_id = sbatch_output_str.split()[-1]

    user = os.getenv('USER')
    print(f"MAGI2 job submitted with ID: {job_id}. Check status with 'squeue -u {user}'.")
    return "",""

def get_magi2_sequences_input(
    sequence_dir: str,
    output_dir: str,
    fasta_filename: str = None,
    overwrite: bool = False
) -> str | None:
    """
    Prepare the protein FASTA file for MAGI2 input, converting nucleotides to amino acids if needed.

    Args:
        sequence_dir (str): Directory containing sequence files.
        output_dir (str): Output directory for the MAGI2 FASTA file.
        fasta_filename (str, optional): Specific FASTA file to use.
        overwrite (bool): Overwrite existing output if True.

    Returns:
        str or None: Path to the protein FASTA file, or None if not found.
    """

    output_filename = f"{output_dir}/magi2_sequences.fasta"
    if os.path.exists(output_filename) and overwrite is False:
        print(f"MAGI2 sequences file already exists: {output_filename}. \nReturning this file.")
        return output_filename

    if fasta_filename is None:
        fasta_files = [f for f in os.listdir(sequence_dir) if 
                       '.fa' in f.lower() and
                       not f.lower().startswith('aa_') and
                       not 'scaffold' in f.lower()]
        if len(fasta_files) == 0:
            print("No fasta files found.")
            return None
        elif len(fasta_files) > 1:
            print(f"Multiple fasta files found: {fasta_files}. \n\nPlease check {sequence_dir} to verify and use the fasta_filename argument.")
            return None
        elif len(fasta_files) == 1:
            fasta_filename = fasta_files[0]
            print(f"Using the single fasta file found in the sequence directory: {fasta_filename}")
    else:
        print(f"Using the provided fasta file: {fasta_filename}.")
    
    input_fasta = f"{sequence_dir}/{fasta_filename}"
    open_func = gzip.open if input_fasta.endswith('.gz') else open
    protein_fasta = []

    with open_func(input_fasta, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if set(record.seq.upper()).issubset(set("ACGTN")):
                print("Nucleotide sequence detected. Converting to amino acid sequence...")
                protein_fasta = convert_nucleotides_to_amino_acids(input_fasta)
                break
            else:
                protein_fasta.append(record)
    
    with open(output_filename, "w") as output_handle:
        SeqIO.write(protein_fasta, output_handle, "fasta")
    
    return output_filename

def get_magi2_compound_input(
    fbmn_results_dir: str,
    polarity: str,
    output_dir: str,
    compound_filename: str = None,
    overwrite: bool = False
) -> str | None:
    """
    Prepare the compound input file for MAGI2 from FBMN results.

    Args:
        fbmn_results_dir (str): Directory with FBMN results.
        polarity (str): Polarity ('positive', 'negative', 'multipolarity').
        output_dir (str): Output directory for MAGI2 compound file.
        compound_filename (str, optional): Specific compound file to use.
        overwrite (bool): Overwrite existing output if True.

    Returns:
        str or None: Path to the MAGI2 compound file, or None if not found.
    """

    magi_output_filename = f"{output_dir}/magi2_compounds.txt"
    if os.path.exists(magi_output_filename) and overwrite is False:
        print(f"MAGI2 compounds file already exists: {magi_output_filename}. \nReturning this file.")
        return magi_output_filename
    
    if compound_filename is None:
        print(f"Using the compound file(s) from fbmn results directory: {fbmn_results_dir}.")
        try:
            if polarity == "multipolarity":
                compound_files = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*library-results.tsv"))
                if compound_files:
                    if len(compound_files) > 1:
                        multipolarity_datasets = []
                        for mx_data_file in compound_files:
                            file_polarity = (
                                "positive" if "positive" in mx_data_file 
                                else "negative" if "negative" in mx_data_file 
                                else "NO_POLARITY"
                            )
                            mx_dataset = pd.read_csv(mx_data_file, sep='\t')
                            mx_data = mx_dataset.copy()
                            mx_data["#Scan#"] = "mx_" + mx_data["#Scan#"].astype(str) + "_" + file_polarity
                            multipolarity_datasets.append(mx_data)
                        compound_data = pd.concat(multipolarity_datasets, axis=0)
                    elif len(compound_files) == 1:
                        print(f"Warning: Only single compound files found with polarity set to multipolarity.")
                        compound_data = pd.read_csv(compound_files[0], sep='\t')
            elif polarity in ["positive", "negative"]:
                compound_file = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*{polarity}*library-results.tsv"))
                if len(compound_file) > 1:
                    print(f"Multiple compound files found: {compound_file}. \n\nPlease check {fbmn_results_dir} to verify and use the compound_filename argument.")
                    return None
                elif len(compound_file) == 0:
                    print(f"No compound files found for {polarity} polarity.")
                    return None
                elif len(compound_file) == 1:
                    compound_data = pd.read_csv(compound_file[0], sep='\t')
                    compound_data["#Scan#"] = "mx_" + compound_data["#Scan#"].astype(str) + "_" + file_polarity
                print(f"\nUsing metabolomics compound file at {compound_file}")
        except Exception as e:
            print(f"Compound file could not be read due to {e}")
            return None
    elif compound_filename is not None:
        print(f"Using the provided compound file: {compound_filename}.")
        if "csv" in compound_filename:
            compound_data = pd.read_csv(compound_filename)
        else:
            compound_data = pd.read_csv(compound_filename, sep='\t')

    print("\tConverting to MAGI2 input format.")
    fbmn_data = compound_data.rename(columns={"Smiles": 'original_compound'})
    fbmn_data['original_compound'] = fbmn_data['original_compound'].str.strip().replace(r'^\s*$', None, regex=True)

    # Check that smiles is valid to get mol object with rdkit
    keep_smiles = []
    for smiles in list(fbmn_data['original_compound']):
        if pd.isna(smiles):
            continue
        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            keep_smiles.append(smiles_str)
    
    fbmn_data = fbmn_data[fbmn_data['original_compound'].isin(keep_smiles)]
    magi2_output = fbmn_data[['original_compound']].dropna().drop_duplicates()

    magi2_output.to_csv(magi_output_filename, index=False)
    
    return magi_output_filename

def summarize_fbmn_results(
    fbmn_results_dir: str,
    polarity: str,
    overwrite: bool = False
) -> None:
    """
    Summarize FBMN results and export a cleaned compound table.

    Args:
        fbmn_results_dir (str): Directory with FBMN results.
        polarity (str): Polarity ('positive', 'negative', 'multipolarity').
        overwrite (bool): Overwrite existing output if True.

    Returns:
        None
    """

    fbmn_output_filename = f"{fbmn_results_dir}/fbmn_compounds.csv"
    if os.path.exists(fbmn_output_filename) and overwrite is False:
        print(f"FBMN output summary already exists: {fbmn_output_filename}. Not overwriting.")
        return
    
    print(f"Summarizing FBMN file {fbmn_output_filename}.")
    try:
        if polarity == "multipolarity":
            compound_files = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*library-results.tsv"))
            if compound_files:
                if len(compound_files) > 1:
                    multipolarity_datasets = []
                    for mx_data_file in compound_files:
                        file_polarity = (
                            "positive" if "positive" in mx_data_file 
                            else "negative" if "negative" in mx_data_file 
                            else "NO_POLARITY"
                        )
                        mx_dataset = pd.read_csv(mx_data_file, sep='\t')
                        mx_data = mx_dataset.copy()
                        mx_data["#Scan#"] = "mx_" + mx_data["#Scan#"].astype(str) + "_" + file_polarity
                        multipolarity_datasets.append(mx_data)
                    fbmn_summary = pd.concat(multipolarity_datasets, axis=0)
                elif len(compound_files) == 1:
                    print(f"Warning: Only single compound files found with polarity set to multipolarity.")
                    fbmn_summary = pd.read_csv(compound_files[0], sep='\t')
        elif polarity in ["positive", "negative"]:
            compound_file = glob.glob(os.path.expanduser(f"{fbmn_results_dir}/*/*{polarity}*library-results.tsv"))
            if len(compound_file) > 1:
                print(f"Multiple compound files found: {compound_file}. \n\nPlease check {fbmn_results_dir} to verify and use the compound_filename argument.")
                return None
            elif len(compound_file) == 0:
                print(f"No compound files found for {polarity} polarity.")
                return None
            elif len(compound_file) == 1:
                fbmn_summary = pd.read_csv(compound_file[0], sep='\t')
                fbmn_summary["#Scan#"] = "mx_" + fbmn_summary["#Scan#"].astype(str) + "_" + file_polarity
            print(f"\nUsing metabolomics compound file at {compound_file}")
    except Exception as e:
        print(f"Compound file could not be read due to {e}")
        return None

    print("\tConverting FBMN outputs to summary format.")
    fbmn_data = fbmn_summary.rename(columns={"Smiles": 'original_compound'})
    fbmn_data['original_compound'] = fbmn_data['original_compound'].str.strip().replace(r'^\s*$', None, regex=True)

    # Check that smiles is valid to get mol object with rdkit
    keep_smiles = []
    for smiles in list(fbmn_data['original_compound']):
        if pd.isna(smiles):
            continue
        smiles_str = str(smiles)
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is not None:
            keep_smiles.append(smiles_str)
    fbmn_data = fbmn_data[fbmn_data['original_compound'].isin(keep_smiles)]

    fbmn_data.to_csv(fbmn_output_filename, index=False)
    
    return

def convert_nucleotides_to_amino_acids(
    input_fasta: str
) -> list:
    """
    Convert nucleotide FASTA sequences to amino acid sequences.

    Args:
        input_fasta (str): Path to nucleotide FASTA file.

    Returns:
        list: List of SeqRecord objects with amino acid sequences.
    """

    amino_acid_sequences = []
    
    if input_fasta.endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open
    
    with open_func(input_fasta, 'rt') as in_handle:
        for record in SeqIO.parse(in_handle, "fasta"):
            amino_acid_seq = record.seq.translate()
            amino_acid_seq = amino_acid_seq.replace('*', '')
            new_record = record[:]
            new_record.seq = amino_acid_seq
            amino_acid_sequences.append(new_record)
    
    print(f"\tConverted nucleotide -> amino acid sequences.")
    return amino_acid_sequences

# ====================================
# Output functions
# ====================================

def create_excel_metadata_sheet(
    tx_metadata: pd.DataFrame,
    mx_metadata: pd.DataFrame,
    output_dir: str,
    project_name: str
) -> None:
    """
    Create an Excel metadata template for user integration of transcriptomics and metabolomics metadata.

    Args:
        tx_metadata (pd.DataFrame): Transcriptomics metadata DataFrame.
        mx_metadata (pd.DataFrame): Metabolomics metadata DataFrame.
        output_dir (str): Output directory for the Excel file.
        project_name (str): Project name for the file.

    Returns:
        None
    """

    # Create the data for the "Instructions" tab
    print("Creating metadata Excel file...\n")
    instructions_text = (
        "Placeholder\n\n"
    )
    critical_message = (
        "Placeholder\n\n"
    )

    instructions_df = pd.DataFrame([instructions_text], columns=["Instructions"])

    # Create the data for the "RNA" tab
    transcript_df = tx_metadata[['library_name', 'sample_name', 'sample_isolated_from', 'collection_isolation_site_or_growth_conditions']]
    transcript_df.columns = ['JGI_Library_Name', 'JGI_Metadata_SampleName', 'JGI_Metadata_IsolatedFrom', 'JGI_Metadata_Conditions']
    transcript_df['JGI_DataType'] = "Transcriptomics"
    transcript_df['JGI_SampleIndex'] = ['tx' + str(i) for i in range(1, len(transcript_df) + 1)]
    transcript_example_df = pd.DataFrame({
    'JGI_SampleIndex': ['Example', 'Example', 'Example', 'Example', 'Example', 'Example'],
    'JGI_DataType': ['Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics', 'Transcriptomics'],
    'JGI_Library_Name': ['GOYZZ', 'GOZAA', 'GOZBS', 'GOZBT', 'GOZAW', 'GOZAX'],
    'JGI_Metadata_SampleName': ['Swg_Sum_GH_Q13_1', 'Swg_Sum_GH_Q13_2', 'Swg_Fall_GH_Q15_1', 'Swg_Fall_GH_Q15_2', 'Swg_Sum_FT_Q15_1', 'Swg_Sum_FT_Q15_2'],
    'JGI_Metadata_IsolatedFrom': ['Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Greenhouse grown Switchgrass (UCBerkeley)', 'Field grown Switchgrass (UCDavis)', 'Field grown Switchgrass (UCDavis)'],
    'JGI_Metadata_Conditions': ['Greenhouse 1st harvest, Summer_August2018', 'Greenhouse 1st harvest, Summer_August2018', 'Greenhouse 2nd harvest, Fall_Oct2018', 'Greenhouse 2nd harvest, Fall_Oct2018', 'Field trial 1st harvest, Summer_June2018', 'Field trial 1st harvest, Summer_June2018'],
    'USER_MetadataVar_1': ['Summer', 'Summer', 'Fall', 'Fall', 'Summer', 'Summer'],
    'USER_MetadataVar_2': ['Greenhouse', 'Greenhouse', 'Greenhouse', 'Greenhouse', 'Field', 'Field'],
    'USER_MetadataVar_3': ['Q13', 'Q13', 'Q15', 'Q15', 'Q15', 'Q15'],
    'USER_MetadataVar_4': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    'USER_MetadataVar_5': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    'USER_Replicate': ["1", "2", "1", "2", "1", "2"]
    })
    

    # Create the data for the "Metabolites" tab
    metabolites_df = mx_metadata[['file', 'full_sample_metadata']]
    metabolites_df = metabolites_df[~metabolites_df['file'].str.contains(r'_QC_|ctrl|CTRL|ExCtrl', regex=True)]
    metabolites_df.columns = ['JGI_Library_Name', 'JGI_Metadata_SampleName']
    metabolites_df['JGI_DataType'] = "Metabolomics"
    metabolites_df['JGI_SampleIndex'] = ['mx' + str(i) for i in range(1, len(metabolites_df) + 1)]
    metabolites_example_df = pd.DataFrame({
    'JGI_SampleIndex': ['Example', 'Example', 'Example', 'Example', 'Example', 'Example'],
    'JGI_DataType': ['Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics', 'Metabolomics'],
    'JGI_Library_Name': ['20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_121_Swg-SumHvst-GrnHGrwn-QsuB13_1_Rg90to1350-CE102040-30mg-S1_Run18', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_121_Swg-SumHvst-GrnHGrwn-QsuB13_2_Rg90to1350-CE102040-30mg-S1_Run19', \
                        '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_97_Swg-SumHvst-FldGrwn-QsuB15_1_Rg90to1350-CE102040-30mg-S1_Run27', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_97_Swg-SumHvst-FldGrwn-QsuB15_2_Rg90to1350-CE102040-30mg-S1_Run28', \
                        '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_82_Swg-SumHvst-FldGrwn-QsuB10_1_Rg90to1350-CE102040-30mg-S1_Run45', '20190114_KBL_JM_504478_Plant_SwGr_QE-139_C18_USDAY47771_NEG_MSMS_82_Swg-SumHvst-FldGrwn-QsuB10_2_Rg90to1350-CE102040-30mg-S1_Run46'],
    'JGI_Metadata_SampleName': ['swg-sumhvst-grnhgrwn-qsub13', 'swg-sumhvst-grnhgrwn-qsub13', \
                     'swg-sumhvst-fldgrwn-qsub15', 'swg-sumhvst-fldgrwn-qsub15', \
                     'swg-sumhvst-fldgrwn-qsub10', 'swg-sumhvst-fldgrwn-qsub10'],
    'USER_MetadataVar_1': ['Summer', 'Summer', 'Summer', 'Summer', 'Summer', 'Summer'],
    'USER_MetadataVar_2': ['Greenhouse', 'Greenhouse', 'Field', 'Field', 'Field', 'Field'],
    'USER_MetadataVar_3': ['Q13', 'Q13', 'Q15', 'Q15', 'Q10', 'Q10'],
    'USER_MetadataVar_4': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    'USER_MetadataVar_5': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    'USER_Replicate': ["1", "2", "1", "2", "1", "2"]
    })

    linked_data = pd.concat([transcript_df, metabolites_df], ignore_index=True, join='outer')
    linked_data = pd.concat([transcript_example_df, linked_data], ignore_index=True, join='outer')
    linked_data = pd.concat([metabolites_example_df, linked_data], ignore_index=True, join='outer')
    linked_data = linked_data.reindex(columns=transcript_df.columns)
    linked_data.fillna("-", inplace=True)
    linked_data['USER_MetadataVar_1'] = pd.NA
    linked_data['USER_MetadataVar_2'] = pd.NA
    linked_data['USER_MetadataVar_3'] = pd.NA
    linked_data['USER_MetadataVar_4'] = pd.NA
    linked_data['USER_MetadataVar_5'] = pd.NA
    linked_data['USER_Replicate'] = pd.NA
    linked_data['JGI_Group'] = pd.NA
    linked_data['JGI_Unique_SampleName'] = pd.NA
    linked_data = linked_data[['JGI_SampleIndex', 'JGI_DataType', 'JGI_Library_Name', 'JGI_Metadata_SampleName', 'JGI_Metadata_IsolatedFrom', 'JGI_Metadata_Conditions', 'USER_MetadataVar_1', 'USER_MetadataVar_2', 'USER_MetadataVar_3', 'USER_MetadataVar_4', 'USER_MetadataVar_5', 'USER_Replicate', 'JGI_Group', 'JGI_Unique_SampleName']]

    # Write to Excel file
    output_path = f'{output_dir}/{project_name}_metadata_integration_template.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        instructions_df.to_excel(writer, sheet_name='Instructions', index=False)
        linked_data.to_excel(writer, sheet_name='linked_Datasets', index=False)
        tx_metadata.to_excel(writer, sheet_name='Full_Transcript_Metadata_Ref', index=False)
        mx_metadata.to_excel(writer, sheet_name='Full_Metabolite_Metadata_Ref', index=False)

    # Load the workbook and set the wrap text format for the Instructions sheet
    workbook = load_workbook(output_path)
    worksheet = workbook['Instructions']
    worksheet['B1'] = "CRITICAL:"
    worksheet['B1'].font = Font(color='FF0000', bold=True)
    worksheet['B2'] = critical_message
    worksheet['B2'].font = Font(bold=True)
    for row in worksheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    # Set column widths for the Instructions sheet
    worksheet.column_dimensions['A'].width = 130
    worksheet.column_dimensions['B'].width = 130
    worksheet.row_dimensions[2].height = 1070
    
    # Set column widths for the RNA sheet
    worksheet = workbook['linked_Datasets']
    for col, width in zip('ABCDEFGHIJKLMN', [15,15,20,30,40,70,30,30,30,30,30,15,40,40]): worksheet.column_dimensions[col].width = width
    # Add formula to combine metadata columns
    for row in range(2, worksheet.max_row + 1):
        group_formula = (
            f'=IF(AND(LEN(G{row})=0, LEN(H{row})=0, LEN(I{row})=0, LEN(J{row})=0, LEN(K{row})=0),"Undefined Group", '
            f'IF(LEN(G{row})>0,G{row},"")'
            f'&IF(LEN(H{row})>0,"---"&H{row},"")'
            f'&IF(LEN(I{row})>0,"---"&I{row},"")'
            f'&IF(LEN(J{row})>0,"---"&J{row},"")'
            f'&IF(LEN(K{row})>0,"---"&K{row},""))'
        )
        worksheet[f'M{row}'] = group_formula
        unique_formula = f'=IF(LEN(L{row})>0,M{row}&"---"&L{row},"Missing Replicate Number")'
        worksheet[f'N{row}'] = unique_formula
    # Highlight cells for user to fill in
    fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    thin_border = Border(left=Side(style='thin', color='000000'),
                         right=Side(style='thin', color='000000'),
                         top=Side(style='thin', color='000000'),
                         bottom=Side(style='thin', color='000000'))    
    for row in range(14, worksheet.max_row + 1):
        for col in ['G', 'H', 'I', 'J', 'K', 'L']:
            worksheet[f'{col}{row}'].fill = fill
            worksheet[f'{col}{row}'].border = thin_border
    for row in range(2, 14):
        for col in range(1, worksheet.max_column + 1):
            cell = worksheet.cell(row=row, column=col)
            cell.font = Font(italic=True, color="D3D3D3")
    worksheet.freeze_panes = worksheet['A14']

    print(f"\tMetadata instructions Excel file exported to {output_path}")
    # Save the workbook
    workbook.save(output_path)

def copy_data(
    src_dir: str,
    results_subdir: str,
    file_map: dict,
    plots: bool = False,
    plot_pattern: str = "*.pdf"
) -> None:
    """
    Copy data and plot files to user output directories.

    Args:
        src_dir (str): Source directory.
        results_subdir (str): Destination subdirectory.
        file_map (dict): Mapping of source to destination filenames.
        plots (bool): Copy plot files if True.
        plot_pattern (str): Glob pattern for plot files.

    Returns:
        None
    """
    os.makedirs(results_subdir, exist_ok=True)
    # Copy main files
    for src, dst in file_map.items():
        src_path = os.path.join(src_dir, src)
        dst_path = os.path.join(results_subdir, dst)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")

    if not plots:
        return

    # Helper to copy plot files
    def copy_plot_files(pattern, subfolder):
        plot_dir = os.path.join(src_dir, "plots")
        dest_dir = os.path.join(results_subdir, subfolder)
        os.makedirs(dest_dir, exist_ok=True)
        for file_path in glob.glob(os.path.join(plot_dir, pattern)):
            file_name = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(dest_dir, file_name))

    copy_plot_files(f"*_nonnormalized{plot_pattern}", "plots_of_non-normalized_data")
    copy_plot_files(f"*_normalized{plot_pattern}", "plots_of_normalized_data")
    copy_plot_files(f"*_grid{plot_pattern}", "plots_of_combined_data")

def create_user_output_directory(
    mx_dir: str,
    tx_dir: str,
    integrated_dir: str,
    final_dir: str,
    project_name: str
) -> None:
    """
    Organize and copy results to a user-facing output directory structure.

    Args:
        mx_dir (str): Metabolomics results directory.
        tx_dir (str): Transcriptomics results directory.
        integrated_dir (str): Integrated results directory.
        final_dir (str): Final user output directory.
        project_name (str): Name of the project.

    Returns:
        None
    """

    copy_data(
        tx_dir,
        f"{final_dir}/transcript_results",
        {
            #"tx_metadata.csv": "full_transcript_metadata.csv",
            "integrated_metadata.csv": "transcript_metadata.csv",
            "tx_data_nonnormalized.csv": "transcript_count_data_non-normalized.csv",
            "tx_data_normalized.csv": "transcript_count_data_normalized.csv"
        },
        plots=True
    )
    copy_data(
        mx_dir,
        f"{final_dir}/metabolite_results",
        {
            #"mx_metadata.csv": "full_metabolite_metadata.csv",
            "integrated_metadata.csv": "metabolite_metadata.csv",
            "mx_data_nonnormalized.csv": "metabolite_count_data_non-normalized.csv",
            "mx_data_normalized.csv": "metabolite_count_data_normalized.csv"
        },
        plots=True
    )
    copy_data(
        integrated_dir,
        f"{final_dir}/integrated_results",
        {
            "integrated_metadata.csv": "integrated_metadata.csv",
            "integrated_data.csv": "integrated_data.csv",
            "integrated_data_annotated.csv": "integrated_data_annotated.csv"
        },
        plots=False
    )

    # Integrated data and plots
    integrated_results_subdir = f"{final_dir}/integrated_results"
    subdirs = {
        "data_distributions": f"{integrated_results_subdir}/data_distributions",
        "network_analyses": f"{integrated_results_subdir}/network_analyses",
        "mofa2_analyses": f"{integrated_results_subdir}/mofa2_analyses",
        "magi2_analyses": f"{integrated_results_subdir}/magi2_analyses"
    }

    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    for src, dst in subdirs.items():
        shutil.copytree(f"{integrated_dir}/{src}", dst, dirs_exist_ok=True)
        if "magi" in src:
            for file in glob.glob(os.path.join(dst, "*.sh")) + \
                        glob.glob(os.path.join(dst, project_name, "log*")) + \
                        glob.glob(os.path.join(dst, project_name, "error*")) + \
                        [os.path.join(dst, project_name, f"output_{project_name}", "magi_results.csv")]:
                os.remove(file)
            intermediate_dir = os.path.join(dst, project_name, f"output_{project_name}", "intermediate_files")
            shutil.rmtree(intermediate_dir)

    print(f"User output directory structure created at {final_dir}\n")

    return

def upload_to_google_drive(
    project_folder: str,
    project_name: str,
    overwrite: bool = True
) -> None:
    """
    Upload all contents of the user results directory to 
    Google Drive JGI_MXTX_Integration_Projects/{project_name} using rclone.
    """
    dest_folder = f"JGI_MXTX_Integration_Projects:{project_name}/"
    orig_folder = project_folder

    if overwrite is True:
        print("Warning! Overwriting existing files in Google Drive.")
        upload_command = (
            f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone sync '
            f'"{orig_folder}/" "{dest_folder}"'
        )
    else:
        print("Warning! Not overwriting existing files in Google Drive, may have previous files in the output.")
        upload_command = (
            f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone copy --ignore-existing '
            f'"{orig_folder}/" "{dest_folder}"'
        )
    try:
        print(f"Uploading to Google Drive with command:\n\t{upload_command}")
        subprocess.check_output(upload_command, shell=True)
    except Exception as e:
        print(f"Warning! Google Drive upload failed with exception: {e}\nCommand: {upload_command}")
        return

    # Check that upload worked
    check_upload_command = (
        f'/global/cfs/cdirs/m342/USA/shared-envs/rclone/bin/rclone ls "{dest_folder}" --max-depth 2'
    )
    try:
        check_upload_out = subprocess.check_output(check_upload_command, shell=True)
        if check_upload_out.decode('utf-8').strip():
            print(f"\nGoogle Drive upload confirmed!")
            return
        else:
            print(f"Warning! Google Drive upload check failed because no data was returned with command:\n{check_upload_command}.\nUpload may not have been successful.")
            return
    except Exception as e:
        print(f"Warning! Google Drive upload failed on upload check with exception: {e}\nCommand: {check_upload_command}")
        return

# ====================================
# Graveyard functions
# ====================================

# def find_dataset_sizes(dataset_list: list[pd.DataFrame]) -> dict[str, int]:
#     """
#     Calculate the number of features (rows) for each dataset in a list.

#     Args:
#         dataset_list (list of pd.DataFrame): List of feature matrices.

#     Returns:
#         dict: Dictionary mapping dataset prefix (first two characters of index) to number of features.
#     """
    
#     dataset_sizes = {}
#     for dataset in dataset_list:
#         dataset_name = dataset.index[0][:2]
#         dataset_sizes[dataset_name] = dataset.shape[0]
#     return dataset_sizes

# def remove_N_low_variance_features(data: pd.DataFrame, min_rows: int) -> pd.DataFrame | None:
#     """
#     Remove all but the top N features with the highest variance.

#     Args:
#         data (pd.DataFrame): Feature matrix (features x samples).
#         min_rows (int): Number of features to retain.

#     Returns:
#         pd.DataFrame or None: Filtered feature matrix, or None if variance is too low.
#     """

#     # Calculate variance for each row
#     row_variances = data.var(axis=1)
    
#     # Check if the variance of variances is very low
#     if np.var(row_variances) < 0.001:
#         print("Low variance detected. Is data already autoscaled?")
#         return None
    
#     # Sort dataframe by row variance (higher on top)
#     sorted_data = data.loc[row_variances.sort_values(ascending=False).index]
    
#     # Keep the top n rows where n is the min_rows parameter
#     filtered_data = sorted_data.iloc[:min_rows]
    
#     print(f"Started with {data.shape[0]} features; kept top {min_rows} features.")

#     return filtered_data

# def scale_feature_percentages(dataset_sizes: dict[str, int], base_percent: float) -> dict[str, float]:
#     """
#     Scale feature filtering percentages for each dataset based on their size.

#     Args:
#         dataset_sizes (dict): Dictionary mapping dataset prefix to number of features.
#         base_percent (float): Percentage to use for the smallest dataset.

#     Returns:
#         dict: Dictionary mapping dataset prefix to scaled percentage.
#     """

#     # Find the dataset with the smaller size
#     min_dataset = min(dataset_sizes, key=dataset_sizes.get)
#     min_size = dataset_sizes[min_dataset]
    
#     # Calculate the percentage for the smaller dataset
#     percentages = {min_dataset: base_percent}
    
#     # Calculate the percentage for the larger dataset proportionally
#     for dataset, size in dataset_sizes.items():
#         if dataset != min_dataset:
#             percentages[dataset] = (base_percent * size) / min_size
    
#     return percentages


# def integrate_metadata(
#     datasets: list,
#     metadata_vars: list[str] = [],
#     unifying_col: str = 'unique_group',
#     output_filename: str = "integrated_metadata",
#     output_dir: str = None
# ) -> pd.DataFrame:
#     """
#     Integrate multiple metadata tables into a single DataFrame using a unifying column.

#     Args:
#         datasets (list): List of dataset objects with linked_metadata attributes.
#         metadata_vars (list of str): Metadata columns to include.
#         unifying_col (str): Column name to join on.
#         output_dir (str, optional): Output directory to save integrated metadata.

#     Returns:
#         pd.DataFrame: Integrated metadata DataFrame.
#     """

#     print("Creating a single integrated (shared) metadata table across datasets...")

#     metadata_tables = [ds.linked_metadata for ds in datasets if hasattr(ds, "linked_metadata")]

#     subset_cols = metadata_vars + [unifying_col]
#     integrated_metadata = metadata_tables[0][subset_cols].copy()

#     # Merge all subsequent tables
#     for i, table in enumerate(metadata_tables[1:], start=2):
#         integrated_metadata = integrated_metadata.merge(
#             table[subset_cols],
#             on=unifying_col,
#             suffixes=(None, f'_{i}'),
#             how='outer'
#         )
#         # Collapse columns if they match
#         for column in metadata_vars:
#             col1 = column
#             col2 = f"{column}_{i}"
#             if col2 in integrated_metadata.columns:
#                 integrated_metadata[column] = integrated_metadata[col1].combine_first(integrated_metadata[col2])
#                 integrated_metadata.drop(columns=[col2], inplace=True)

#     integrated_metadata.rename(columns={unifying_col: 'sample'}, inplace=True)
#     integrated_metadata.sort_values('sample', inplace=True)
#     integrated_metadata.drop_duplicates(inplace=True)
#     integrated_metadata.set_index('sample', inplace=True)

#     print("Writing integrated metadata table...")
#     write_integration_file(data=integrated_metadata, output_dir=output_dir, filename=output_filename)
    
#     return integrated_metadata

# def integrate_data(
#     datasets: list,
#     overlap_only: bool = True,
#     output_filename: str = "integrated_data",
#     output_dir: str = None
# ) -> pd.DataFrame:
#     """
#     Integrate multiple normalized datasets into a single feature matrix by concatenating features.

#     Args:
#         datasets (list): List of dataset objects with normalized_data attributes.
#         overlap_only (bool): If True, restrict to overlapping samples across datasets.
#         output_dir (str, optional): Output directory to save integrated data.

#     Returns:
#         pd.DataFrame: Integrated feature matrix with all datasets combined.
#     """
    
#     print("Creating a single integrated feature matrix across datasets...")
    
#     # Collect normalized data from all datasets
#     dataset_data = {}
#     sample_sets = {}
    
#     for ds in datasets:
#         if hasattr(ds, 'normalized_data') and not ds.normalized_data.empty:
#             # Add dataset prefix to feature names to avoid conflicts
#             data_copy = ds.normalized_data.copy()
#             if not data_copy.index.str.startswith(f"{ds.dataset_name}_").any():
#                 data_copy.index = [f"{ds.dataset_name}_{idx}" for idx in data_copy.index]
            
#             dataset_data[ds.dataset_name] = data_copy
#             sample_sets[ds.dataset_name] = set(data_copy.columns)
#             print(f"\tAdding {data_copy.shape[0]} features from {ds.dataset_name}")
#         else:
#             raise ValueError(f"Dataset {ds.dataset_name} missing normalized_data. Run processing pipeline first.")
    
#     # Handle overlapping samples if requested
#     if overlap_only and len(dataset_data) > 1:
#         print("\tRestricting to overlapping samples across all datasets...")
#         overlapping_samples = set.intersection(*sample_sets.values())
        
#         if not overlapping_samples:
#             raise ValueError("No overlapping samples found across datasets.")
        
#         # Filter each dataset to only include overlapping samples
#         for ds_name, data in dataset_data.items():
#             overlapping_cols = [col for col in overlapping_samples if col in data.columns]
#             dataset_data[ds_name] = data[overlapping_cols]
#             print(f"\t{ds_name}: {len(overlapping_cols)} overlapping samples")
    
#     # Combine all datasets vertically (concatenate features)
#     integrated_data = pd.concat(dataset_data.values(), axis=0)
#     integrated_data.index.name = 'features'
#     integrated_data = integrated_data.fillna(0)
    
#     print(f"Final integrated dataset: {integrated_data.shape[0]} features x {integrated_data.shape[1]} samples")
    
#     # Save result if output directory provided
#     if output_dir:
#         print("Writing integrated data table...")
#         write_integration_file(integrated_data, output_dir, output_filename, indexing=True)
    
#     return integrated_data

# def write_integrated_metadata(
#     metadata: pd.DataFrame,
#     output_dir: str = None
# ) -> None:
#     """
#     Save integrated metadata to disk.

#     Args:
#         metadata (pd.DataFrame): Metadata DataFrame.
#         output_dir (str, optional): Output directory.

#     Returns:
#         None
#     """

#     print("Saving integrated metadata for individual data types...")
#     write_integration_file(metadata, output_dir, "integrated_metadata", indexing=False)
#     return

# def integrate_data(
#     tx_metadata: pd.DataFrame,
#     tx_data: pd.DataFrame,
#     mx_metadata: pd.DataFrame,
#     mx_data: pd.DataFrame,
#     unifying_col: str = 'unique_group',
#     overlap_only: bool = False
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Integrate transcriptomics and metabolomics data by matching sample names.

#     Args:
#         tx_metadata (pd.DataFrame): Transcriptomics metadata.
#         tx_data (pd.DataFrame): Transcriptomics data.
#         mx_metadata (pd.DataFrame): Metabolomics metadata.
#         mx_data (pd.DataFrame): Metabolomics data.
#         unifying_col (str): Column to unify sample names.
#         overlap_only (bool): If True, restrict to overlapping samples.

#     Returns:
#         tuple: (tx_data_subset, mx_data_subset) with matched columns.
#     """

#     print("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     print("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     print("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         print("\tRestricting matching samples to only those present in both datasets...")
#         # Find overlapping columns
#         overlapping_columns_samples = list(set(mx_data_subset.columns).intersection(set(tx_data_subset.columns)))

#         def get_overlapping_columns(data, base_columns_list, specific_column):
#             base_columns_list.append(specific_column)
#             data = data[base_columns_list]
#             base_columns_list.remove(specific_column)
#             cols = data.columns.tolist()
#             cols = [cols[-1]] + cols[:-1]
#             output = data[cols]
#             output.set_index(output.columns[0], inplace=True)
#             return output

#         tx_data_subset = get_overlapping_columns(tx_data_subset, overlapping_columns_samples, tx_data.columns[0])
#         mx_data_subset = get_overlapping_columns(mx_data_subset, overlapping_columns_samples, mx_data.columns[0])

#         if tx_data_subset.shape[1] != mx_data_subset.shape[1]:
#             print(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


# def write_integrated_data(
#     data: pd.DataFrame,
#     output_dir: str = None,
#     dataset_name: str = None
# ) -> pd.DataFrame:
#     """
#     Save integrated data to disk.

#     Args:
#         data (pd.DataFrame): DataFrame to save.
#         output_dir (str, optional): Output directory.
#         dataset_name (str, optional): Name for output file.

#     Returns:
#         pd.DataFrame: The same DataFrame.
#     """

#     print("Saving integrated data...")
#     write_integration_file(data, output_dir, dataset_name)
#     return data

# def calculate_correlated_features(
#     data: pd.DataFrame,
#     output_filename: str,
#     output_dir: str,
#     corr_method: str = "pearson",
#     corr_cutoff: float = 0.5,
#     overwrite: bool = False,
#     keep_negative: bool = False,
#     only_bipartite: bool = False,
#     save_corr_matrix: bool = False
# ) -> pd.DataFrame:
#     """
#     Calculate and filter feature-feature correlation matrix.

#     Args:
#         data (pd.DataFrame): Feature matrix (features x samples).
#         output_dir (str): Output directory for results.
#         corr_method (str): Correlation method.
#         corr_cutoff (float): Correlation threshold.
#         overwrite (bool): Overwrite existing results.
#         keep_negative (bool): Keep negative correlations if True.
#         only_bipartite (bool): Only keep bipartite correlations.
#         save_corr_matrix (bool): Save correlation matrix to disk.

#     Returns:
#         pd.DataFrame: Filtered correlation matrix.
#     """

#     output_subdir = output_dir
#     #output_subdir = f"{output_dir}/feature_correlation"
#     os.makedirs(output_subdir, exist_ok=True)

#     existing_output = f"{output_subdir}/{output_filename}"
#     if os.path.exists(existing_output) and overwrite is False:
#         print(f"Correlation table already exists at {existing_output}. \nSkipping calculation and returning existing matrix.")
#         return pd.read_csv(existing_output, sep=',')

#     data_input = data.T

#     print(f"Calculating feature correlation matrix...")
#     correlation_matrix = data_input.corr(method=corr_method)
#     melted_corr = correlation_matrix.reset_index().melt(id_vars='features', var_name='features_2', value_name='correlation')
#     melted_corr.rename(columns={'features': 'feature_1', 'features_2': 'feature_2'}, inplace=True)
#     print(f"\tCorrelation matrix calculated with {melted_corr.shape[0]} pairs.")

#     print(f"Filtering correlation matrix by cutoff value {corr_cutoff}...")
#     if keep_negative is True:
#         melted_corr = melted_corr[(abs(melted_corr['correlation']) >= corr_cutoff) & (melted_corr['correlation'] != 1) & (melted_corr['correlation'] != -1)]
#     else:
#         melted_corr = melted_corr[(melted_corr['correlation'] >= corr_cutoff) & (melted_corr['correlation'] != 1)]
#     print(f"\tCorrelation matrix filtered to {melted_corr.shape[0]} pairs.")

#     if only_bipartite is True:
#         print("Filtering for only bipartite correlations...")
#         melted_corr = melted_corr[melted_corr['feature_1'].str[:3] != melted_corr['feature_2'].str[:3]]
#         print(f"\tCorrelation matrix filtered to {melted_corr.shape[0]} pairs.")

#     if save_corr_matrix is True:
#         print(f"Saving and returning feature correlation table to disk...")
#         write_integration_file(melted_corr, output_subdir, output_filename, indexing=False)

#     return melted_corr

# def run_custom_metadata_integration_script(
#     tx_dir: str,
#     mx_dir: str,
#     apid: str,
#     config: dict
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Run a custom script to integrate transcriptomics and metabolomics metadata.

#     Args:
#         tx_dir (str): Transcriptomics data directory.
#         mx_dir (str): Metabolomics data directory.
#         apid (str): Analysis project ID.
#         config (dict): Project configuration dictionary.

#     Returns:
#         tuple: (tx_metadata_input, mx_metadata_input) DataFrames.
#     """

#     # Get the path to the custom integration script
#     custom_script_path = os.path.join(os.path.expandvars(config['integration']['metadata_script']), f"sample_integration_{config['integration']['proposal_ID']}.py")
#     print(f"Using the following script to merge metadata: {custom_script_path}\n")

#     # Load the custom integration script
#     spec = importlib.util.spec_from_file_location("custom_integration", custom_script_path)
#     custom_integration = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(custom_integration)

#     # Call the integration function from the custom script
#     tx_metadata_input, mx_metadata_input = custom_integration.integrate_metadata(tx_dir, mx_dir, apid)

#     summarize_group_differences(tx_metadata_input, mx_metadata_input)

#     return tx_metadata_input, mx_metadata_input

# def summarize_group_differences(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
#     """
#     Print a summary of unique and shared groups between two metadata DataFrames.

#     Args:
#         df1 (pd.DataFrame): First metadata DataFrame, must contain a 'group' column.
#         df2 (pd.DataFrame): Second metadata DataFrame, must contain a 'group' column.

#     Returns:
#         None
#     """
#     combined_groups = pd.concat([df1['group'], df2['group']], axis=0, keys=['df1', 'df2']).reset_index(level=0)
#     group_counts = combined_groups['group'].value_counts()
#     unique_groups = group_counts[group_counts == 1].index
#     shared_groups = group_counts[group_counts > 1].index
    
#     print("Unique groups (appear in only one dataframe):")
#     for group in unique_groups:
#         source_df = combined_groups[combined_groups['group'] == group]['level_0'].values[0]
#         count = group_counts[group]
#         print(f"{group} (from {source_df}, count: {count})")
    
#     print("\nShared groups (appear in both dataframes):")
#     for group in shared_groups:
#         count = group_counts[group]
#         print(f"{group} (count: {count})")