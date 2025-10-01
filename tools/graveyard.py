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
#         log.info("Low variance detected. Is data already autoscaled?")
#         return None
    
#     # Sort dataframe by row variance (higher on top)
#     sorted_data = data.loc[row_variances.sort_values(ascending=False).index]
    
#     # Keep the top n rows where n is the min_rows parameter
#     filtered_data = sorted_data.iloc[:min_rows]
    
#     log.info(f"Started with {data.shape[0]} features; kept top {min_rows} features.")

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

# def integrate_datasets(
#     datasets: dict,
#     output_dir: str
# ) -> pd.DataFrame:
#     """
#     Integrate multiple normalized datasets into a single DataFrame.

#     Args:
#         datasets (dict): Dictionary of DataFrames to integrate.
#         output_dir (str): Output directory.

#     Returns:
#         pd.DataFrame: Integrated dataset.
#     """

#     integrated_data = pd.concat(datasets.values(), join='inner', ignore_index=False)
#     integrated_data.index.name = 'features'
#     integrated_data = integrated_data.fillna(0)
#     log.info("Saving integrated dataset to disk...")
#     write_integration_file(integrated_data, output_dir, "integrated_data", indexing=True)
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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


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

#     log.info("Saving integrated metadata for individual data types...")
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

#     log.info("Creating dataframes with matching sample names...")

#     # Figure out which tx_metadata columns represents the column names in tx_data (prior to replacement)
#     def data_colnames_to_replace(metadata, data):
#         data_columns = data.columns.tolist()[1:]
#         for column in metadata.columns:
#             metadata_columns = set(metadata[column])
#             if set(data_columns).issubset(metadata_columns) or metadata_columns.issubset(set(data_columns)):
#                 return column
#         return None
    
#     # Process tx_metadata
#     log.info("Processing tx_metadata...")
#     tx_sample_col = data_colnames_to_replace(tx_metadata, tx_data)
#     library_names_tx = tx_metadata[tx_sample_col].tolist()
#     tx_data_subset = tx_data[[tx_data.columns[0]] + [col for col in tx_data.columns if col in library_names_tx]]
#     mapping_tx = dict(zip(tx_metadata[tx_sample_col], tx_metadata[unifying_col]))
#     tx_data_subset.columns = [tx_data_subset.columns[0]] + [mapping_tx.get(col, col) for col in tx_data_subset.columns[1:]]
#     tx_data_subset.set_index(tx_data_subset.columns[0], inplace=True)

#     # Process mx_metadata
#     log.info("Processing mx_metadata...")
#     mx_sample_col = data_colnames_to_replace(mx_metadata, mx_data)
#     library_names_mx = mx_metadata[mx_sample_col].tolist()
#     mx_data_subset = mx_data[[mx_data.columns[0]] + [col for col in mx_data.columns if col in library_names_mx]]
#     mapping_mx = dict(zip(mx_metadata[mx_sample_col], mx_metadata[unifying_col]))
#     mx_data_subset.columns = [mx_data_subset.columns[0]] + [mapping_mx.get(col, col) for col in mx_data_subset.columns[1:]]
#     mx_data_subset.set_index(mx_data_subset.columns[0], inplace=True)
#     mx_data_subset = mx_data_subset.groupby(mx_data_subset.columns, axis=1).sum() # This is needed for multipolarity

#     if overlap_only is True:
#         log.info("\tRestricting matching samples to only those present in both datasets...")
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
#             log.info(f"Overlap only is True but number of rows DO NOT match between tx output data ({tx_data_subset.shape[1]}) and mx output data ({mx_data_subset.shape[1]}). Something might be wrong.")

#     return tx_data_subset, mx_data_subset


# def write_integrated_metadata(
#     metadata: pd.DataFrame,
#     output_dir: str = None
# ) -> None:
#     """
#     Save integrated metadata to disk.

#     Args:
#         metadata (pd.DataFrame): Metadata DataFrame.
#         output_dir


# def _nx_to_cytoscape(
#     G,
#     node_color_attr="submodule_color",
#     node_size_attr="node_size",
#     layout="cose",
#     animate=True,
#     precomputed_pos=None,
# ):
#     """
#     Convert a NetworkX graph to an ipycytoscape widget.

#     Parameters
#     ----------
#     G : networkx.Graph
#         Graph to visualise.
#     node_color_attr : str
#         Node attribute that holds a colour string.
#     node_size_attr : str
#         Node attribute that holds a numeric size.
#     layout : str
#         Cytoscape layout name (e.g. "cose", "grid", …). Ignored if
#         ``precomputed_pos`` is supplied - “preset” will be forced.
#     animate : bool
#         Whether the layout animation should be streamed. Ignored for
#         “preset”.
#     precomputed_pos : dict or None
#         Mapping ``node → (x, y)`` produced by a NetworkX layout function.
#         If given the positions are written into ``node["position"]`` and
#         the widget uses the static “preset” layout.
#     """

#     if precomputed_pos is not None:
#         for n, (x, y) in precomputed_pos.items():
#             G.nodes[n]["position"] = {"x": float(x), "y": float(y)}
#         layout = "preset"
#         animate = False

#     cyto = CytoscapeWidget()
#     cyto.graph.add_graph_from_networkx(G)

#     cyto.set_style([
#         {
#             "selector": "node",
#             "style": {
#                 "background-color": f"data({node_color_attr})",
#                 "width": f"data({node_size_attr})",
#                 "height": f"data({node_size_attr})",
#                 "label": "data(name)",
#                 "font-size": 10,
#                 "color": "#fff",
#             },
#         },
#         {"selector": "edge", "style": {"line-color": "#888", "width": 1}},
#         {"selector": "node:selected", "style": {"border-color": "#ff0", "border-width": 3}},
#     ])

#     cyto.set_layout(name=layout, animate=animate, randomize=False, fit=True)

#     return cyto

# def nx_to_cytoscape(G,
#                     node_color_attr="submodule_color",
#                     node_size_attr="size",
#                     layout="cose",
#                     animate=True):
#     """Convert a NetworkX graph to a styled ipycytoscape widget."""
#     cyto = CytoscapeWidget()
#     cyto.graph.add_graph_from_networkx(G)

#     # Default stylesheet – you can extend/customize as needed
#     cyto.set_style([
#         {"selector": "node",
#          "style": {
#             "background-color": f"data({node_color_attr})",
#             "width": f"data({node_size_attr})",
#             "height": f"data({node_size_attr})",
#             "label": "data(name)",
#             "font-size": 10,
#             "color": "#fff"
#          }},
#         {"selector": "edge",
#          "style": {"line-color": "#888", "width": 1}},
#         {"selector": "node:selected",
#          "style": {"border-color": "#ff0", "border-width": 3}}
#     ])

#     cyto.set_layout(name=layout, animate=animate, randomize=False, fit=True)
#     return cyto

# def _nx_to_cytoscape_layout(
#     G,
#     layout_name="perfuse-force-directed",
#     seed=42,
#     max_iterations=1000,
#     temp_dir=None
# ):
#     """
#     Use Cytoscape to compute node positions for a NetworkX graph.
    
#     Parameters
#     ----------
#     G : networkx.Graph
#         Graph to layout
#     layout_name : str
#         Cytoscape layout algorithm name
#     seed : int
#         Random seed for reproducible layouts
#     max_iterations : int
#         Maximum iterations for layout algorithm
#     temp_dir : str, optional
#         Directory for temporary files
        
#     Returns
#     -------
#     dict
#         Node positions as {node_id: (x, y)}
#     """
    
#     if layout_name not in [
#         "perfuse-force-directed",
#         "prefuse-force-directed",
#         "organic",
#         "circular",
#         "hierarchical",
#         "grid"]:
#         raise ValueError(f"Unsupported Cytoscape layout '{layout_name}'")

#     if temp_dir is None:
#         temp_dir = tempfile.mkdtemp()
    
#     # Export graph to Cytoscape JSON format
#     cyjs_file = os.path.join(temp_dir, "network.cyjs")
#     positions_file = os.path.join(temp_dir, "positions.json")
    
#     # Convert NetworkX to Cytoscape JSON format
#     cyjs_data = {
#         "elements": {
#             "nodes": [
#                 {
#                     "data": {
#                         "id": str(node),
#                         "name": str(node),
#                         **data
#                     }
#                 }
#                 for node, data in G.nodes(data=True)
#             ],
#             "edges": [
#                 {
#                     "data": {
#                         "id": f"{source}_{target}",
#                         "source": str(source),
#                         "target": str(target),
#                         "weight": data.get("weight", 1.0)
#                     }
#                 }
#                 for source, target, data in G.edges(data=True)
#             ]
#         }
#     }
    
#     # Save network to file
#     with open(cyjs_file, 'w') as f:
#         json.dump(cyjs_data, f)
    
#     # Create Cytoscape automation script
#     automation_script = f"""
# import json
# import py4cytoscape as p4c
# import time

# # Connect to Cytoscape
# try:
#     p4c.cytoscape_ping()
# except:
#     print("Error: Cytoscape is not running. Please start Cytoscape first.")
#     exit(1)

# # Load network
# network_suid = p4c.import_network_from_file('{cyjs_file}')
# p4c.set_current_network(network_suid)

# # Apply layout with parameters
# layout_params = {{
#     'randomSeed': {seed},
#     'maxIterations': {max_iterations}
# }}

# p4c.layout_network(layout_name='{layout_name}', **layout_params)

# # Give layout time to complete
# time.sleep(2)

# # Get node positions
# node_table = p4c.get_table_columns('node', ['name', 'x', 'y'])
# positions = {{}}
# for i, row in node_table.iterrows():
#     positions[row['name']] = [row['x'], row['y']]

# # Save positions
# with open('{positions_file}', 'w') as f:
#     json.dump(positions, f)

# # Clean up
# p4c.delete_network(network_suid)
# print("Layout completed successfully")
# """
    
#     script_file = os.path.join(temp_dir, "layout_script.py")
#     with open(script_file, 'w') as f:
#         f.write(automation_script)
    
#     try:
#         # Run the Cytoscape automation script
#         log.info("Computing layout using Cytoscape...")
#         print(script_file)
#         result = subprocess.run([
#             'python3', script_file
#         ], capture_output=True, text=True, timeout=120)
        
#         if result.returncode != 0:
#             raise RuntimeError(f"Cytoscape layout failed: {result.stderr}")
        
#         # Load the computed positions
#         if os.path.exists(positions_file):
#             with open(positions_file, 'r') as f:
#                 positions = json.load(f)
            
#             # Convert string keys back to original node types and normalize coordinates
#             pos_dict = {}
#             if positions:
#                 x_vals = [pos[0] for pos in positions.values()]
#                 y_vals = [pos[1] for pos in positions.values()]
#                 x_min, x_max = min(x_vals), max(x_vals)
#                 y_min, y_max = min(y_vals), max(y_vals)
                
#                 # Normalize to [0, 1] range
#                 for node, (x, y) in positions.items():
#                     if x_max != x_min and y_max != y_min:
#                         norm_x = (x - x_min) / (x_max - x_min)
#                         norm_y = (y - y_min) / (y_max - y_min)
#                     else:
#                         norm_x, norm_y = 0.5, 0.5
#                     pos_dict[node] = (norm_x, norm_y)
            
#             return pos_dict
#         else:
#             raise RuntimeError("Cytoscape did not generate position file")
            
#     except Exception as e:
#         log.warning(f"Cytoscape layout failed: {e}")
#         return None
#     finally:
#         # Clean up temporary files
#         for file in [cyjs_file, positions_file, script_file]:
#             if os.path.exists(file):
#                 os.remove(file)