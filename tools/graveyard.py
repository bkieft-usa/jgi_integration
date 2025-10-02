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


# def plot_pca(
#     data: dict[str, pd.DataFrame],
#     metadata: pd.DataFrame,
#     metadata_variables: List[str],
#     alpha: float = 0.75,
#     output_dir: str = None,
#     analysis_outdir: str = None,
#     output_filename: str = None,
#     dataset_name: str = None,
#     show_plot: bool = False,
# ) -> None:
#     """
#     Plot a PCA of the data colored by a metadata variable.

#     Args:
#         data (dict): A dictionary containing two DataFrames: {"linked": linked_data, "normalized": normalized_data}.
#         metadata (pd.DataFrame): Metadata DataFrame.
#         metadata_variables (List[str]): List of metadata variables to plot.
#         alpha (float): Transparency for points.
#         output_dir (str, optional): Output directory for plots.
#         output_filename (str, optional): Name for output file.
#         dataset_name (str, optional): Name for output file.

#     Returns:
#         None
#     """

#     clear_directory(output_dir)
#     plot_paths = {}
#     for df_type, df in data.items():
#         plot_paths[df_type] = []
#         df_samples = df.columns.tolist()
#         metadata_samples = metadata['unique_group'].tolist()
#         if not set(df_samples).intersection(set(metadata_samples)):
#             log.info(f"Warning: No matching samples between {df_type} data and metadata. Skipping PCA plot for {df_type}.")
#             continue
#         common_samples = list(set(df_samples).intersection(set(metadata_samples)))
#         metadata_input = metadata[metadata['unique_group'].isin(common_samples)]
#         data_input = df.T
#         data_input = data_input.loc[common_samples]
#         data_input = data_input.fillna(0)
#         data_input = data_input.replace([np.inf, -np.inf], 0)

#         # Perform PCA
#         pca = PCA(n_components=2)
#         pca_result = pca.fit_transform(data_input)
        
#         # Merge PCA results with metadata
#         pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'], index=data_input.index)
#         pca_df = pca_df.reset_index().rename(columns={'index': 'unique_group'})
#         pca_df = pca_df.merge(metadata_input, on='unique_group', how='left')
        
#         # Plot PCA using seaborn
#         for metadata_variable in metadata_variables:
#             plt.figure(figsize=(8, 6))
#             sns.kdeplot(
#                 data=pca_df, 
#                 x='PCA1', 
#                 y='PCA2', 
#                 hue=metadata_variable, 
#                 fill=True, 
#                 alpha=alpha, 
#                 palette='viridis',
#                 bw_adjust=2
#             )
#             sns.scatterplot(
#                 x='PCA1', y='PCA2', 
#                 hue=metadata_variable, 
#                 palette='viridis', 
#                 data=pca_df, 
#                 alpha=alpha,
#                 s=100,
#                 edgecolor='w',
#                 linewidth=0.5
#             )
#             plt.xlabel('PCA1')
#             plt.ylabel('PCA2')
#             plt.title(f"{dataset_name} {df_type} data PCA by {metadata_variable}")
#             plt.legend(title=metadata_variable)
            
#             # Save the plot if output_dir is specified
#             if output_dir:
#                 filename = f"PCA_of_{df_type}_data_by_{metadata_variable}_for_{dataset_name}.pdf"
#                 output_plot = f"{output_dir}/{filename}"
#                 log.info(f"Saving plot to {output_plot}")
#                 plt.savefig(output_plot)
#                 plot_paths[df_type].append(output_plot)
#                 if show_plot:
#                     plt.show()
#                 plt.close()
#             else:
#                 log.info("Not saving plot to disk.")

#     plot_pdf_grids(plot_paths, analysis_outdir, metadata_variables, output_filename)

#     return

# def plot_pdf_grids(plot_paths: dict[str, List[str]], output_dir: str, variables: List[str], output_filename: str) -> None:
#     """
#     Combine PDF plots into a grid PDF (rows: data types, columns: metadata variables).

#     Args:
#         plot_paths (dict): {"linked": [...], "normalized": [...]}, each a list of PDF paths.
#         variables (List[str]): Metadata variable names (columns).
#         output_dir (str): Output directory for combined PDF.
#         output_filename (str): Name for output PDF.

#     Returns:
#         None
#     """
#     # Build grid: each row is a data type, each column is a variable
#     grid = []
#     for data_type, paths in plot_paths.items():
#         grid_row = []
#         for var in variables:
#             match = next((f for f in paths if f and f"_by_{var}_" in f), None)
#             grid_row.append(match)
#         grid.append(grid_row)

#     # Set page size based on grid
#     n_rows = len(grid)
#     n_cols = len(variables)
#     page_height = 250 * n_rows
#     page_width = 250 * n_cols

#     doc = fitz.open()
#     page = doc.new_page(width=page_width, height=page_height)

#     img_w = page_width / n_cols
#     img_h = page_height / n_rows

#     for i, grid_row in enumerate(grid):
#         for j, pdf in enumerate(grid_row):
#             if pdf and os.path.exists(pdf):
#                 try:
#                     src = fitz.open(pdf)
#                     src_page = src[0]
#                     scale_x = img_w / src_page.rect.width
#                     scale_y = img_h / src_page.rect.height
#                     scale = min(scale_x, scale_y)
#                     mat = fitz.Matrix(scale, scale)
#                     rect = fitz.Rect(
#                         j * img_w,
#                         i * img_h,
#                         (j + 1) * img_w,
#                         (i + 1) * img_h
#                     )
#                     page.show_pdf_page(rect, src, 0, mat)
#                     src.close()
#                 except Exception as e:
#                     log.info(f"Error processing {pdf}: {e}")

#     output_pdf = os.path.join(output_dir, output_filename)
#     doc.save(output_pdf)
#     doc.close()
#     log.info(f"Combined PDF saved as {output_pdf}")
#     print(os.path.exists(output_pdf))
#     display(IFrame(output_pdf, width=900, height=600))
#     return


# def glm_selection(
#     data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     category: str,
#     reference: str,
#     significance_level: float = 0.05,
#     log2fc_cutoff: float = 0.25,
#     max_features: int = 10000,
# ) -> pd.DataFrame:
#     """
#     Fit a Gaussian GLM for each feature vs. a categorical metadata column.
#     Returns a subset of features that satisfy BOTH a p-value threshold
#     (FDR-adjusted) and an absolute log2-fold-change threshold.
#     """
#     # Merge once
#     df = data.T.join(metadata)
#     if category not in df.columns:
#         raise ValueError(f"Metadata column '{category}' not found.")
#     # Force ordered categorical with reference first
#     uniq = df[category].dropna().unique()
#     if reference not in uniq:
#         raise ValueError(f"Reference group '{reference}' not present in '{category}'.")
#     ordered_cats = [reference] + [c for c in uniq if c != reference]
#     df[category] = pd.Categorical(df[category], categories=ordered_cats, ordered=True)

#     log.info(f"GLM: fitting {data.shape[0]} features against '{category}' (ref={reference})")
#     coeffs = {}
#     pvals = {}
#     for feat in data.index:
#         formula = f'Q("{feat}") ~ C({category})'
#         try:
#             res = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()
#             # The first non-intercept term corresponds to the reference contrast
#             term = [t for t in res.params.index if t != "Intercept"][0]
#             coeffs[feat] = res.params[term]
#             pvals[feat] = res.pvalues[term]
#         except Exception as exc:  # pragma: no cover
#             log.debug(f"GLM failed for {feat}: {exc}")
#             coeffs[feat] = np.nan
#             pvals[feat] = np.nan

#     coeffs_s = pd.Series(coeffs, name="coef")
#     pvals_s = pd.Series(pvals, name="pvalue")
#     # Multiple-testing correction
#     adj, sig_mask = _fdr_correct(pvals_s.values, alpha=significance_level)
#     adj_s = pd.Series(adj, index=pvals_s.index, name="adj_pvalue")

#     # Build a DataFrame of filters
#     passes = (sig_mask) & (coeffs_s.abs() >= log2fc_cutoff)
#     selected = passes[passes].index.tolist()
#     if len(selected) > max_features:
#         # rank by absolute coefficient (largest effect first)
#         selected = coeffs_s.loc[selected].abs().sort_values(ascending=False).index[:max_features].tolist()
#     log.info(f"GLM selected {len(selected)} features (max_features={max_features})")
#     return data.loc[selected]

# def kruskal_selection(
#     data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     category: str,
#     significance_level: float = 0.05,
#     lfc_cutoff: float = 0.5,
#     max_features: int = 10000,
# ) -> pd.DataFrame:
#     """
#     Kruskal-Wallis test per feature against a categorical metadata column.
#     Returns features that satisfy both an FDR-adjusted p-value < `significance_level`
#     and an absolute median-difference (as a proxy for effect size) >= `lfc_cutoff`.
#     """
#     if category not in metadata.columns:
#         raise ValueError(f"Metadata column '{category}' missing.")
#     groups = metadata[category].dropna().unique()
#     if len(groups) < 2:
#         raise ValueError(f"Need at least two groups in '{category}' for Kruskal-Wallis.")
#     # Build dict of sample indices per group
#     idx_by_grp = {g: metadata[metadata[category] == g].index for g in groups}
#     pvals = {}
#     effects = {}
#     log.info(f"Kruskal-Wallis test: evaluating {data.shape[0]} features for '{category}'")
#     for feat in data.index:
#         samples = [data.loc[feat, idx_by_grp[g]].values for g in groups]
#         # Constant vectors yield p=1
#         if all(np.allclose(s, s[0]) for s in samples):
#             p = 1.0
#             eff = 0.0
#         else:
#             _, p = stats.kruskal(*samples)
#             medians = [np.median(s) for s in samples]
#             eff = max(abs(m1 - m2) for i, m1 in enumerate(medians)
#                                           for m2 in medians[i + 1 :])
#         pvals[feat] = p
#         effects[feat] = eff

#     pvals_s = pd.Series(pvals, name="pvalue")
#     eff_s = pd.Series(effects, name="effect")
#     adj, sig_mask = _fdr_correct(pvals_s.values, alpha=significance_level)
#     adj_s = pd.Series(adj, index=pvals_s.index, name="adj_pvalue")
#     # Apply both thresholds
#     passes = (sig_mask) & (eff_s.abs() >= lfc_cutoff)
#     selected = passes[passes].index.tolist()
#     if len(selected) > max_features:
#         selected = eff_s.loc[selected].abs().sort_values(ascending=False).index[:max_features].tolist()
#     log.info(f"Kruskal-Wallis selected {len(selected)} features")
#     return data.loc[selected]


# def remove_low_replicable_features(
#     data: pd.DataFrame,
#     metadata: pd.DataFrame,
#     dataset_name: str,
#     output_filename: str,
#     output_dir: str,
#     method: str = "variance",
#     group_col: str = "group",
#     threshold: float = 0.6,
# ):
#     """
#     Remove features (rows) from `data` with high within-group variability.

#     Args:
#         data: DataFrame, features as rows, samples as columns.
#         metadata: DataFrame, index are sample names, must contain group_col.
#         dataset_name: str, Name of the dataset (used for output).
#         output_dir: str, Directory to save the filtered data.
#         output_filename: str, Name for output file.
#         method: Method to assess variability ('variance' supported).
#         group_col: Name of the column in metadata for group assignment.
#         threshold: Maximum allowed within-group variability (default 0.6).

#     Returns:
#         Filtered DataFrame.
#     """

#     if method == "none":
#         log.info(f"\tNot removing any features based on replicability in {dataset_name}. Retaining all {data.shape[0]} features.")
#         replicable_data = data
#     elif method == "variance":
#         variability_func = lambda x: x.std(axis=1, skipna=True)

#         if group_col not in metadata.columns:
#             raise ValueError(f"Column '{group_col}' not found in metadata.")
#         groups = metadata[group_col].unique()
#         group_samples = {
#             group: metadata[metadata[group_col] == group]['unique_group'].tolist()
#             for group in groups
#         }

#         keep_mask = []
#         log.info(f"Removing features with high within-group variability (threshold: {threshold})...")
#         for idx, row in data.iterrows():
#             high_var = False
#             enough_replicates = False
#             for group, samples in group_samples.items():
#                 samples_in_data = [s for s in samples if s in data.columns]
#                 if len(samples_in_data) < 2:
#                     continue 
#                 enough_replicates = True
#                 vals = row[samples_in_data]
#                 var = variability_func(vals.to_frame().T)
#                 if var.values[0] > threshold:
#                     high_var = True
#                     break
#             # If there are not enough replicates in any group, keep the feature
#             keep_mask.append(not high_var or not enough_replicates)

#         replicable_data = data.loc[keep_mask]
#         log.info(f"Started with {data.shape[0]} features; filtered out {data.shape[0] - replicable_data.shape[0]} to keep {replicable_data.shape[0]}.")
#     else:
#         raise ValueError("Currently only 'variance' or 'none' methods are supported for removing low replicable features.")

#     log.info(f"Saving replicable data for {dataset_name}...")
#     write_integration_file(replicable_data, output_dir, output_filename, indexing=True)
#     return replicable_data
