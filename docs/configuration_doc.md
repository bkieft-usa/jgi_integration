# Integration Workflow Configuration Reference

This document outlines all available parameters and options for the configuration file used in the integration workflow. The configuration is structured in YAML format and consists of three main sections: `project`, `datasets`, and `analyses`.

---

## Quick Reference: Key Options

**Feature Selection Methods (`feature_selection_method`):**
- `variance`: Select features with highest variance.
- `glm`: Generalized linear model for group comparison.
- `kruskalwallis`: Non-parametric group comparison.
- `feature_list`: Use a user-supplied list of features.
- `none`: No feature selection.

**Networking Modes (`network_mode`):**
- `bipartite`: Only cross-dataset edges.
- `full`: All possible feature-feature edges.

**Submodule Extraction (`submodule_mode`):**
- `community`: Community detection.
- `subgraphs`: Connected components.

**Filtering Methods (`filtering.method`):**
- `minimum`: Minimum average value.
- `proportion`: Minimum proportion of non-missing values.
- `none`: No filtering.

**Devariance Methods (`devariancing.method`):**
- `percent`: Remove lowest variance features.
- `none`: No devariance.

**Scaling Methods (`scaling.method`):**
- `zscore`: Standard z-score.
- `modified_zscore`: Robust z-score.

**Annotation Methods (`annotation.method`):**
- `magi2`: Use MAGI2 annotation.
- `custom`: Use user-supplied annotation file.
- `jgi`: Use JGI annotation (if available).
- `fbmn`: Use FBMN annotation (metabolomics only).

---

## 1. `project` Section

Defines general project-level settings.

| Parameter              | Type      | Description                                                                                 | Example / Options                |
|------------------------|-----------|---------------------------------------------------------------------------------------------|----------------------------------|
| `project_tag`          | int/str   | Tag or version for the project directory naming.                                            | `2`                              |
| `dataset_list`         | list      | List of dataset types to include.                                                           | `['tx', 'mx']`                   |
| `variable_list`        | list      | List of metadata variables to track and use for grouping/analysis.                          | `['group', 'salinity', ...]`     |
| `PI_name`              | str       | Principal Investigator's name.                                                              |                                  |
| `proposal_ID`          | int/str   | JGI Proposal ID.                                                                            |                                  |
| `keyword`              | str       | Project keyword for naming and search.                                                      |                                  |

---

## 2. `datasets` Section

Defines settings for each dataset type (e.g., `tx`, `mx`) and global dataset processing options.

### Global Dataset Options

| Parameter                | Type    | Description                                         | Example / Options |
|--------------------------|---------|-----------------------------------------------------|-------------------|
| `data_processing_tag`    | int/str | Tag for dataset processing versioning.              | `0`               |

### Per-Dataset Options

#### Transcriptomics (`tx`)

| Parameter                    | Type      | Description                                                                 | Example / Options                |
|------------------------------|-----------|-----------------------------------------------------------------------------|----------------------------------|
| `dataset_dir`                | str       | Directory name for transcriptomics data.                                    | `transcriptomics`                |
| `index`                      | int       | Index for selecting the correct analysis project.                           | `1`                              |
| `datatype`                   | str       | Data type for transcriptomics.                                              | `counts`                         |
| `normalization_parameters`   | dict      | Parameters for filtering, scaling, etc. (see below)                         |                                  |
| `annotation`                 | dict      | Annotation method and file info (see below)                                 |                                  |

#### Metabolomics (`mx`)

| Parameter                    | Type      | Description                                                                 | Example / Options                |
|------------------------------|-----------|-----------------------------------------------------------------------------|----------------------------------|
| `dataset_dir`                | str       | Directory name for metabolomics data.                                       | `metabolomics`                   |
| `mode`                       | str       | Acquisition mode.                                                           | `untargeted`                     |
| `chromatography`             | str       | Chromatography type.                                                        | `HILIC`                          |
| `polarity`                   | str       | Polarity mode.                                                              | `multipolarity`                  |
| `datatype`                   | str       | Data type for metabolomics.                                                 | `peak-height`                    |
| `normalization_parameters`   | dict      | Parameters for filtering, scaling, etc. (see below)                         |                                  |
| `annotation`                 | dict      | Annotation method and file info (see below)                                 |                                  |

#### `normalization_parameters` (for both `tx` and `mx`)

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `filtering`              | dict    | Feature filtering options:                                                  |                                  |
| ├─ `method`              | str     | Filtering method: `minimum`, `proportion`, `none`                           | `minimum`                        |
| └─ `value`               | float   | Threshold value for filtering.                                              | `100` (tx), `100000` (mx)        |
| `devariancing`           | dict    | Feature variance filtering options:                                         |                                  |
| ├─ `method`              | str     | Devariance method: `percent`, `none`                                        | `percent`                        |
| └─ `value`               | float   | Percent of features to remove (lowest variance).                            | `25`                             |
| `scaling`                | dict    | Feature scaling options:                                                    |                                  |
| ├─ `method`              | str     | Scaling method: `zscore`, `modified_zscore`                                 | `modified_zscore`                |
| └─ `log2`                | bool    | Whether to apply log2 transformation before scaling.                        | `true`                           |
| `replicate_handling`     | dict    | Replicate filtering options:                                                |                                  |
| ├─ `group`               | str     | Metadata column for replicate grouping.                                     | `group`                          |
| └─ `value`               | float   | Maximum allowed within-group variability.                                   | `0.5`                            |

#### `annotation` (for both `tx` and `mx`)

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `method`                 | str     | Annotation method: `magi2`, `custom`, `jgi`, `fbmn` (mx only)               | `magi2`                          |
| `id_colname`             | str     | Column name for feature IDs (if using custom annotation).                    |                                  |
| `name_colname`           | str     | Column name for annotation names (if using custom annotation).               |                                  |
| `custom_file`            | str     | Path to custom annotation file (if using custom annotation).                 |                                  |

---

## 3. `analyses` Section

Defines settings for the analysis and integration steps.

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `data_analysis_tag`      | int/str | Tag for analysis versioning.                                                | `0`                              |
| `analysis_parameters`    | dict    | Parameters for feature selection, networking, and integration (see below)   |                                  |

### `analysis_parameters`

#### `feature_selection`

| Parameter                    | Type      | Description                                                                 | Example / Options                |
|------------------------------|-----------|-----------------------------------------------------------------------------|----------------------------------|
| `feature_selection_method`   | str       | Method for feature selection: `glm`, `kruskalwallis`, `variance`, `none`, `feature_list` | `glm`                |
| `metadata_category`          | str       | Metadata column to use for group comparison.                                | `salinity`                       |
| `metadata_category_reference`| str       | Reference group for GLM or group comparison.                                | `LowS`                           |
| `significance_level`         | float     | FDR-corrected p-value cutoff.                                               | `0.05`                           |
| `log_fold_level`             | float     | Minimum absolute log2 fold change.                                          | `0.5`                            |
| `max_features`               | int       | Maximum number of features to retain.                                       | `10000`                          |
| `feature_list_file`          | str       | Path to file with list of features to keep (if using `feature_list`).       |                                  |

#### `networking`

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `network_mode`           | str     | Network type: `bipartite`, `full`                                           | `bipartite`                      |
| `submodule_mode`         | str     | Submodule extraction: `community`, `subgraphs`                              | `community`                      |
| `corr_method`            | str     | Correlation method: `pearson`, `spearman`, etc.                             | `pearson`                        |
| `corr_cutoff`            | float   | Correlation threshold for network edges.                                    | `0.8`                            |
| `keep_negative`          | bool    | Whether to keep negative correlations.                                      | `false`                          |

#### `magi`

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `magi_source_dir`        | str     | Path to MAGI2 source directory.                                             |                                  |
| `magi_sequence_input`    | str     | Path to MAGI2 sequence input file.                                          |                                  |
| `magi_compound_input`    | str     | Path to MAGI2 compound input file.                                          |                                  |

#### `mofa`

| Parameter                | Type    | Description                                                                 | Example / Options                |
|--------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `num_mofa_factors`       | int     | Number of MOFA factors to compute.                                          | `5`                              |
| `num_mofa_iterations`    | int     | Number of MOFA training iterations.                                         | `1000`                           |
| `seed_for_training`      | int     | Random seed for MOFA training.                                              | `555`                            |

---

## Notes

- All file paths should be absolute or relative to the workflow root.
- For custom annotation, provide both `id_colname`, `name_colname`, and `custom_file`.
- For `feature_selection_method: feature_list`, provide `feature_list_file`.
- The `network_mode` and `submodule_mode` control the structure and extraction of the correlation network.
- The `magi` and `mofa` sections are only required if those analyses are to be run.