# Data Processing Configuration Reference

---  

## Overview

This document describes the practical effect of each option in the **data processing** configuration file (`/input_data/config/data_processing.yml`). Each omics dataset (`tx` counts, `mx` peak heights, `px` peak heights) declares whether its input is `raw` or `prenormalized`, then passes through: raw filtering, sample normalization, imputation and log transform, variance filtering, and replicate-reliability filtering. How the processed data are represented for integration (per-sample or per-condition) is set separately by `analysis.replicate_pairing` in the analysis config.

---

## Table of Contents
- [Configuration](#configuration)  
- [Input State](#input-state)  
- [Normalization Parameters](#normalization-parameters)  
  - [Filtering](#filtering)  
  - [Sample normalization, imputation and log transform](#sample-normalization)  
  - [Devariancing](#devariancing)  
  - [Replicate Handling](#replicate-handling)  
- [Example Configuration](#example-configuration)  
- [Processing Order and Outputs](#outputs)  

---

## Configuration <a id="configuration"></a>

The workflow automatically generates a **data processing hash** based on the parameters in the config file. When you change any parameter (filtering thresholds, normalization methods, etc.), a new hash is generated, ensuring:

- Fresh calculations with new parameters
- Preservation of previous results
- No accidental mixing of results from different parameter sets

Example: Changing the filtering threshold (see below) will generate a new hash like `x9y8z7w6`, creating a new directory `Dataset_Processing--x9y8z7w6/`. All results generated with the new parameter set are saved in their own folder. To re-produce the exact results of the data processing steps again, use the configuration file path that was produced during the previous run.

---

## Input State <a id="input-state"></a>

`datasets.<dataset_name>.input_state` (default `raw`):

| Value | Meaning |
|-------|---------|
| `raw` | Counts (`tx`) or peak heights (`mx`, `px`). Values must be non-negative and linear; negative or log-like values stop the run with an error. Non-integer `tx` counts (e.g. expected counts) are accepted with a warning. |
| `prenormalized` | Already normalized and/or transformed. The scale is detected automatically (linear, log, or already feature-scaled) from the fraction of negative values and zeros, the value range, skewness, and whether features are centered with unit variance. The report is logged and written to `input_report.json` in the dataset output folder. Raw filters and sample normalization are never applied; a log transform is applied only if the data are detected as linear. |

---

## Normalization Parameters <a id="normalization-parameters"></a>

All normalization sub-sections share the same path pattern: `datasets.<dataset_name>.normalization_parameters.<step>`.

### Filtering <a id="filtering"></a>

Raw filters (raw input only; each can be `null` to skip):

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `presence_min_percent` | number 0-100 | `null` | Keep features with a raw value > 0 in at least this percentage of samples. |
| `magnitude_min_mean` | number | `null` | Keep features whose mean raw value across all samples is greater than this value. |

### Sample normalization, imputation and log transform <a id="sample-normalization"></a>

Applied by `analysis.normalize_all_datasets()`.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `sample_normalization` | string | `auto` | Raw input only. `auto` uses median-of-ratios size factors for `tx` and probabilistic quotient normalization (PQN) for `mx`/`px` (`mx` is normalized separately per polarity). Other options: `median_of_ratios`, `pqn`, `median`, `tic`, `none`. |
| `pseudocount` | number | `1` | `tx` only. Counts are transformed as `log2(count / size_factor + pseudocount)`. |
| `recenter_samples` | boolean | `false` | Prenormalized input only. Aligns per-sample medians on the log scale. A warning is logged when sample medians differ by more than one log2 unit. |

For `mx` and `px`, zeros are treated as missing and replaced after normalization by each feature's minimum observed value, then `log2` is applied. `tx` is never imputed. Prenormalized data detected as log or feature-scaled are not transformed again.

### Devariancing <a id="devariancing"></a>

Applied to the log-scale matrix (also for prenormalized input).

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `method` | string | `"none"` | `percent` or `none`. |
| `params.value` | number | — | Percent of features with the lowest variance to drop (0-100). |
| `params.mean_adjusted` | boolean | `true` | Rank variance relative to features of similar mean level instead of absolute variance. |

### Replicate Handling <a id="replicate-handling"></a>

Applied to the log-scale matrix (also for prenormalized input). Replicate groups come from the `group` column of the linked metadata.

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `method` | string | `"none"` | `sd` or `none`. |
| `params.sd_threshold` | number | `1.0` | Within-group standard deviation (log2 scale) above which a group is considered unreliable for a feature. |
| `params.majority_fraction` | number | `0.5` | A feature is removed when it is unreliable in more than this fraction of evaluable groups. |
| `params.min_replicates` | integer | `2` | A group is evaluated for a feature only if the feature is detected in at least this many of its replicates. |

---

## Example Configuration <a id="example-configuration"></a>

A minimal `datasets` block for a transcriptomics dataset (`tx`). The same structure applies to `mx` and `px` (the `px` dataset uses the same annotation files and formats as `tx`).

```yaml
datasets:
  tx:
    dataset_dir: transcriptomics
    input_state: raw
    normalization_parameters:
      filtering:
        presence_min_percent: 20
        magnitude_min_mean: 10
      sample_normalization: auto
      pseudocount: 1
      devariancing:
        method: percent
        params:
          value: 25
          mean_adjusted: true
      replicate_handling:
        method: sd
        params:
          sd_threshold: 1.0
          majority_fraction: 0.5
          min_replicates: 2
```

---

## Processing Order and Outputs <a id="outputs"></a>

| Step | Notebook call | Output file |
|------|---------------|-------------|
| Raw filters | `analysis.filter_all_datasets()` | `filtered_data.csv` |
| Normalize, impute, log2 | `analysis.normalize_all_datasets()` | `normalized_data.csv`, `detected_mask.csv`, `log_data.csv` |
| Variance filter | `analysis.devariance_all_datasets()` | `devarianced_data.csv` |
| Replicate filter | `analysis.replicability_test_all_datasets()` | `replicate_filtered_data.csv` |
| Representation | `analysis.scale_all_datasets()` | `scaled_data_paired.csv` or `scaled_data_unpaired.csv` |

The representation is per-feature z-scores across samples (`paired`) or per-feature z-scored condition medians (`unpaired`), chosen by `analysis.replicate_pairing`.
