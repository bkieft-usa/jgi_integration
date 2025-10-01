# Datasets Parameters Explained

---  

## Overview

This document describes the practical effect of each option in the **datasets** section of the integration workflow configuration file (`/input_data/config/project_config.yml`). It covers the data‑processing tag, the dataset directory, and the four normalization steps applied to each omics dataset (filtering, devariancing, scaling, and replicate handling). Each option lists available methods, their parameters, default values, and a short description. An example configuration and a brief walk‑through are provided at the end.

---

## Table of Contents
- [Tagging](#tagging)  
- [Normalization Parameters](#normalization-parameters)  
  - [2.1 Filtering](#filtering)  
  - [2.2 Devariancing](#devariancing)  
  - [2.3 Scaling](#scaling)  
  - [2.4 Replicate Handling](#replicate-handling)  
- [Example Configuration](#example-configuration)  

---

## Tagging <a id="tagging"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `data_processing_tag` | string | `"0"` | Tag used to create a sub‑folder `Data_Processing--<TAG>` under the data‑processing output. Changing the tag creates a fresh output directory and prevents overwriting previous runs. |
| `dataset_dir` | string | *fixed* (e.g., `transcriptomics`, `metabolomics`) | Directory name for the omics type. **Do not modify**; it is currently tied to the workflow structure. |

---

## Normalization Parameters <a id="normalization-parameters"></a>

All normalization sub‑sections share the same path pattern: `datasets.<dataset_name>.normalization_parameters.<step>`.

### Filtering <a id="filtering"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `method` | string | `"minimum"` | Filtering method. Options: `minimum`, `proportion`, `none`. |
| `value` | number | — | Threshold for the chosen method. <br>• **minimum** – real > 0 (absolute value). <br>• **proportion** – real 0‑100 (percentage of samples). <br>• **none** – ignored. |

### Devariancing <a id="devariancing"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `method` | string | `"percent"` | Devariancing method. Options: `percent`, `none`. |
| `value` | number | — | Percent of features with lowest variance to drop (0‑100). Ignored when method is `none`. |

### Scaling <a id="scaling"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `log2` | boolean | `true` | If `true`, apply `log2(x + 1)` to all values before scaling. |
| `method` | string | `"modified_zscore"` | Scaling method. Options: `modified_zscore`, `zscore`, `none`. <br>• **modified_zscore** – median‑MAD based standardization (robust to outliers). <br>• **zscore** – mean‑std standardization. <br>• **none** – raw values (not recommended for integration). |

### Replicate Handling <a id="replicate-handling"></a>

| Config key | Type | Default | Description |
|------------|------|---------|-------------|
| `method` | string | `"variance"` | Replicate‑handling method. Options: `variance`, `none`. |
| `group` | string | `"group"` | Metadata column used to define replicate groups (must be listed in `user_settings.variable_list`). Ignored when method is `none`. |
| `value` | number | `0.5` | Maximum allowed within‑group variability (e.g., variance or MAD). Features exceeding this threshold are removed. Ignored when method is `none`. |

---

## Example Configuration <a id="example-configuration"></a>

Below is a minimal yet complete `datasets` block for a transcriptomics dataset (`tx`). The same structure can be duplicated for other omics types (e.g., `mx`).

```yaml
datasets:
  data_processing_tag: 0
  tx:
    dataset_dir: transcriptomics
    normalization_parameters:
      filtering:
        method: minimum
        value: 10
      devariancing:
        method: percent
        value: 20
      scaling:
        log2: true
        method: modified_zscore
      replicate_handling:
        method: variance
        group: group
        value: 0.5
```

And the following transcriptomics dataset (features as rows, samples as columns), with two sample groupings (high or low):

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 900 | 850 | 920 | 870 | 910 | 20 | 25 | 22 | 18 | 24 |
| FeatureB | 5 | 8 | 7 | 6 | 9 | 4 | 3 | 5 | 6 | 4 |
| FeatureC | 100 | 120 | 110 | 130 | 115 | 90 | 95 | 85 | 100 | 92 |
| FeatureD | 500 | 520 | 510 | 530 | 515 | 480 | 490 | 470 | 495 | 485 |
| FeatureE | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| FeatureF | 700 | 750 | 720 | 710 | 740 | 680 | 690 | 670 | 700 | 685 |
| FeatureG | 50 | 55 | 52 | 54 | 53 | 51 | 56 | 53 | 55 | 52 |
| FeatureH | 15 | 18 | 17 | 16 | 19 | 14 | 13 | 15 | 16 | 14 |
| FeatureI | 300 | 320 | 310 | 330 | 315 | 290 | 295 | 285 | 300 | 292 |
| FeatureJ | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 1 |

### Step 1: Filtering

  * **method:** minimum, **value:** 10  
    Remove features whose average observed value is below 10.  
    _Result_: Remove FeatureB (average observed = 5.7, below threshold)

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 900 | 850 | 920 | 870 | 910 | 20 | 25 | 22 | 18 | 24 |
| FeatureC | 100 | 120 | 110 | 130 | 115 | 90 | 95 | 85 | 100 | 92 |
| FeatureD | 500 | 520 | 510 | 530 | 515 | 480 | 490 | 470 | 495 | 485 |
| FeatureE | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| FeatureF | 700 | 750 | 720 | 710 | 740 | 680 | 690 | 670 | 700 | 685 |
| FeatureG | 50 | 55 | 52 | 54 | 53 | 51 | 56 | 53 | 55 | 52 |
| FeatureH | 15 | 18 | 17 | 16 | 19 | 14 | 13 | 15 | 16 | 14 |
| FeatureI | 300 | 320 | 310 | 330 | 315 | 290 | 295 | 285 | 300 | 292 |
| FeatureJ | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 1 |

### Step 2: Devariancing

  * **method:** percent, **value:** 20  
    Remove 20% of features with the lowest variance (rounded down, i.e., 1 feature).  
    _Result_: Remove FeatureE (variance = 0)

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 900 | 850 | 920 | 870 | 910 | 20 | 25 | 22 | 18 | 24 |
| FeatureC | 100 | 120 | 110 | 130 | 115 | 90 | 95 | 85 | 100 | 92 |
| FeatureD | 500 | 520 | 510 | 530 | 515 | 480 | 490 | 470 | 495 | 485 |
| FeatureF | 700 | 750 | 720 | 710 | 740 | 680 | 690 | 670 | 700 | 685 |
| FeatureG | 50 | 55 | 52 | 54 | 53 | 51 | 56 | 53 | 55 | 52 |
| FeatureH | 15 | 18 | 17 | 16 | 19 | 14 | 13 | 15 | 16 | 14 |
| FeatureI | 300 | 320 | 310 | 330 | 315 | 290 | 295 | 285 | 300 | 292 |
| FeatureJ | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 1 |

### Step 3: Scaling

  * **log2:** true, **method:** modified_zscore  
    First, apply log2(x+1) transformation to all values.  
    Then, apply modified z-score standardization.

**First, apply log2(x+1) transformation:**

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 9.813 | 9.741 | 9.842 | 9.770 | 9.831 | 4.392 | 4.700 | 4.523 | 4.247 | 4.643 |
| FeatureC | 6.658 | 6.918 | 6.797 | 7.044 | 6.857 | 6.507 | 6.614 | 6.426 | 6.658 | 6.523 |
| FeatureD | 8.967 | 9.025 | 8.995 | 9.053 | 9.010 | 8.918 | 8.965 | 8.888 | 8.977 | 8.931 |
| FeatureF | 9.454 | 9.561 | 9.492 | 9.470 | 9.545 | 9.419 | 9.453 | 9.398 | 9.454 | 9.423 |
| FeatureG | 5.672 | 5.807 | 5.700 | 5.779 | 5.740 | 5.700 | 5.857 | 5.740 | 5.807 | 5.700 |
| FeatureH | 4.000 | 4.322 | 4.247 | 4.170 | 4.392 | 3.907 | 3.807 | 4.000 | 4.170 | 3.907 |
| FeatureI | 8.233 | 8.330 | 8.285 | 8.375 | 8.309 | 8.201 | 8.236 | 8.154 | 8.233 | 8.207 |
| FeatureJ | 3.700 | 3.700 | 3.700 | 3.700 | 3.700 | 3.700 | 3.700 | 3.700 | 3.700 | 1.000 |

**Then, apply modified z-score:**

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 0.009 | -0.009 | 0.034 | -0.028 | 0.025 | -1.022 | -0.622 | -0.883 | -1.263 | -0.713 |
| FeatureC | 0.073 | 0.442 | 0.253 | 0.701 | 0.326 | -0.179 | 0.011 | -0.357 | 0.073 | -0.126 |
| FeatureD | 0.025 | 0.093 | 0.062 | 0.130 | 0.087 | -0.025 | 0.022 | -0.055 | 0.034 | -0.012 |
| FeatureF | 0.044 | 0.186 | 0.089 | 0.047 | 0.179 | -0.022 | 0.044 | -0.067 | 0.044 | -0.015 |
| FeatureG | -0.067 | 0.179 | -0.022 | 0.134 | 0.067 | -0.022 | 0.224 | 0.067 | 0.179 | -0.022 |
| FeatureH | -0.134 | 0.224 | 0.134 | 0.044 | 0.313 | -0.224 | -0.313 | -0.134 | 0.044 | -0.224 |
| FeatureI | 0.044 | 0.186 | 0.089 | 0.228 | 0.120 | -0.022 | 0.044 | -0.067 | 0.044 | -0.015 |
| FeatureJ | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -2.494 |

### Step 4: Replicate Handling

  * **method:** variance, **group:** group, **value:** 0.5  
    Remove features with high within-group variance (threshold = 0.5).  
    _Result_: Remove FeatureJ (Low_5 sample value is an outlier, causing high within-group variance in "low" group)

| Feature | High_1 | High_2 | High_3 | High_4 | High_5 | Low_1 | Low_2 | Low_3 | Low_4 | Low_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FeatureA | 0.009 | -0.009 | 0.034 | -0.028 | 0.025 | -1.022 | -0.622 | -0.883 | -1.263 | -0.713 |
| FeatureC | 0.073 | 0.442 | 0.253 | 0.701 | 0.326 | -0.179 | 0.011 | -0.357 | 0.073 | -0.126 |
| FeatureD | 0.025 | 0.093 | 0.062 | 0.130 | 0.087 | -0.025 | 0.022 | -0.055 | 0.034 | -0.012 |
| FeatureF | 0.044 | 0.186 | 0.089 | 0.047 | 0.179 | -0.022 | 0.044 | -0.067 | 0.044 | -0.015 |
| FeatureG | -0.067 | 0.179 | -0.022 | 0.134 | 0.067 | -0.022 | 0.224 | 0.067 | 0.179 | -0.022 |
| FeatureH | -0.134 | 0.224 | 0.134 | 0.044 | 0.313 | -0.224 | -0.313 | -0.134 | 0.044 | -0.224 |
| FeatureI | 0.044 | 0.186 | 0.089 | 0.228 | 0.120 | -0.022 | 0.044 | -0.067 | 0.044 | -0.015 |

**Final Output:**

After all normalization steps, the dataset contains 7 features and all 10 samples, with all values log2-transformed and modified z-score standardized. This dataset now has a quality-controlled and standardized quantitative distribution and can be integrated with other datasets that have undergone the same treatment.