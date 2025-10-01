# User Settings Parameters Explained

This document describes the practical effect of each option in the **user_settings** section of the integration workflow configuration file (`/input_data/config/project_config.yml`). These settings define your project name and the metadata variables that will be used in your analysis.

# User Settings Parameters

## 1. Project Name

  **project_name**

  Short name for your project. This will be used as the main directory for storing results. A default is given based on the PI name, JGI proposal ID, and a keyword determined by a data analyst.

## 2. Variable List

  **variable_list**

  List of metadata variables (columns from `<dataset_name>.linked_metadata`) that you want to use in your analysis. These variables will be used for grouping samples, statistical tests, and plotting.

  _Warning_: Do not change or remove the "group" line. This is a placeholder that combines all supplied metadata variables to create unique groups of samples.

  _Note_: The list of possible metadata variables will already be populated based on the information you provided to JGI during sample processing and data generation. To view how your metadata and samples are organized, run the workflow notebook up to the section that links datasets. You can then preview the metadata for your dataset by adding a new notebook cell and running the following command: `<dataset_name>.linked_metadata`

# Example

Suppose you use the following configuration:

```yaml
user_settings:
    project_name: my_jgi_project
    variable_list:
        - site
        - treatment
        - timepoint
        - group
```

**Result:**

*   The output directory `project_data/output_data` will get a subdirectory `my_jgi_project` that will store all results.

*   The metadata variables that distinguish samples by site, treatment, and timepoint will be used in the statistical analyses, and the "group" variable will combine all listed metadata variables to define unique sample groups (replicates).

**Final Output:**

Your project will be organized under the specified project name, and the selected metadata variables will be available for feature selection and downstream analysis. These variables are pre-set but can be subset as needed or changed by contacting your JGI Project Manager.