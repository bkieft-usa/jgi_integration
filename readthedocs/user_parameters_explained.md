# User Settings Parameters Explained

This document describes the practical effect of each option in the **user_settings** section of the integration workflow configuration file (`/input_data/config/project_config.yml`).  
These settings define the project name and the metadata variables that will be used throughout the analysis.

---

## Table of Contents
- [Project Name](#project-name)  
- [Variable List](#variable-list)  
- [Example Configuration](#example-configuration)  

---

## Project Name <a id="project-name"></a>

| Config key                     | Type   | Default | Description |
|--------------------------------|--------|---------|-------------|
| `project_name`   | string | *auto‑generated* | Short name for the project. Used as the top‑level directory for storing all results (e.g., `project_data/output_data/<project_name>/`). If omitted, a default name is built from the PI name, JGI proposal ID, and a keyword supplied by the data analyst. |

---

## Variable List <a id="variable-list"></a>

| Config key                        | Type      | Default | Description |
|-----------------------------------|-----------|---------|-------------|
| `variable_list`      | list of strings | – | List of metadata variables (columns from `<dataset>.linked_metadata`) to be used for grouping samples, statistical tests, and plotting. The special entry **`group`** must be present; it combines all listed variables to create unique sample groups. |
| _Warning_                         | –         | – | Do **not** change or remove the `"group"` line. |
| _Note_                            | –         | – | The set of possible metadata variables is pre‑populated based on the information provided to JGI during sample processing. To inspect the metadata for a specific dataset, run `<dataset>.linked_metadata` in a notebook cell. |

---

## Example Configuration <a id="example-configuration"></a>

```yaml
user_settings:
  project_name: my_jgi_project
  variable_list:
    - site
    - treatment
    - timepoint
    - group