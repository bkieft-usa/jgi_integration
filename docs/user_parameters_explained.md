# User Settings Parameters Explained

This document explains the main options in the user\_settings section of your configuration file. These settings define your project name and the metadata variables that will be used in your analysis.

## 1\. Project Name

*   **project\_name**

A short name for your project. This will be used as the main directory for storing results. A default is given based on the PI name, JGI proposal ID, and a keyword determined by a data analyst.

## 2\. Variable List

*   **variable\_list**

A list of metadata variables (columns from <dataset\_name>.linked\_metadata) that you want to use in your analysis. These variables will be used for grouping samples, statistical tests, and plotting. _Warning_: do not change or remove the "group" line, this is a placeholder that combines all supplied metadata variables to create unique groups of samples.

_Note_: The list of possible metadata variables will already be populated based on the information you provided to JGI during sample processing and data generation. To view how your metadata and samples are organized, run the workflow notebook up to the "_Link analysis datasets by finding corresponding samples and metadata_" section. You can then preview the metadata for your dataset by adding a new notebook cell and running the following command: <dataset\_name>.linked\_metadata)

**Summary:** Set project\_name to determine your base output directory, and review the user\_settings->variable\_list of metadata variables you want to use during analysis - these will be pre-set, but can be subset as needed. These variables will be available for feature selection and downstream analysis.