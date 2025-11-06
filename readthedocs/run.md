# **Running the JGI Integration Workflow**

## **Overview**

The general stages of the workflow are:

*   Loading and normalizing multi-omics datasets (e.g., transcriptomics and metabolomics)

*   Integrating data (quantitative values) and metadata (sample info) between datasets

*   Selecting features for analysis via statistical tests

*   Building and visualizing a correlation network of integrated features

*   Running multi-omics factor analysis

*   Saving your configuration and notebook files to create persistent copies of the workflow run

_Note:_ This tutorial assumes you’ve already completed all stages of the workflow and environment setup detailed in _setup.md._

## **Configuration**

The workflow uses a hash-based tagging system that automatically generates unique identifiers for your analysis runs based on the exact parameters used. This ensures:

- **Automatic results checking**: Changing any parameter creates a new hash, preventing accidental use of stale results or overwriting existing outputs
- **Precise dependency tracking**: Only the specific processing steps affected by parameter changes are re-run
- **Safe experimentation**: Previous results are preserved with their exact parameter combinations

### Configuration Files Structure

The configuration is split into three separate files:

1. **`project.yml`** - Unchanging project metadata (shown above)
2. **`data_processing.yml`** - Parameters affecting data processing steps
3. **`analysis.yml`** - Parameters affecting analysis steps

When you initialize a project, hashes are automatically generated. Your results will be organized in hash-tagged directories. For example, running the workflow with default parameter would produce an output like this:

```
my_jgi_project/
├── configs/
│   ├── Dataset_Processing--a1b2c3d4_Analysis--e5f6g7h8_project_config.yml
│   ├── Dataset_Processing--a1b2c3d4_Analysis--e5f6g7h8_data_processing_config.yml
│   └── Dataset_Processing--a1b2c3d4_Analysis--e5f6g7h8_analysis_config.yml
├── notebooks/
│   └── Dataset_Processing--a1b2c3d4_Analysis--e5f6g7h8_notebook.ipynb
└── Dataset_Processing--a1b2c3d4/
    └── Analysis--e5f6g7h8/
        └── [all results files]
```

### Running Fresh vs. Existing Analysis

**Fresh Analysis (new parameters):**
For a new analysis, you can update the configuration files in the JupyterLab interface or your local directory at `input_data/config/*yml`, which will create a new set of results in unique output folders.

**Re-run Existing Analysis (exact reproduction):**
```python
# Optionally, see what existing analyses are available in the previously saved configuration files
hlp.list_persistent_configs()

# Use hashes to load specific saved configuration (replace the default None)
project = objs.Project(data_processing_hash="a1b2c3d4", analysis_hash="e5f6g7h8")

# If, for some reason, you want to overwrite existing workflow results
project = objs.Project(data_processing_hash="a1b2c3d4", analysis_hash="e5f6g7h8", overwrite=True)
```

When re-running existing analysis, the workflow automatically:
- Loads exact parameters from your previous run
- Finds and reuses all existing results
- Re-runs any steps that do not have existing results

Re-runs are useful for reproducing results, continuing interrupted analyses, or sharing configurations with JGI or collaborators.

## **Running the Notebook**

1.  You should see the JupyterLab interface rendered in your browser tab.

2.  Double-click the workflow notebook `/integration_workflow.ipynb` in the left menu navigator to bring it into the workspace.

3.  Double-click the data processing and analysis configuration files in `/input_data/config/*.yml` to bring them into the workspace. The JupyterLab interface will open the files in a text editor for you to change parameters during analyses.

4.  Run a workflow with all default parameters:
    
    - Start with the `integration_workflow.ipynb` in the workspace window.

    - Ensure that the kernel is “JGI Integration” by checking the top right corner of the workspace. If not, click the kernel name and select “JGI Integration” from the dropdown menu.

    - Run all notebook cells in order, either with the “play” button in the top menu bar or with the keyboard shortcut in each cell (ctrl/cmd/shift-click).

    - Each cell performs a workflow step (data loading, normalization, integration, correlation analysis, etc.) and prints some information to the standard output for review.

    - Review the cell print statements to see where plots and tables are saved to the `/output_data` directory; some key results or previews are also displayed in the notebook.

5.  Run a workflow with customized parameters (optional):

    - Edit the `/input_data/config/*.yml` files to update the data processing and analysis parameters. Use the guides provided in the `/jgi_integration/docs/*_parameters_explained.md` files to understand how parameters work.

    - Rerun the notebook as described above from the beginning to re-load the updated configuration settings and run a full workflow with new parameters.

## **Access Workflow Results**

1.  There are a few different ways you can access and examine the workflow results during or after a workflow run:

    - To view the outputs within the JupyterLab web interface, navigate to the `/output_data` directory in the left side panel and click through the files and folders, or open a new workspace tab and use the built-in terminal to view files.

    - To view the outputs within the notebook itself, you can access and view attributes of a dataset or analysis by creating a new cell and executing the command `<object>.<attribute>`. For example, running a cell with `tx_dataset.normalized_data` will show the transcriptomics count table (a dataframe) after all dataset normalization steps, or `mx_dataset.linked`_metadata will show the metabolomics metadata table after it has been linked to the other datasets.

    _Note_: Each cell that produces viewable output will print the name of the `attribute` to the notebook standard output for your reference.. But to see the names of all possible attributes that you can view for each object (dataset or analysis), run a cell with the command `vars(<object>).keys()`. For example, `vars(analysis).keys()` or `vars(mx_dataset).keys()`.

    - To view the outputs on your system (which is mounted to the docker container), navigate to `$INTEGRATION_DIR/project_data/output_data` directory (see _setup.md_ if you do not have this directory) and view output files as you would normally on your local filesystem.

## **Persistent Configuration and Motebook Files**

*   The default config files will contain a set of parameters that can be used right from the outset, but these can be changed during your analysis and exploration of the data. Each time the config parameters are changed and the workflow is run, a cell in the notebook will save the configuration file to a persistent directory in `/output_data/<your_project_name>/configs`. This configuration file can then be accessed in the future using the hashes if you want to recreate the results from that particular run (see the section above).

*   Similarly, the notebook itself (which includes some logging/error printouts, plots, a configuration file path, and other cells you may have added) will be saved to a persistent directory in `/output_data/<your_project_name>/notebooks` to revisit.

## **Exiting the Workflow Container**

1.  When you are finished running the workflow and producing outputs, it is best to close it down gracefully rather than simply close your browser window. There are several methods to properly close down the docker container, any of which will work equally well:

    - Recommended: To close with the running terminal/console, open the window that is running the container (where you ran the `… docker compose …` command during instructions in _setup.md_) and press ctrl/cmd-C on your keyboard.

    - To close with the Docker Desktop app, open the app and click on the Containers tab in the top left menu bar. Find the container named “jgi-integration” in the main panel and click the Stop button (black square) in the Actions column.

    - To close with another terminal/console, open a new window and stop the container with a single command:

      `> docker stop jgi-integration`

## **Notes on JupyterLab**

*   You will not be able to delete directories on the JupyterLab file browser (the left navigation panel), so if you need to remove a directory for any reason you can do so from your local file browser (for the directories mounted to your system) or via the built-in JupyterLab terminal with `rm -rf <directory>`.

*   The default working directory for JupyterLab is `/home/joyvan/work`, which throughout these instructions is referred to as the root directory `/` – this is set by the docker image and should not be changed for security purposes.

*   Details about how to navigate the JupyterLab interface can be found on the [Jupyter documentation page](https://docs.jupyter.org/en/latest/).