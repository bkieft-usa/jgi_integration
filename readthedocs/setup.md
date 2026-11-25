# **Setting Up the JGI Integration Workflow**

## **Introduction**

This guide covers the current input-data layout and how to launch the workflow. The container reads project files from `input_data` and writes generated tables, plots, and other results to the sibling `output_data` directory. Data are staged locally before running; the workflow does not download transcriptomics or metabolomics source data during setup.

The examples use Unix-style paths. Substitute your own host path where needed.

## **Prerequisites**

The workflow runs on macOS and Linux.

- **macOS:** install [Docker Desktop](https://www.docker.com/products/docker-desktop).
- **Linux:** install [Docker Engine](https://docs.docker.com/engine/install/) with the Compose plugin (`docker compose`), or Docker Desktop.
- **Windows:** use WSL. Install WSL with an Ubuntu distribution (`wsl --install` in PowerShell), then Docker Desktop with WSL integration turned on for that distribution. Keep the project folder inside the Linux file system (e.g. `~/project_data`, not `/mnt/c/...`) and follow the Linux instructions from an Ubuntu terminal. There is no native Windows image.

In Docker Desktop, give Docker enough memory: **Settings → Resources → Memory**, at least 12 GB if your computer has 16 GB. With less, large runs can be stopped partway through for lack of memory.

[Cytoscape Desktop](https://cytoscape.org/download.html) is optional for opening generated `.graphml` network files.

## **Input Directory Layout**

Keep the supplied project and container files in place. The `raw_data` subfolder names must agree with each dataset's `dataset_dir` in `config/data_processing.yml`; the project config's `raw_data_path` must point to this `raw_data` directory.

```text
project_data/
├── input_data/
│   ├── config/
│   │   ├── project.yml
│   │   ├── data_processing.yml
│   │   └── analysis.yml
│   ├── docker-compose.yml
│   ├── start.command                 # starts the container and opens JupyterLab
│   ├── link_table.csv
│   └── raw_data/
│       ├── transcriptomics/          # dataset_dir for datasets.tx
│       │   ├── counts.csv
│       │   ├── raw_metadata.csv
│       │   ├── genes.gff3             # recommended/needed for ID mapping
│       │   └── *_annotation_table.tsv # organism-specific; see below
│       └── metabolomics/              # dataset_dir for datasets.mx
│           └── <chromatography>/      # e.g. HILIC, C18
│               ├── *peak-height*.csv
│               ├── *_metadata.tab
│               └── <library-folder>/
│                   └── *library-results.tsv
└── output_data/                       # created/filled by workflow
```

The names `transcriptomics` and `metabolomics` are the defaults in the example data-processing config, not fixed API names. If you change either `dataset_dir`, use that exact folder name beneath `raw_data`. Each chromatography folder name must match an entry in `datasets.mx.chromatography`.

## **Transcriptomics Input**

The current workflow reads `raw_data/<tx dataset_dir>/counts.csv` as a comma-separated file. Its first column contains gene identifiers; all remaining columns are numeric sample counts. The workflow renames the first column to `GeneID` and adds the `tx_` prefix to IDs that do not already have it. Sample-column names are the raw TX sample names used in the link table.

Place the staged, non-empty transcriptomics sample metadata at `raw_data/<tx dataset_dir>/raw_metadata.csv`. It is read as CSV; keep source metadata fields, including `APID` when available. This file is required even though sample-to-sample integration is driven by the link table.

### **Transcript Annotation Files by Origin**

Set `project.genome_type` in `config/project.yml` to `microbe`, `plant`, or `algal`. For bacterial data use `microbe`. `metagenome` annotation processing is not implemented.

All annotation tables are tab-separated (`.tsv`) and must be placed directly in the transcriptomics raw-data directory. At least one supported annotation table and a matching GFF3 file are required. The first column header in every annotation table must exactly match the first column header in `counts.csv`; its values must use the same identifiers. That header also selects the GFF3 attribute used as the transcriptomics ID. The attribute must be present on the organism's supported GFF3 records: `CDS` for microbes, `gene` for algae, and `mRNA` for plants.

| Origin / `genome_type` | Supported annotation files and required source columns | GFF3 mapping |
|---|---|---|
| Bacteria / `microbe` | Filenames containing `cog_annotation_table`: the counts identifier, `cog_id`, `cog_name`; `ipr_annotation_table`: the counts identifier, `iprid`, `iprdesc`, `go_info`; `kegg_annotation_table`: the counts identifier, `ko_id`, `ko_name`; `pfam_annotation_table`: the counts identifier, `pfam_id`, `pfam_name`; `tigrfam_annotation_table`: the counts identifier, `tigrfam_id`, `tigrfam_name`. | The selected identifier attribute must be on GFF3 `CDS` records; `product` is used as the display name. |
| Algal / `algal` | `go_annotation_table`: the counts identifier, `gotermId`, `goName`, `gotermType`, `goAcc`; `ipr_annotation_table`: the counts identifier, `iprId`, `iprDesc`; `kegg_annotation_table`: the counts identifier, `ecNum`, `definition`; `kog_annotation_table`: the counts identifier, `kogid`, `kogdefline`. | The selected identifier attribute must be on GFF3 `gene` records; `product_name` is used as the display name. |
| Plant / `plant` | `kegg_annotation_table.tsv` with the counts identifier as its first column; optional annotation columns include `Pfam`, `Panther`, `KOG`, `KEGG/ec`, `KO`, `GO`, `Best-hit-arabi-name`, `arabi-symbol`, and `arabi-defline`. | The selected identifier attribute must be on GFF3 `mRNA` records; `Name` is used as the display name. |

Keep the remaining organism-specific source headers shown above. Do not retain a leading `#` on the identifier header unless the `counts.csv` first-column header and the GFF3 attribute use that exact name. The annotation pipeline validates that annotation transcript IDs overlap the IDs in the counts table; mismatched identifiers can stop dataset creation.

## **Metabolomics Input**

The current MX dataset uses untargeted peak-height data. It searches under `raw_data/<mx dataset_dir>/<chromatography>/` for CSV result files. Configure `chromatography` as one folder name or a list, and `polarity` as `POS`, `NEG`, or a list such as `[POS, NEG]`. Filenames must end in `_<positive|negative>_peak-height-filtered-3x-exctrl.csv`.

```text
raw_data/metabolomics/HILIC/
├── study_positive_peak-height-filtered-3x-exctrl.csv
├── study_negative_peak-height-filtered-3x-exctrl.csv
├── study_positive_metadata.tab
├── study_negative_metadata.tab
└── library-results/
   ├── positive_library-results.tsv
   └── negative_library-results.tsv
```

The result CSV should be the vendor/JGI peak-height table: its first column is the compound/scan identifier, with one numeric column per sample. Keep the `row m/z` and `row retention time` columns if present; the loader removes them. Sample headers should identify the corresponding `.mzML` files; the loader removes the `.mzML` suffix and ` Peak height` text before matching them to the link table. When multiple chromatography/polarity combinations are configured, feature IDs are suffixed with chromatography and polarity (for example, `mx_123_hilic_neg`), and normalization is performed separately for each combination.

Place polarity metadata as tab-separated `*_metadata.tab` files anywhere beneath the chromatography folder. A single `raw_data/<mx dataset_dir>/raw_metadata.csv` is accepted as a compatibility input if no metadata TSVs are present. Metadata filenames are normalized by removing `.mzML`; they must therefore resolve to the sample names represented in the peak-height table and link table.

FBMN compound annotations are optional and are read from `*/*library-results.tsv` under the MX raw-data directory. The table must include `#Scan#` for matching IDs; fields such as `Compound_Name`, `INCHI`, `InChiKey`, `molecular_formula`, and class fields supply annotations. Without a matching library-results row, the metabolite is retained but its compound annotation fields are unassigned.

## **Proteomics Input (optional)**

The `px` dataset uses `raw_data/<px dataset_dir>/peak-height.csv` (default `dataset_dir: proteomics`) with the same layout as the other tables: first column is the protein/feature ID, remaining columns are samples. Place a `raw_metadata.csv` beside it as for transcriptomics. Annotation files and formats are identical to the transcriptomics ones (see above) and depend on `project.genome_type`; the first column header of each annotation file must equal the first column header of `peak-height.csv`. Add a `px` column to the link table and a `px` block to `data_processing.yml`, then create the dataset with `objs.PX(project)`.

## **Metadata Link Table**

Set `project.link_table` in `config/project.yml` to the master table, normally `/home/jovyan/work/input_data/link_table.csv`. This CSV/TSV maps raw sample names from each quantitative table to a shared sample name. Use dataset names (`tx`, `mx`) as the column headers, not the raw-data folder names.

```csv
unique_group,tx,mx,treatment,timepoint
sample_01,library_001,study_01,control,day0
sample_02,library_002,study_02,treated,day0
sample_03,library_003,,control,day7
```

Requirements:

- Include one column for every configured dataset, such as `tx` and `mx`. When multiple MX chromatography/polarity combinations are configured, use one MX column per combination in the form `mx_<chromatography>_<polarity>`, e.g. `mx_hilic_neg`, `mx_c18_pos`; chromatography and `pos`/`neg` are lowercase.
- Include one shared-name column: `unique_group` (recommended), `shared_sample`, `shared_name`, or `sample`. Every row must have a nonblank shared name.
- Put the raw quantitative-table sample identifier in the dataset column. It must exactly match a sample header after the loader's normalization described above. For MX, this normally means omitting `.mzML` and ` Peak height`; verify the final normalized peak-height column names.
- Leave a dataset cell blank when that shared sample is absent from that datatype. A raw sample name may appear only once in each dataset column.
- Remaining columns are sample categories/metadata. Include variables needed for grouping and analysis, and list the desired variables (including `group`) in `user_settings.variable_list` in `project.yml`.

The link table is what renames raw data sample columns to shared names and aligns the datasets. With overlap-only integration enabled, shared names must overlap across the datasets to be integrated.

## **Configuration Paths**

The example container paths are:

```yaml
project:
  results_path: /home/jovyan/work/output_data
  raw_data_path: /home/jovyan/work/input_data/raw_data
  link_table: /home/jovyan/work/input_data/link_table.csv
```

In `config/data_processing.yml`, each configured dataset has a `dataset_dir` that selects its raw-data subfolder. The example values are `transcriptomics` for `tx` and `metabolomics` for `mx`. Also set `datasets.mx.chromatography` and `datasets.mx.polarity` to match the files staged above. See [project configuration parameters](project_config_parameters_explained.md) for the complete config reference.

## **Step 1: Prepare the Project Folder**

Create a project directory and place the supplied `project_data` archive contents there, or create the layout described above. Do not move `docker-compose.yml`, `start.command` or the `config` directory out of `input_data`. `output_data` is created next to `input_data` on first start if it does not exist.

If you use your own compose file, publish both ports, `8888` for JupyterLab and `8050` for the results explorer:

```yaml
    ports:
      - "127.0.0.1:8888:8888"   # JupyterLab
      - "127.0.0.1:8050:8050"   # integration explorer
```

## **Step 2: Start the Container**

**macOS:** double-click `start.command` in `input_data` in Finder. If macOS refuses to open it because it was downloaded from the internet, either go to **System Settings → Privacy & Security** and click **Open Anyway**, or run it from Terminal as `bash start.command` (see below).

**Linux (and Windows through WSL), or a macOS terminal:**

```sh
cd /path/to/project_data/input_data
./start.command
```

`start.command` starts Docker Desktop if it is not running, downloads the newest image (`ghcr.io/bkieft-usa/jgi-integration:latest`; Docker picks the build for your computer's CPU), starts the container and opens JupyterLab in your browser. If no tab opens, copy the link it prints into your browser.

Leave the `start.command` window open while you work. To finish, press **Ctrl+C** in that window, or close it; either stops the container. Your files in `input_data` and `output_data` are kept.

It stops with a message if Docker is not installed, or if another program is already using port 8888 or 8050 (for example another JupyterLab). Stop that program and run it again.

To use a specific published image instead of the newest, set its tag, e.g. `tag=1.2.0 ./start.command` for a release or `tag=sha-1a2b3c4` for a single commit (tags are listed on the [package page](https://github.com/bkieft-usa/jgi_integration/pkgs/container/jgi-integration)).

### For developers

**Working on the code.** `./start.command --dev` starts your local `ghcr.io/bkieft-usa/jgi-integration:dev` image with `dev/docker-compose-dev.yml` (copy it from `docker/docker-compose-dev.example.yml`). The dev image, built from `docker/Dockerfile-dev`, contains only the Python environment; the compose file mounts your `jgi_integration` checkout into the container, so code edits take effect without rebuilding. Build it once, and again only when `venv/pyproject.toml` or `venv/uv.lock` change:

```sh
docker build -t ghcr.io/bkieft-usa/jgi-integration:dev -f docker/Dockerfile-dev .
```

**Releasing.** `.github/workflows/docker-image.yml` builds `docker/Dockerfile` for linux/amd64 and linux/arm64 on every push to `main` and publishes it as `:latest` and `:sha-<commit>`. Pushing a git tag `vX.Y.Z` also publishes `:X.Y.Z` and `:X.Y`. The image contains the code of the commit that was pushed, and each run's `run_manifest.json` records that commit. Images are published to GitHub Container Registry (`ghcr.io/bkieft-usa/jgi-integration`) with the workflow's built-in token, so no secrets are needed. The package must be public so users can pull it without logging in; set this once on the package page under **Package settings → Change visibility**.

**Packaging a project for users.** Users need only the `project_data` folder; the code comes with the image. In `input_data/`, put a real copy (not a symlink) of `docker/start.command`, executable, and `docker/docker-compose.example.yml` renamed `docker-compose.yml`. Archive it as `.tar.gz`, which keeps `start.command` executable (`-h` stores the files that symlinks point to):

```sh
tar -czhf <project>_project_data.tar.gz project_data
```

Everything in the folder is included, so package a clean copy: without `input_data/dev/` and with an empty `output_data/`.

## **Step 3: Generate the Results**

Everything below happens inside JupyterLab. Choose **one** of the two ways; both run the same code and give identical results.

**Option A: the notebook.** Open `workflow.ipynb` and run all cells, following [run.md](run.md). Figures are shown inline as each stage finishes.

**Option B: the terminal.** Open a terminal in JupyterLab (**File → New → Terminal**) and run:

```bash
jgi-integrate run --config-dir /home/jovyan/work/input_data/config
```

This runs every stage without showing figures (they are saved to `output_data`). You can run it from any folder. When it finishes, it prints the command to explore the run. See [cli.md](cli.md) for partial runs, resuming a run, and running on an HPC cluster or under slurm.

Either way, the last stage precomputes everything the explorer needs into the run's `explorer/` folder.

## **Step 4: Explore the Results**

Start the explorer in one of these ways, then open **http://localhost:8050** in your browser.

**From the notebook.** The last cell of `workflow.ipynb` starts the explorer for the run you just made:

```python
from tools.explorer_launch import start_explorer, stop_explorer
start_explorer(project.analysis_dir)
```

It runs in the background, so you can keep using the notebook, and it keeps running if you restart or close the notebook. Run `stop_explorer()` to stop it.

**From a terminal.** Paste the command printed at the end of `jgi-integrate run`. It has this form:

```bash
jgi-integrate explore --project-dir /home/jovyan/work/output_data/<project_name> -d <data hash> -a <analysis hash>
```

The terminal stays busy while the explorer runs; press **Ctrl+C** to stop it. To find the hashes of earlier runs, run `jgi-integrate list --results-dir /home/jovyan/work/output_data`.

Only one explorer can use port 8050 at a time. To switch runs, stop the current one first (re-running `start_explorer` in the notebook does this for you). See [explorer.md](explorer.md) for what each page shows.
