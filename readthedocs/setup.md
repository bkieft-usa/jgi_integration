# **Setting Up the JGI Integration Workflow**

## **Introduction**

This guide covers the current input-data layout and how to launch the workflow. The container reads project files from `input_data` and writes generated tables, plots, and other results to the sibling `output_data` directory. Data are staged locally before running; the workflow does not download transcriptomics or metabolomics source data during setup.

The examples use Unix-style paths. Substitute your own host path where needed.

## **Prerequisites**

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop). [Cytoscape Desktop](https://cytoscape.org/download.html) is optional for opening generated `.graphml` network files.

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

The names `transcriptomics` and `metabolomics` are the defaults in the example data-processing config, not fixed API names. If you change either `dataset_dir`, use that exact folder name beneath `raw_data`. The chromatography folder name must match `datasets.mx.chromatography`.

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

The current MX dataset uses untargeted peak-height data. It searches under `raw_data/<mx dataset_dir>/<chromatography>/` for CSV result files. For `polarity: positive` or `negative`, names must end in `_<polarity>_peak-height-filtered-3x-exctrl.csv`; for `polarity: multipolarity`, they must end in `_peak-height-filtered-3x-exctrl.csv`. For example:

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

The result CSV should be the vendor/JGI peak-height table: its first column is the compound/scan identifier, with one numeric column per sample. Keep the `row m/z` and `row retention time` columns if present; the loader removes them. Sample headers should identify the corresponding `.mzML` files; the loader removes the `.mzML` suffix and ` Peak height` text before matching them to the link table. Multipolarity mode can combine positive and negative result files; include polarity in each filename so generated metabolite IDs distinguish the two.

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

- Include one column for every configured dataset, such as `tx` and `mx`.
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

Create a project directory and place the supplied `project_data` archive contents there, or create the layout described above. Do not move `docker-compose.yml` or the `config` directory from `input_data`. Ensure `output_data` exists alongside `input_data` and is writable.

## **Step 2: Launch the Docker Container**

1. Confirm Docker is running with `docker info`.
2. Change to the directory containing `docker-compose.yml`:

   ```sh
   cd /path/to/project_data/input_data
   ```

3. Pull and start the container. Set the architecture for your system: `windows-amd64`, `mac-arm64`, `mac-amd64`, or `linux-amd64`. The Windows image is currently unstable and not recommended.

   ```sh
   tag=<arch> docker-compose -p jgi-integration up -d --force-recreate --pull
   ```

4. In the command output, find the Jupyter Server URL beginning with `http://127.0.0.1:8888/lab` and open it in a browser.

## **Step 3: Run the Workflow in JupyterLab**

Open the workflow notebook and follow [run.md](run.md). If another JupyterLab server is already running, stop it before starting this container to avoid a port collision.
