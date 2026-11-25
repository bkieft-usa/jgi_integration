# Running the workflow from the command line

The workflow can be run two ways, and both produce identical results:

- **CLI:** `python -m tools.cli run`. Headless; figures are written to disk. Use it on a laptop, an HPC node or in a slurm job.
- **Notebook:** `notebooks/workflow.ipynb`. One stage per cell, with figures shown inline.

Both call the same stage functions in `tools/pipeline.py`.

Every command below can also be written `bin/jgi-integrate <command>`. Inside the container, `jgi-integrate` is on the `PATH`. **Use `jgi-integrate` in JupyterLab terminals:** they activate conda's base Python, which lacks the workflow's packages, so a plain `python -m tools.cli` fails with `ModuleNotFoundError`. `jgi-integrate` always uses the workflow's own environment.

## Commands

| Command | What it does |
|---|---|
| `list --results-dir DIR` | Lists every run (project, data-processing hash, analysis hash, grouping method, whether enrichment and the explorer bundle exist). `configs_ok = False` means the saved configs are missing or do not hash to the run's name. |
| `run --config-dir DIR` | Runs all stages, finishing with the explorer bundle. |
| `build-explorer` | (Re)builds only the explorer bundle for an existing run. |
| `explore` | Starts the interactive explorer in your browser. Builds the bundle first if it is missing or older than the run's outputs. |

### `run`

```bash
python -m tools.cli run --config-dir /home/jovyan/work/input_data/config
```

| Option | Meaning |
|---|---|
| `--config-dir` | Folder with `project.yml`, `data_processing.yml`, `analysis.yml`. When resuming with hashes, use `<results_path>/<project_name>/configs`. |
| `-d/--data-processing-hash`, `-a/--analysis-hash` | Re-run or resume a saved configuration (both required together). |
| `--start-at STAGE`, `--stop-after STAGE` | Run part of the workflow. Earlier stages' outputs are loaded from disk. |
| `--skip STAGE ...`, `--no-explorer` | Skip stages (e.g. the explorer bundle). |
| `--overwrite` | Recompute stages even if their outputs exist. |
| `--explorer-config FILE` | Optional `explorer.yml` (not part of the run hash). |
| `--log-file FILE`, `-v` | Log file and verbose logging. |

Stages, in order: `link, filter, normalize, devariance, replicability, scale, integrate, select_features, annotate, group_features, enrich, build_explorer`.

When the run finishes, it prints the `explore` command for that run.

### `explore`

```bash
# by run identity
python -m tools.cli explore --project-dir /path/output_data/<project> -d 5d861c04 -a d38b6cf2
# or a copied bundle folder (nothing else from the run is needed)
python -m tools.cli explore --bundle /path/to/explorer
```

Then open `http://localhost:8050`. Use `--port` to change the port. The terminal stays busy while the explorer runs; press Ctrl+C to stop it.

**Where it listens.** Inside Docker, the explorer listens on all interfaces, so the published port works: the container must publish it (`-p 8050:8050`, or `"8050:8050"` under `ports:` in compose; the supplied compose file does this). Everywhere else, including apptainer/shifter on a shared HPC node, it listens on localhost only, so other users cannot reach it. Override this with `--host`, or by setting `JGI_EXPLORER_HOST`.

**From a notebook** (runs in the background and survives kernel restarts):
```python
from tools.explorer_launch import start_explorer, stop_explorer
start_explorer("/path/output_data/<project>/Dataset_Processing--<D>/Analysis--<A>")
stop_explorer()
```

## Where to run it

**Laptop (JupyterLab container).** Start the container with `./start.command`, open a terminal in JupyterLab (File → New → Terminal) and run `jgi-integrate run --config-dir /home/jovyan/work/input_data/config` from any folder. See [setup.md](setup.md), steps 2–4.

**HPC login node.** Use a login node only for `list` and for small runs; most centers limit CPU and memory there. Hierarchical clustering of 25,000 features needs about 5–10 GB of memory.

**Slurm job.** `scripts/slurm_run.sbatch` runs the same container image with apptainer/singularity, podman-hpc or shifter (set `RUNTIME`):
```bash
RUNTIME=apptainer sbatch scripts/slurm_run.sbatch /path/input_data /path/output_data
EXTRA_ARGS="--stop-after group_features" sbatch scripts/slurm_run.sbatch ...   # split a long run
```
Submit from outside the container; the job starts the container itself.

**Viewing results from an HPC run.** You can either:
- copy the small `explorer/` folder to your laptop and run `explore --bundle`, or
- run `explore` on the cluster and tunnel the port with `ssh -L 8050:localhost:8050 <host>`, then open `http://localhost:8050` locally.

## Reproducibility

- Output folders are named by a hash of `data_processing.yml` and `analysis.yml`.
- `configs/` keeps the exact config text that was hashed when the run started.
- `Analysis--<A>/run_manifest.json` records the command, workflow git commit, host, slurm job ID and the time each stage took.
