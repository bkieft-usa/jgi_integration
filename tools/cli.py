"""Command-line interface for the JGI integration workflow.

    python -m tools.cli list            --results-dir DIR
    python -m tools.cli run             --config-dir DIR [--data-processing-hash H --analysis-hash H]
    python -m tools.cli build-explorer  (--analysis-dir DIR | --project-dir DIR -d H -a H)
    python -m tools.cli explore         (--bundle DIR | --analysis-dir DIR | --project-dir DIR -d H -a H)

``run`` executes every pipeline stage headlessly (figures are written to disk, never
displayed) and finishes by building the explorer bundle. ``explore`` serves the
interactive Dash explorer for one run, building its bundle first if needed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

# Headless plotting must be configured before any tools module imports pyplot.
import matplotlib

matplotlib.use("Agg")

log = logging.getLogger("jgi_integration")


def _setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s", handlers=handlers, force=True)
    _quiet_module_handlers()


def _quiet_module_handlers() -> None:
    """tools modules attach their own colored stdout handler at import; route them through the root
    logger instead so each line appears once (and also reaches --log-file)."""
    for name in ("tools.objects", "tools.helpers", "tools.preprocessing", "tools.pipeline"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


def _silence_notebook_display() -> None:
    """IPython ``display()`` calls in helpers print object reprs outside a notebook."""
    import tools.helpers as hlp
    hlp.display = lambda *args, **kwargs: None


# ---- run identification -----------------------------------------------------------

def resolve_analysis_dir(args) -> str:
    if getattr(args, "analysis_dir", None):
        path = os.path.abspath(args.analysis_dir)
    elif getattr(args, "project_dir", None) and args.data_processing_hash and args.analysis_hash:
        path = os.path.join(os.path.abspath(args.project_dir), f"Dataset_Processing--{args.data_processing_hash}",
                            f"Analysis--{args.analysis_hash}")
    else:
        raise SystemExit("Identify the run with --analysis-dir, or --project-dir with "
                         "--data-processing-hash and --analysis-hash.")
    if not os.path.isdir(path):
        raise SystemExit(f"Analysis directory not found: {path}")
    return path


def run_context(analysis_dir: str) -> dict:
    """Configs and identifiers for a run, read from its persistent configs."""
    from tools.runs import infer_grouping_method, run_config_status
    proc_dir = os.path.dirname(analysis_dir)
    project_dir = os.path.dirname(proc_dir)
    dp = os.path.basename(proc_dir).split("--", 1)[-1]
    an = os.path.basename(analysis_dir).split("--", 1)[-1]
    cfg, ok = run_config_status(project_dir, dp, an)
    project_cfg = cfg.get("project", {}) or {}
    analysis_cfg = (cfg.get("analysis") or {}).get("analysis", {}) or {}
    datasets_cfg = (cfg.get("data_processing") or {}).get("datasets", {}) or {}
    dataset_names = (project_cfg.get("project") or {}).get("dataset_list") or list(datasets_cfg) or None
    grouping = analysis_cfg.get("feature_grouping", {}) or {}
    if not ok:
        log.warning(f"Saved configs for {os.path.basename(proc_dir)}/{os.path.basename(analysis_dir)} are missing "
                    "or do not hash to this run's name; grouping settings are inferred from the outputs.")
        grouping = {"method": infer_grouping_method(analysis_dir), "params": {}}
        analysis_cfg = {k: v for k, v in analysis_cfg.items() if k == "replicate_pairing"}
    return {
        "dataset_names": dataset_names,
        "requested_variables": (project_cfg.get("user_settings") or {}).get("variable_list") or [],
        "datasets_config": datasets_cfg,
        "run_info": {
            "project_name": (project_cfg.get("user_settings") or {}).get("project_name") or os.path.basename(project_dir),
            "data_processing_hash": dp,
            "analysis_hash": an,
            "replicate_pairing": analysis_cfg.get("replicate_pairing"),
            "grouping_method": grouping.get("method"),
            "grouping_params": grouping.get("params", {}),
        },
    }


def _load_explorer_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg.get("explorer", cfg)


# ---- commands --------------------------------------------------------------------

def cmd_list(args) -> int:
    import pandas as pd
    from tools.runs import list_runs
    runs = list_runs(args.results_dir)
    if runs.empty:
        print(f"No runs found under {args.results_dir}")
        return 0
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_colwidth", 60):
        cols = [c for c in runs.columns if c != "analysis_dir" or args.paths]
        print(runs[cols].to_string(index=False))
    return 0


def cmd_run(args) -> int:
    _setup_logging(args.log_file, args.verbose)
    _silence_notebook_display()
    import tools.objects as objs
    from tools.pipeline import run_pipeline
    _quiet_module_handlers()
    if (args.data_processing_hash is None) != (args.analysis_hash is None):
        raise SystemExit("--data-processing-hash and --analysis-hash must be given together.")
    project = objs.Project(config_dir=args.config_dir, data_processing_hash=args.data_processing_hash,
                           analysis_hash=args.analysis_hash, overwrite=args.overwrite)
    skip = list(args.skip or [])
    if args.no_explorer:
        skip.append("build_explorer")
    timings = run_pipeline(project, start_at=args.start_at, stop_after=args.stop_after, skip=skip,
                           explorer_config=_load_explorer_config(args.explorer_config), argv=sys.argv)
    log.info(f"Run complete: Dataset_Processing--{project.data_processing_hash} / Analysis--{project.analysis_hash}")
    log.info(f"Stage timings (s): {timings}")
    import shutil
    exe = "jgi-integrate" if shutil.which("jgi-integrate") else "python -m tools.cli"
    print(f"\nExplore this run with:\n  {exe} explore --project-dir {project.project_dir} "
          f"-d {project.data_processing_hash} -a {project.analysis_hash}")
    return 0


def cmd_build_explorer(args) -> int:
    _setup_logging(args.log_file, args.verbose)
    from tools.explorer_bundle.build import build_explorer_bundle
    analysis_dir = resolve_analysis_dir(args)
    ctx = run_context(analysis_dir)
    explorer_cfg = _load_explorer_config(args.explorer_config)
    if args.n_permutations is not None:
        explorer_cfg["n_permutations"] = args.n_permutations
    out = build_explorer_bundle(analysis_dir, explorer_config=explorer_cfg, force=args.force, **ctx)
    print(f"Explorer bundle: {out}")
    return 0


def cmd_explore(args) -> int:
    _setup_logging(None, args.verbose)
    from tools.explorer_bundle.build import bundle_is_current, build_explorer_bundle
    if args.bundle:
        bundle_dir = os.path.abspath(args.bundle)
        if not os.path.exists(os.path.join(bundle_dir, "manifest.json")):
            raise SystemExit(f"No explorer bundle (manifest.json) in {bundle_dir}")
    else:
        analysis_dir = resolve_analysis_dir(args)
        bundle_dir = os.path.join(analysis_dir, "explorer")
        if args.rebuild or not bundle_is_current(analysis_dir):
            log.info("Explorer bundle missing or out of date; building it now (one-time cost).")
            build_explorer_bundle(analysis_dir, explorer_config=_load_explorer_config(args.explorer_config),
                                  force=True, **run_context(analysis_dir))
    from explorer.app import create_app
    app = create_app(bundle_dir)
    from tools.explorer_launch import browser_url, default_host
    host = args.host or default_host()
    print(f"\nExplorer running at {browser_url(host, args.port)}  (Ctrl+C to stop)\n", flush=True)
    app.run(host=host, port=args.port, debug=args.debug)
    return 0


# ---- parser ----------------------------------------------------------------------

def _add_run_id_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--analysis-dir", help="Path to a Dataset_Processing--*/Analysis--* directory")
    p.add_argument("--project-dir", help="<results_path>/<project_name> (use with the two hashes)")
    p.add_argument("-d", "--data-processing-hash", help="Dataset_Processing hash (8 hex characters)")
    p.add_argument("-a", "--analysis-hash", help="Analysis hash (8 hex characters)")


def build_parser() -> argparse.ArgumentParser:
    from tools.runs import STAGE_NAMES
    parser = argparse.ArgumentParser(prog="jgi-integrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List runs (hash pairs) under a results directory")
    p.add_argument("--results-dir", required=True, help="project.results_path from project.yml")
    p.add_argument("--paths", action="store_true", help="Also print analysis directory paths")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("run", help="Run the full workflow headlessly (ends by building the explorer bundle)")
    p.add_argument("--config-dir", required=True,
                   help="Directory with project.yml, data_processing.yml, analysis.yml; or a project's "
                        "persistent configs/ directory when resuming with hashes")
    p.add_argument("-d", "--data-processing-hash", help="Resume/re-run a saved configuration")
    p.add_argument("-a", "--analysis-hash", help="Resume/re-run a saved configuration")
    p.add_argument("--overwrite", action="store_true", help="Recompute stages even if outputs exist")
    p.add_argument("--start-at", choices=STAGE_NAMES, help="First stage to run (earlier outputs load from disk)")
    p.add_argument("--stop-after", choices=STAGE_NAMES, help="Last stage to run")
    p.add_argument("--skip", nargs="*", choices=STAGE_NAMES, help="Stages to skip")
    p.add_argument("--no-explorer", action="store_true", help="Do not build the explorer bundle")
    p.add_argument("--explorer-config", help="Optional explorer.yml (not part of the run hash)")
    p.add_argument("--log-file", help="Also write logs to this file")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("build-explorer", help="(Re)build the explorer bundle for an existing run")
    _add_run_id_args(p)
    p.add_argument("--explorer-config", help="Optional explorer.yml")
    p.add_argument("--n-permutations", type=int, help="Permutations for agreement null distributions")
    p.add_argument("--force", action="store_true", help="Rebuild even if the bundle is current")
    p.add_argument("--log-file")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_build_explorer)

    p = sub.add_parser("explore", help="Launch the interactive explorer in a browser")
    _add_run_id_args(p)
    p.add_argument("--bundle", help="Path to a copied explorer/ bundle directory (no other run files needed)")
    p.add_argument("--explorer-config", help="Optional explorer.yml (used if the bundle must be built)")
    p.add_argument("--rebuild", action="store_true", help="Rebuild the bundle before launching")
    p.add_argument("--host", default=None,
                   help="Interface to bind. Default: all interfaces inside Docker (reach it through the "
                        "published port), otherwise localhost only. Env JGI_EXPLORER_HOST overrides the default.")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--debug", action="store_true", help="Dash debug mode (hot reload, error overlay)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_explore)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
