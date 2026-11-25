"""The integration workflow as an ordered list of stages.

Both ``notebooks/workflow.ipynb`` (one stage per cell, with live figures) and the CLI
(``python -m tools.cli run``, headless) call these same functions, so the two entry
points cannot drift apart.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import tools.helpers as hlp
import tools.objects as objs
import tools.runs as runs
from tools.runs import list_runs, read_run_configs  # noqa: F401  (re-exported for notebooks)

log = logging.getLogger(__name__)

DATASET_CLASSES = {"tx": objs.TX, "mx": objs.MX, "px": objs.PX}


def build_datasets(project: "objs.Project") -> List["objs.Dataset"]:
    """Instantiate one Dataset object per entry in ``project.dataset_list``."""
    datasets = []
    for name in project.data_types:
        if name not in DATASET_CLASSES:
            raise ValueError(f"Unknown dataset '{name}' in project.dataset_list; expected one of {list(DATASET_CLASSES)}")
        datasets.append(DATASET_CLASSES[name](project))
    return datasets


def build_analysis(project: "objs.Project") -> "objs.Analysis":
    return objs.Analysis(project, datasets=build_datasets(project))


# ---- stage functions --------------------------------------------------------------

def stage_link(analysis: "objs.Analysis") -> None:
    """Rename quantitative-table sample columns using the master link table."""
    analysis.link_data()
    for ds in analysis.datasets:
        hlp.plot_simple_histogram(ds.linked_data, plot_title=f"{ds.dataset_name.upper()} data (raw)",
                                  ylog=True, bins=50, output_dir=analysis.output_dir)


def stage_filter(analysis):
    """Validate/detect input scale, then filter rare raw features (presence and magnitude)."""
    analysis.filter_all_datasets()


def stage_normalize(analysis):
    """Sample-normalize (raw input only), impute mx/px zeros, and log-transform."""
    analysis.normalize_all_datasets()


def stage_devariance(analysis):
    """Remove features not impacted by experimentation (low variance)."""
    analysis.devariance_all_datasets()


def stage_replicability(analysis):
    """Remove features with outlier replicates."""
    analysis.replicability_test_all_datasets()


def stage_scale(analysis):
    """Scale each dataset independently at sample or condition resolution."""
    analysis.scale_all_datasets()
    analysis.plot_dataset_distributions(ylog=True, bins=50)


def stage_integrate(analysis):
    """Concatenate scaled matrices into the integrated matrix."""
    analysis.integrate_data()
    variables = analysis.metadata_variables
    meta = analysis._metadata_for_integrated_data()
    hlp.plot_simple_pca(analysis.integrated_data, meta, metadata_variables=variables,
                        title="Integrated Data PCA (all features)", output_dir=analysis.output_dir)
    for ds in analysis.datasets:
        hlp.plot_simple_pca(ds.linked_data, analysis.integrated_metadata.copy(), metadata_variables=variables,
                            title=f"{ds.dataset_name.upper()} linked data PCA", output_dir=analysis.output_dir)


def stage_select_features(analysis):
    """Subset features by variance or list for grouping."""
    analysis.perform_feature_selection()
    analysis.plot_integrated_pca()


def stage_annotate(analysis):
    """Knowledge-driven: attach the full annotation scope to every integrated feature."""
    analysis.annotate_integrated_features()


def stage_group_features(analysis):
    """Data-driven: group features by co-variation across samples/conditions."""
    analysis.group_features()
    _release_worker_pool()


def stage_enrich(analysis):
    """Compare data-driven groups against every annotation type."""
    analysis.enrich_all()


def stage_build_explorer(analysis, explorer_config: Optional[dict] = None):
    """Precompute statistics, embeddings and structure for the interactive explorer."""
    analysis.build_explorer(explorer_config=explorer_config)


STAGES: List[Tuple[str, Callable]] = [
    ("link", stage_link),
    ("filter", stage_filter),
    ("normalize", stage_normalize),
    ("devariance", stage_devariance),
    ("replicability", stage_replicability),
    ("scale", stage_scale),
    ("integrate", stage_integrate),
    ("select_features", stage_select_features),
    ("annotate", stage_annotate),
    ("group_features", stage_group_features),
    ("enrich", stage_enrich),
    ("build_explorer", stage_build_explorer),
]
STAGE_NAMES = [name for name, _ in STAGES]
assert STAGE_NAMES == runs.STAGE_NAMES, "tools.runs.STAGE_NAMES is out of sync with STAGES"


def select_stages(start_at: Optional[str] = None, stop_after: Optional[str] = None,
                  skip: Optional[List[str]] = None) -> List[Tuple[str, Callable]]:
    for label, value in (("start_at", start_at), ("stop_after", stop_after)):
        if value is not None and value not in STAGE_NAMES:
            raise ValueError(f"{label}='{value}' is not a stage; choose from {STAGE_NAMES}")
    i0 = STAGE_NAMES.index(start_at) if start_at else 0
    i1 = STAGE_NAMES.index(stop_after) if stop_after else len(STAGES) - 1
    if i0 > i1:
        raise ValueError(f"start_at '{start_at}' comes after stop_after '{stop_after}'")
    skip = set(skip or [])
    return [s for s in STAGES[i0:i1 + 1] if s[0] not in skip]


def _release_worker_pool() -> None:
    """Shut down joblib's reusable worker pool. Workers started by a parallel step (e.g. the
    correlation in group_features) otherwise stay alive, holding their memory, for the rest of the run."""
    try:
        from joblib.externals.loky import reusable_executor
    except ImportError:
        return
    executor = getattr(reusable_executor, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


def run_pipeline(project: "objs.Project", start_at: Optional[str] = None, stop_after: Optional[str] = None,
                 skip: Optional[List[str]] = None, explorer_config: Optional[dict] = None,
                 argv: Optional[List[str]] = None) -> Dict[str, float]:
    """Run the selected stages in order and write the run record. Returns stage timings.

    Earlier stages' outputs are loaded from disk when present (hash-tagged directories),
    so ``start_at`` can resume a run, e.g. on a different slurm job.
    """
    analysis = build_analysis(project)
    timings: Dict[str, float] = {}
    for name, fn in select_stages(start_at, stop_after, skip):
        log.info(f"===== Stage: {name} =====")
        t0 = time.time()
        if name == "build_explorer":
            fn(analysis, explorer_config=explorer_config)
        else:
            fn(analysis)
        hlp.plt.close("all")
        timings[name] = round(time.time() - t0, 2)
        log.info(f"===== Stage {name} finished in {timings[name]:.1f}s =====")
    project.save_run_record(stage_timings=timings, argv=argv)
    return timings

