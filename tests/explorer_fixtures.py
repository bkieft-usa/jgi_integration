"""Synthetic run directories (CSV outputs shaped like a real pipeline run) for explorer tests.

Design: 6 organisms x 4 replicates, organisms nested in 2 lineages; 8 groups of 25 features
(tx and mx mixed) sharing a latent profile. Planted signals:
* group_1 is strongly up in organism o0 (feature- and group-level association);
* KEGG module M00001 is carried by 12 features of group_1 plus 2 elsewhere (enrichment);
* M00099 is spread at random (no enrichment).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ORGS = [f"o{i}" for i in range(6)]
REPS = 4


def make_run(root: str, method: str = "hierarchical_clustering", seed: int = 0, unnamed_index: bool = True) -> str:
    rng = np.random.default_rng(seed)
    samples = [f"{o}-L{1 if i < 3 else 2}-{r}" for i, o in enumerate(ORGS) for r in range(1, REPS + 1)]
    meta = pd.DataFrame({
        "tx": [f"raw_tx_{s}" for s in samples],
        "mx_hilic_pos": [f"raw_mx_{s}" for s in samples],
        "organism": [s.split("-")[0] for s in samples],
        "lineage": [s.split("-")[1] for s in samples],
        "group": [s.rsplit("-", 1)[0] for s in samples],
        "replicate": [int(s.rsplit("-", 1)[1]) for s in samples],
        "unique_group": samples,
    })
    n_groups, per = 8, 25
    feats, labels = [], []
    for g in range(n_groups):
        for k in range(per):
            dt = "tx" if (k % 5) < 3 else "mx"
            feats.append(f"{dt}_g{g}f{k}" if dt == "tx" else f"{dt}_{g}{k:02d}_hilic_pos")
            labels.append(f"group_{g + 1}")
    org_idx = np.array([ORGS.index(o) for o in meta["organism"]])
    latent = rng.normal(0, 1, (n_groups, len(ORGS)))
    log = np.empty((len(feats), len(samples)))
    for i, lab in enumerate(labels):
        g = int(lab.split("_")[1]) - 1
        prof = latent[g][org_idx] * 1.5
        if g == 0:
            prof = prof + 4.0 * (org_idx == 0)
        log[i] = 8 + prof + rng.normal(0, 0.4, len(samples))
    if method == "hdbscan":  # last group becomes noise: no shared profile
        for i, lab in enumerate(labels):
            if lab == f"group_{n_groups}":
                log[i] = 8 + rng.normal(0, 1.0, len(samples))
                labels[i] = "noise"
    log_df = pd.DataFrame(log, index=feats, columns=samples)
    z = log_df.sub(log_df.mean(axis=1), axis=0).div(log_df.std(axis=1), axis=0)

    proc = os.path.join(root, "Proj", "Dataset_Processing--aaaaaaaa")
    a = os.path.join(proc, "Analysis--bbbbbbbb")
    os.makedirs(a, exist_ok=True)
    for ds, ddir, idcol in (("tx", "transcriptomics", "OG"), ("mx", "metabolomics", "CompoundID")):
        os.makedirs(os.path.join(proc, ddir), exist_ok=True)
        sub = log_df[log_df.index.str.startswith(f"{ds}_")].copy()
        sub.index = [i[len(ds) + 1:] for i in sub.index]
        sub.index.name = idcol
        sub.to_csv(os.path.join(proc, ddir, "replicate_filtered_data.csv"))

    zf = z.copy()
    zf.index.name = "features"
    zf.to_csv(os.path.join(a, "integrated_data.csv"))
    zf.to_csv(os.path.join(a, "integrated_data_selected.csv"))
    meta.to_csv(os.path.join(a, "integrated_metadata.csv"), index=unnamed_index)

    # annotations
    ann = pd.DataFrame({"feature_id": feats})
    ann["kegg_module"] = "Unassigned"
    ann["kegg_module_name"] = "Unassigned"
    g1 = [i for i, l in enumerate(labels) if l == "group_1"]
    for i in g1[:12]:
        ann.loc[i, ["kegg_module", "kegg_module_name"]] = ["M00001", "Glycolysis"]
    others = [i for i, l in enumerate(labels) if l not in ("group_1",)]
    for i in rng.choice(others, 2, replace=False):
        ann.loc[i, ["kegg_module", "kegg_module_name"]] = ["M00001", "Glycolysis"]
    for i in rng.choice(len(feats), 30, replace=False):
        cur = ann.loc[i, "kegg_module"]
        ann.loc[i, ["kegg_module", "kegg_module_name"]] = (
            ["M00099", "Random module"] if cur == "Unassigned" else [f"{cur};M00099", f"Glycolysis;Random module"])
    ann["kegg_pathway"] = np.where(np.arange(len(feats)) % 7 == 0, "path:map00010;path:map01100", "Unassigned")
    ann["kegg_pathway_name"] = np.where(ann["kegg_pathway"] != "Unassigned", "Glycolysis;Metabolic pathways",
                                        "Unassigned")
    ann["modelseed_pathway"] = np.where(np.arange(len(feats)) % 5 == 0,
                                        "rn00010 (Glycolysis <i>a</i>);NOPATHWAY_rxn1", "Unassigned")
    is_tx = ann["feature_id"].str.startswith("tx_")
    ann["tx_COG24_CATEGORY_ACC"] = np.where(is_tx, np.where(np.arange(len(feats)) % 2 == 0, "C", "C;;E"), "Unassigned")
    ann["tx_COG24_CATEGORY"] = np.where(is_tx, np.where(np.arange(len(feats)) % 2 == 0, "Energy",
                                                        "Energy;;Amino acids"), "Unassigned")
    ann["tx_AAI_avg"] = np.where(is_tx, rng.uniform(0.5, 1, len(feats)).round(3).astype(str), "Unassigned")
    ann["mx_superclass"] = np.where(~is_tx, np.where(np.arange(len(feats)) % 3 == 0, "Lipids", "Organic acids"),
                                    "Unassigned")
    ann["mx_Compound_Name"] = np.where(~is_tx, [f"compound {i}" for i in range(len(feats))], "Unassigned")
    ann.to_csv(os.path.join(a, "feature_annotation_table.csv"), index=unnamed_index)

    nodes = pd.DataFrame({"group": labels, "datatype_color": "#000", "datatype_shape": "circle"},
                         index=pd.Index(feats, name="node_id"))
    nodes.to_csv(os.path.join(a, "feature_network_node_table.csv"))
    edges = pd.DataFrame(columns=["source", "target", "weight"])
    if method == "network_modules":
        rows = []
        for g in sorted(set(labels)):
            mem = [f for f, l in zip(feats, labels) if l == g]
            for i in range(len(mem) - 1):
                rows.append((mem[i], mem[i + 1], 0.95))
                rows.append((mem[i + 1], mem[i], 0.95))  # duplicates in both directions, as in real runs
        edges = pd.DataFrame(rows, columns=["source", "target", "weight"])
    edges.index.name = "edge_index"
    edges.to_csv(os.path.join(a, "feature_network_edge_table.csv"))

    if method == "hierarchical_clustering":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        Z = linkage(pdist(z.values, metric="correlation"), method="average")
        sdir = os.path.join(a, "grouping_structure")
        os.makedirs(sdir, exist_ok=True)
        pd.DataFrame(Z, columns=["left", "right", "distance", "size"]).to_csv(
            os.path.join(sdir, "hierarchical_linkage.csv"), index=False)
        pd.Series(feats, name="feature_id").to_csv(os.path.join(sdir, "hierarchical_linkage_leaf_ids.csv"), index=False)
        # make saved groups consistent with the linkage cut so the faithful tree is exercised
        cut = fcluster(Z, t=n_groups, criterion="maxclust")
        nodes["group"] = [f"group_{c}" for c in cut]
        # keep the planted group's name: relabel the cluster holding most of group_1's features
        main = pd.Series(cut[:per]).mode().iloc[0]
        relabel = {f"group_{main}": "group_1", "group_1": f"group_{main}"}
        nodes["group"] = nodes["group"].replace(relabel)
        nodes.to_csv(os.path.join(a, "feature_network_node_table.csv"))
        with open(os.path.join(sdir, "hierarchical_linkage.json"), "w") as fh:
            json.dump({"distance_metric": "correlation", "linkage_method": "average", "height_cutoff": None}, fh)
    return a


def run_info(method: str) -> dict:
    return {"project_name": "Proj", "data_processing_hash": "aaaaaaaa", "analysis_hash": "bbbbbbbb",
            "replicate_pairing": "paired", "grouping_method": method,
            "grouping_params": {"metric": "euclidean", "min_cluster_size": 5, "min_samples": 3}}
