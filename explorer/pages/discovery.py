"""Unannotated feature discovery: features that may take part in a biological process but carry no
database annotation, found through the data-driven groupings ("guilt by association")."""
from __future__ import annotations

from urllib.parse import parse_qs

import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
from dash import Input, Output, State, dcc, html, no_update

from explorer import components as ui
from explorer import theme as th
from tools.explorer_bundle import network as net

MAX_GRAPH = 160
UNANNOTATED = [{"label": "no annotation of any type", "value": "any"},
               {"label": "not annotated with this annotation type", "value": "type"}]
LINKS = [{"label": "directly connected in the network", "value": "direct"},
         {"label": "in the same data-driven group", "value": "group"}]


def _param(search, key):
    return parse_qs((search or "").lstrip("?")).get(key, [None])[0]


def layout(b, params):
    term, fid = params.get("term"), params.get("feature")
    tab = "feature" if fid else ("term" if term else "term")
    return html.Div([
        ui.page_header("Unannotated Feature Discovery",
                       "Features with no database annotation can still be placed in a biological process by the "
                       "company they keep: if a feature's network neighbours or group-mates share an annotation, that "
                       "annotation is a hypothesis for the feature. These are hypotheses to test, not annotations."),
        dbc.Tabs([
            dbc.Tab(label="Start from an annotation", tab_id="term", children=html.Div([
                html.P("Pick a pathway, module, reaction, KO, EC number, COG or compound class. Unannotated features "
                       "linked to the features that carry it are listed as candidates.", className="small mt-2"),
                html.Div([
                    ui.labeled("Annotation term", ui.dropdown(
                        "dc-term", [{"label": b.term_label(term, 120), "value": term}] if term else [], term,
                        placeholder="Search terms…", clearable=True, searchable=True, style={"minWidth": "460px"})),
                    ui.labeled("Unannotated means", ui.dropdown("dc-unann", UNANNOTATED, "any",
                                                                style={"minWidth": "290px"})),
                    ui.labeled("Linked by", ui.dropdown("dc-link", LINKS, "direct", style={"minWidth": "290px"})),
                ], className="controls"),
                html.Div(id="dc-term-body"),
            ])),
            dbc.Tab(label="Start from a feature", tab_id="feature", children=html.Div([
                html.P("Pick any feature (annotated or not) to see the annotations of the features closest to it, and "
                       "how strongly they agree.", className="small mt-2"),
                html.Div([ui.labeled("Feature", ui.dropdown(
                    "dc-feature", [{"label": b.feature_label(fid), "value": fid}] if fid else [], fid,
                    placeholder="Search a feature ID or name…", clearable=True, searchable=True,
                    style={"minWidth": "460px"}))], className="controls"),
                html.Div(id="dc-feature-body"),
            ])),
            dbc.Tab(label="All candidates", tab_id="all", children=html.Div([
                html.P("Every feature lacking an annotation type whose direct network neighbours carry one of that "
                       "type's terms more often than expected (neighbour-voting test, FDR within each annotation "
                       "type). The last column shows whether the term is also enriched in the feature's group.",
                       className="small mt-2"),
                html.Div([
                    ui.labeled("Annotation type", ui.dropdown(
                        "dc-all-type", [{"label": "All", "value": "all"}] +
                        [{"label": a["label"], "value": a["key"]} for a in b.ann_types], "all",
                        style={"minWidth": "240px"})),
                    ui.labeled("Data type", ui.dropdown("dc-all-dtype", [{"label": "All", "value": "all"}] +
                                                        [{"label": th.DTYPE_LABEL.get(d, d), "value": d} for d in b.dtypes],
                                                        "all")),
                    dbc.Switch(id="dc-all-strict", label="Only features with no annotation of any type", value=True,
                               className="mb-1 small"),
                ], className="controls"),
                html.Div(id="dc-all-body"),
            ])),
        ], id="dc-tabs", active_tab=tab, className="mt-1"),
    ])


def _annotated_ids(b, mode, ann_type):
    if mode == "type":
        return set(b.ann.loc[b.ann["ann_type"] == ann_type, "feature_id"])
    return set(b.features.index[b.features["is_annotated"]])


def _scope(b, ann_type):
    return next((a.get("dtype_scope") for a in b.ann_types if a["key"] == ann_type), None)


def _term_candidates(b, term, mode, link):
    """Candidate table and the term's network features for one term."""
    at, tid = term.split("::", 1)
    term_feats = b.term_members(term)
    in_net = term_feats.index.intersection(b.network_nodes)
    annotated = _annotated_ids(b, mode, at)
    scope = _scope(b, at) if mode == "type" else None
    f = b.features
    votes = b.votes[(b.votes["ann_type"] == at) & (b.votes["term_id"] == tid)].set_index("feature_id") \
        if len(b.votes) else pd.DataFrame()
    genr = b.enrich[(b.enrich["ann_type"] == at) & (b.enrich["term_id"] == tid)].set_index("group")["padj"]
    if link == "direct":
        e = b.neighbors_of_set(in_net)
        e = e[~e["neighbor"].isin(annotated)]
        if scope:
            e = e[f.loc[e["neighbor"], "dtype"].to_numpy() == scope]
        if e.empty:
            return pd.DataFrame(), in_net
        c = e.groupby("neighbor").agg(term_neighbors=("feature_id", "nunique"), max_r=("weight", "max"))
        c["degree"] = f.loc[c.index, "degree"].to_numpy()
    else:
        groups = f.loc[in_net, "group"].dropna()
        groups = groups[~groups.map(b.groups["is_noise"]).fillna(False).astype(bool)]
        counts = groups.value_counts()
        members = f[f["group"].isin(counts.index) & ~f.index.isin(annotated)]
        if scope:
            members = members[members["dtype"] == scope]
        if members.empty:
            return pd.DataFrame(), in_net
        c = pd.DataFrame(index=members.index)
        c["term_in_group"] = members["group"].map(counts).to_numpy()
    c = c.rename_axis("feature_id")
    c["dtype"] = f.loc[c.index, "dtype"].to_numpy()
    c["display_name"] = f.loc[c.index, "display_name"].to_numpy()
    c["group"] = f.loc[c.index, "group"].to_numpy()
    c["kME"] = f.loc[c.index, "kME"].to_numpy()
    c["group_padj"] = c["group"].map(genr)
    if len(votes):
        c["vote_k"] = votes["k"].reindex(c.index)
        c["vote_n"] = votes["n"].reindex(c.index)
        c["vote_padj"] = votes["padj"].reindex(c.index)
    sort = ["vote_padj", "term_neighbors"] if link == "direct" and "vote_padj" in c else \
        (["term_neighbors"] if link == "direct" else ["group_padj", "kME"])
    asc = [True, False] if sort == ["vote_padj", "term_neighbors"] else ([False] if sort == ["term_neighbors"] else [True, False])
    return c.sort_values(sort, ascending=asc, na_position="last").reset_index(), in_net


def _subgraph(b, term_ids, cand, theme):
    cand = cand.head(MAX_GRAPH)
    cids = set(cand["feature_id"])
    e = b.adj[b.adj["feature_id"].isin(term_ids) & b.adj["neighbor"].isin(cids)]
    tids = set(e["feature_id"])
    c = th.chrome(theme)
    nodes = [{"data": {"id": t, "label": "", "kind": "term"}} for t in tids] + \
            [{"data": {"id": x, "label": "", "kind": "cand"}} for x in cids]
    edges = [{"data": {"source": s, "target": t}} for s, t in e[["feature_id", "neighbor"]].itertuples(index=False)]
    ss = [
        {"selector": "node", "style": {"width": 12, "height": 12, "border-width": 1, "border-color": c["surface"]}},
        {"selector": '[kind = "term"]', "style": {"background-color": th.cat(theme, 0)}},
        {"selector": '[kind = "cand"]', "style": {"background-color": c["surface"], "border-width": 2.5,
                                                  "border-color": th.cat(theme, 1), "width": 14, "height": 14}},
        {"selector": "edge", "style": {"width": 1, "line-color": c["axis"], "curve-style": "haystack"}},
        {"selector": "node:selected", "style": {"border-color": c["ink"], "border-width": 3}},
    ]
    legend = html.Div([
        html.Span([html.Span(className="legend-dot", style={"background": th.cat(theme, 0)}), "carries the term"],
                  className="me-3"),
        html.Span([html.Span(className="legend-dot", style={"background": "transparent",
                                                            "border": f"2px solid {th.cat(theme, 1)}"}),
                   "unannotated candidate"], className="me-3"),
        html.Span(f"top {min(len(cand), MAX_GRAPH)} candidates · click a node to open the feature",
                  className="text-secondary"),
    ], className="small mb-1")
    return html.Div([legend, cyto.Cytoscape(id="dc-cyto", elements=nodes + edges, stylesheet=ss,
                                            layout={"name": "cose", "animate": False, "numIter": 500},
                                            style={"width": "100%", "height": "460px"}, wheelSensitivity=0.2)])


def _votes_table(b, grid_id, votes, fdr, caption):
    if votes.empty:
        return ui.note(caption + " None of them carries an annotation.")
    v = votes.assign(entity_id=votes["ann_type"] + "::" + votes["term_id"],
                     annotation=votes["ann_type"].map(b.ann_label),
                     support=votes["k"].astype(str) + " of " + votes["n"].astype(str))
    return html.Div([html.Div(caption, className="small text-secondary mb-1"),
                     ui.grid(grid_id, v, [("annotation", "type", None, 160), ("entity_id", None, None, 130),
                                          ("term_name", "name"), ("support", "neighbours with term", None, 150),
                                          ("share", "share", "num", 80), ("p", "p", "p", 90)],
                             height=300, export_name=grid_id)])


def register(app, b):
    @app.callback(Output("dc-term", "options"), Input("dc-term", "search_value"), State("dc-term", "value"))
    def term_options(query, current):
        opts = [{"label": o["label"], "value": o["value"].split(":", 1)[1]}
                for o in b.search(query or "", kinds=["term"], limit=80)]
        if current and not any(o["value"] == current for o in opts):
            opts = [{"label": b.term_label(current, 120), "value": current}] + opts
        return opts

    @app.callback(Output("dc-feature", "options"), Input("dc-feature", "search_value"), State("dc-feature", "value"))
    def feature_options(query, current):
        opts = [{"label": o["label"], "value": o["value"].split(":", 1)[1]}
                for o in b.search(query or "", kinds=["feature"], limit=80)]
        if current and not any(o["value"] == current for o in opts):
            opts = [{"label": b.feature_label(current), "value": current}] + opts
        return opts

    @app.callback(Output("dc-term-body", "children"), Input("dc-term", "value"), Input("dc-unann", "value"),
                  Input("dc-link", "value"), Input("filters", "data"), Input("theme", "data"))
    def term_body(term, mode, link, filters, theme):
        fdr = (filters or {}).get("fdr", b.fdr)
        if not term or term not in b.terms.index:
            return ui.note("Search for an annotation term above.")
        at = term.split("::", 1)[0]
        cand, in_net = _term_candidates(b, term, mode, link)
        n_all = len(b.term_members(term))
        stats = ui.stat_row([
            ui.stat("Features with the term", f"{n_all:,}", f"{len(in_net):,} in the network"),
            ui.stat("Candidates", f"{len(cand):,}",
                    "directly connected" if link == "direct" else "in the same groups"),
            ui.stat("With neighbour-vote FDR < " + f"{fdr:g}",
                    f"{int((cand.get('vote_padj', pd.Series(dtype=float)) < fdr).sum()):,}" if len(cand) else "0")
            if link == "direct" else
            ui.stat(f"In groups enriched for the term (FDR < {fdr:g})",
                    f"{int((cand['group_padj'] < fdr).sum()):,}" if len(cand) else "0"),
        ])
        notes = []
        if link == "direct" and b.bipartite:
            notes.append(ui.note("This network only links features of different data types (bipartite: e.g. genes to "
                                 "metabolites), so direct candidates are always of the other data type. Switch "
                                 "'Linked by' to 'same data-driven group' to find same-data-type candidates."))
        if cand.empty:
            return html.Div([stats, *notes, ui.note("No candidates for this term with these settings.")])
        if link == "direct":
            cols = [("feature_id", None, None, 220), ("dtype", None, None, 60), ("display_name", "name"),
                    ("group", None, None, 120), ("term_neighbors", "neighbours with term", "int", 160),
                    ("max_r", "strongest r", "num", 110)]
            if "vote_padj" in cand:
                cols += [("vote_padj", "neighbour-vote padj", "p", 160)]
            cols += [("group_padj", "group enrichment padj", "p", 170)]
            explain = ("Neighbours with term: how many features carrying the term are directly connected to the "
                       "candidate. Neighbour-vote padj: is the term over-represented among all the candidate's "
                       "annotated neighbours (hypergeometric, FDR)? Group enrichment padj: is the term enriched in "
                       "the candidate's group?")
            graph = _subgraph(b, set(in_net), cand, theme)
            body = dbc.Row([dbc.Col(ui.grid("dc-term-grid", cand, cols, height=460,
                                            export_name=f"candidates_{term.replace('::', '_')}"), lg=7),
                            dbc.Col(graph, lg=5)])
        else:
            cols = [("feature_id", None, None, 220), ("dtype", None, None, 60), ("display_name", "name"),
                    ("group", None, None, 120), ("term_in_group", "group members with term", "int", 180),
                    ("kME", "kME", "num", 80), ("group_padj", "group enrichment padj", "p", 170)]
            explain = ("Unannotated members of every group that contains features with the term. Group enrichment "
                       "padj: is the term enriched in that group (hypergeometric, FDR)? kME: how typical the "
                       "candidate is of its group.")
            body = ui.grid("dc-term-grid", cand, cols, height=460, export_name=f"candidates_{term.replace('::', '_')}")
        return html.Div([stats, *notes, html.Div(explain, className="small text-secondary mb-2"), body])

    @app.callback(Output("dc-feature-body", "children"), Input("dc-feature", "value"), Input("filters", "data"))
    def feature_body(fid, filters):
        fdr = (filters or {}).get("fdr", b.fdr)
        if not fid or fid not in b.features.index:
            return ui.note("Search for a feature above.")
        r = b.features.loc[fid]
        group = r["group"] if isinstance(r["group"], str) else None
        own = b.feature_annotations(fid)
        head = html.Div([
            html.H6([html.A(fid, href=ui.href("/feature", id=fid)),
                     html.Span(f"  {r['display_name']}" if r["display_name"] else "", className="text-secondary")]),
            html.Div([f"{th.DTYPE_LABEL.get(r['dtype'], r['dtype'])} · ",
                      html.A(group, href=ui.href("/groups", group=group)) if group else "not grouped",
                      f" · {len(own):,} annotations of its own" if len(own) else " · no annotation of any type"],
                     className="small text-secondary mb-2"),
        ])
        panels = []
        nb = b.neighbors(fid)
        if len(nb):
            v = net.votes_from_neighbors(nb["neighbor"], b.ann, b.vote_bg)
            panels.append(ui.card("Annotations of its direct network neighbours", _votes_table(
                b, "dc-nb-votes", v, fdr,
                f"{len(nb):,} directly connected features. For each annotation term: how many of the neighbours "
                "annotated with that type carry it, and the neighbour-voting p-value (hypergeometric).")))
        pool = b.group_members(group).index if group else b.network_nodes
        pool = [p for p in pool if p != fid and b.features.at[p, "is_annotated"]]
        corr = b.correlated_features(fid, pool, top=10)
        if len(corr):
            v = net.votes_from_neighbors(corr.index, b.ann, b.vote_bg)
            where = f"in its group ({group})" if group else "in the network"
            panels.append(ui.card(f"Annotations of the 10 most correlated annotated features {where}", [
                _votes_table(b, "dc-corr-votes", v, fdr,
                             "Same test on the 10 annotated features whose profiles correlate best with this one. "
                             "Useful when direct neighbours are of another data type."),
                html.Div("Those features:", className="small text-secondary mt-2 mb-1"),
                ui.grid("dc-corr-feats", b.features.loc[corr.index, ["dtype", "display_name", "group"]]
                        .assign(r=corr.to_numpy()).rename_axis("feature_id").reset_index(),
                        [("feature_id", None, None, 220), ("dtype", None, None, 60), ("display_name", "name"),
                         ("r", "r", "num", 70)], height=240, export_name=f"{fid}_correlated"),
            ]))
        if group:
            e = b.enrichment_for_group(group)
            e = e[e["padj"] < fdr].sort_values("padj").head(30).assign(
                entity_id=lambda d: d["ann_type"] + "::" + d["term_id"], annotation=lambda d: d["ann_type"].map(b.ann_label))
            panels.append(ui.card(f"Terms enriched in its group ({group})", ui.grid(
                "dc-group-enr", e, [("annotation", "type", None, 160), ("entity_id", None, None, 130),
                                    ("term_name", "name"), ("k", "k", "int", 60), ("fold", "fold", "num", 70),
                                    ("padj", "padj", "p", 80)], height=260, export_name=f"{group}_enrichment")
                if len(e) else ui.note(f"No term is enriched in {group} at FDR < {fdr}.")))
        return html.Div([head, *panels])

    @app.callback(Output("dc-all-body", "children"), Input("dc-all-type", "value"), Input("dc-all-dtype", "value"),
                  Input("dc-all-strict", "value"), Input("filters", "data"))
    def all_body(at, dtype, strict, filters):
        fdr = (filters or {}).get("fdr", b.fdr)
        v = b.votes[b.votes["padj"] < fdr] if len(b.votes) else b.votes
        if v.empty:
            return ui.note(f"No neighbour-voting result reaches FDR < {fdr}.")
        if at and at != "all":
            v = v[v["ann_type"] == at]
        f = b.features
        v = v.assign(dtype=f.loc[v["feature_id"], "dtype"].to_numpy(),
                     display_name=f.loc[v["feature_id"], "display_name"].to_numpy(),
                     group=f.loc[v["feature_id"], "group"].to_numpy(),
                     any_annotation=f.loc[v["feature_id"], "is_annotated"].to_numpy())
        if dtype and dtype != "all":
            v = v[v["dtype"] == dtype]
        if strict:
            v = v[~v["any_annotation"]]
        e = b.enrich.set_index(["ann_type", "term_id", "group"])["padj"]
        v = v.assign(group_padj=pd.MultiIndex.from_frame(v[["ann_type", "term_id", "group"]]).map(e.to_dict()),
                     entity_id=v["ann_type"] + "::" + v["term_id"], annotation=v["ann_type"].map(b.ann_label),
                     term_name=[b.terms["term_name"].get(f"{a}::{t}", "") for a, t in zip(v["ann_type"], v["term_id"])],
                     support=v["k"].astype(str) + " of " + v["n"].astype(str)).sort_values("padj")
        return ui.card(f"{v['feature_id'].nunique():,} features, {len(v):,} feature–term hypotheses", ui.grid(
            "dc-all-grid", v, [("feature_id", None, None, 210), ("dtype", None, None, 60), ("display_name", "name"),
                               ("group", None, None, 110), ("annotation", "type", None, 150),
                               ("entity_id", "term ID", None, 120), ("term_name", "term name"),
                               ("support", "neighbours with term", None, 150), ("padj", "vote padj", "p", 100),
                               ("group_padj", "group enrichment padj", "p", 160)],
            height=620, export_name="discovery_candidates"))

    @app.callback(Output("url", "href", allow_duplicate=True), Input("dc-cyto", "tapNodeData"),
                  prevent_initial_call=True)
    def tap(data):
        return ui.href("/feature", id=data["id"]) if data else no_update
