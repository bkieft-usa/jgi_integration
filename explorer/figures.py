"""Plotly figure builders. Every function takes ``theme`` ('light' | 'dark')."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from explorer import theme as th

HOVER_BOX = dict(namelength=-1)


def _base(fig: go.Figure, theme, height: int, **layout) -> go.Figure:
    fig.update_layout(template=th.template(theme), height=height, **layout)
    return fig


def sample_scatter(samples: pd.DataFrame, x: str, y: str, color_var: Optional[str], theme, levels: List,
                   xlab: str, ylab: str, height: int = 420) -> go.Figure:
    fig = go.Figure()
    hover_cols = [c for c in samples.columns if not c.startswith(("PC", "UMAP"))][:8]
    def hover(df):
        return ["<br>".join([f"<b>{i}</b>"] + [f"{c}: {r[c]}" for c in hover_cols]) for i, r in df.iterrows()]
    if color_var and color_var in samples.columns and len(levels) <= 40:
        styles = th.category_style(theme, levels)
        for lv in levels:
            sub = samples[samples[color_var] == lv]
            col, sym = styles[lv]
            fig.add_trace(go.Scatter(x=sub[x], y=sub[y], mode="markers", name=str(lv),
                                     marker=dict(size=10, color=col, symbol=sym,
                                                 line=dict(width=1.5, color=th.chrome(theme)["surface"])),
                                     text=hover(sub), hovertemplate="%{text}<extra></extra>"))
    elif color_var and color_var in samples.columns:  # continuous
        fig.add_trace(go.Scatter(x=samples[x], y=samples[y], mode="markers", text=hover(samples),
                                 marker=dict(size=10, color=pd.to_numeric(samples[color_var], errors="coerce"),
                                             colorscale=th.sequential_scale(theme), showscale=True,
                                             colorbar=dict(title=color_var, thickness=10)),
                                 hovertemplate="%{text}<extra></extra>"))
    else:
        fig.add_trace(go.Scatter(x=samples[x], y=samples[y], mode="markers", text=hover(samples),
                                 marker=dict(size=10, color=th.cat(theme, 0)), hovertemplate="%{text}<extra></extra>"))
    return _base(fig, theme, height, xaxis_title=xlab, yaxis_title=ylab,
                 legend=dict(title=color_var, itemsizing="constant"))


def similarity_heatmap(sim: pd.DataFrame, theme, height: int = 480) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=sim.to_numpy(), x=sim.columns, y=sim.index, zmin=-1, zmax=1,
                               colorscale=th.diverging_scale(theme), colorbar=dict(title="r", thickness=10),
                               hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>"))
    return _base(fig, theme, height, xaxis=dict(showticklabels=False, ticks=""),
                 yaxis=dict(showticklabels=len(sim) <= 60, tickfont=dict(size=8), autorange="reversed"),
                 margin=dict(l=8 if len(sim) > 60 else 160, r=8, t=8, b=8))


def agreement_dots(types: pd.DataFrame, labels: Dict[str, str], theme, height: Optional[int] = None) -> go.Figure:
    c = th.chrome(theme)
    df = types.sort_values("z")
    names = [labels.get(a, a) for a in df["ann_type"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["null_mean"], y=names, mode="markers", name="random groups (null mean ± 2 SD)",
        marker=dict(size=8, color=c["neutral_mark"], symbol="line-ns-open", line=dict(width=2, color=c["muted"])),
        error_x=dict(type="data", array=2 * df["null_sd"], color=c["muted"], thickness=1.5, width=0),
        hovertemplate="%{y}<br>null mean %{x:.3f}<extra></extra>"))
    sig = df["p"] < 0.05
    fig.add_trace(go.Scatter(
        x=df["comembership"], y=names, mode="markers", name="observed",
        marker=dict(size=10, color=np.where(sig, th.cat(theme, 0), c["muted"]),
                    line=dict(width=1.5, color=c["surface"])),
        customdata=np.stack([df["z"], df["p"], df["ami"], df["n_features"], df["n_terms"]], axis=1),
        hovertemplate="%{y}<br>observed %{x:.3f}<br>z = %{customdata[0]:.1f}, perm p = %{customdata[1]:.3f}"
                      "<br>AMI = %{customdata[2]:.3f}<br>%{customdata[3]} features, %{customdata[4]} terms"
                      "<extra></extra>"))
    return _base(fig, theme, height or max(260, 28 * len(df) + 80),
                 xaxis_title="Same-group fraction of co-annotated feature pairs (mean over terms)",
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), margin=dict(l=200, r=16, t=40, b=48))


def size_histogram(groups: pd.DataFrame, theme, height: int = 260) -> go.Figure:
    real = groups[~groups["is_noise"]]["size"]
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 1e9]
    labels = ["1", "2", "3–4", "5–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500–999", "1k–2k", "2k–5k",
              "5k+"]
    counts = pd.cut(real, bins=bins, right=False, labels=labels).value_counts().reindex(labels).fillna(0)
    last = max((i for i, v in enumerate(counts) if v > 0), default=0)
    counts = counts.iloc[: last + 1]
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker=dict(color=th.cat(theme, 0), cornerradius=4),
                           hovertemplate="%{x} features: %{y} groups<extra></extra>"))
    return _base(fig, theme, height, xaxis_title="Group size (features)", yaxis_title="Groups", bargap=0.15)


def volcano(df: pd.DataFrame, theme, fdr: float, effect_label: str, name_col: str = "name",
            height: int = 460) -> go.Figure:
    c = th.chrome(theme)
    df = df.assign(nlp=-np.log10(df["p"].clip(lower=1e-300)))
    sig = df["padj"] < fdr
    fig = go.Figure()
    for mask, nm, col, size in ((~sig, f"not significant (FDR ≥ {fdr})", c["neutral_mark"], 5),
                                (sig, f"FDR < {fdr}", th.cat(theme, 0), 6)):
        sub = df[mask]
        fig.add_trace(go.Scattergl(
            x=sub["effect"], y=sub["nlp"], mode="markers", name=f"{nm} ({mask.sum():,})",
            marker=dict(size=size, color=col, opacity=0.85 if col != c["neutral_mark"] else 0.6),
            customdata=np.stack([sub["entity_id"], sub[name_col], sub["padj"]], axis=1) if len(sub) else None,
            hovertemplate="<b>%{customdata[1]}</b><br>effect %{x:.2f}<br>p %{y:.1f} (−log10)"
                          "<br>padj %{customdata[2]:.2e}<extra></extra>"))
    return _base(fig, theme, height, xaxis_title=effect_label, yaxis_title="−log10 p",
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), clickmode="event")


def ranked_bars(df: pd.DataFrame, theme, fdr: float, value_col: str, label_col: str, xlab: str,
                height: Optional[int] = None) -> go.Figure:
    c = th.chrome(theme)
    df = df.iloc[::-1]
    sig = df["padj"] < fdr
    fig = go.Figure(go.Bar(
        x=df[value_col], y=df[label_col], orientation="h",
        marker=dict(color=np.where(sig, th.cat(theme, 0), c["neutral_mark"]), cornerradius=4),
        customdata=np.stack([df["entity_id"], df["padj"]], axis=1),
        hovertemplate="%{y}<br>" + xlab + " %{x:.2f}<br>padj %{customdata[1]:.2e}<extra></extra>"))
    return _base(fig, theme, height or max(260, 22 * len(df) + 70), xaxis_title=xlab,
                 margin=dict(l=260, r=16, t=16, b=48), bargap=0.25)


def feature_map(features: pd.DataFrame, theme, selected_group: Optional[str] = None,
                highlight: Optional[Iterable[str]] = None, color_by: str = "group",
                height: int = 560) -> go.Figure:
    c = th.chrome(theme)
    f = features.dropna(subset=["x"])
    fig = go.Figure()
    hl = set(highlight or [])
    def custom(df):
        return np.stack([df.index, df["group"].fillna(""), df["display_name"].fillna("")], axis=1)
    tmpl = "<b>%{customdata[0]}</b><br>%{customdata[2]}<br>group: %{customdata[1]}<extra></extra>"
    if color_by == "dtype":
        for dt in sorted(f["dtype"].unique()):
            sub = f[f["dtype"] == dt]
            fig.add_trace(go.Scattergl(x=sub["x"], y=sub["y"], mode="markers", name=th.DTYPE_LABEL.get(dt, dt),
                                       marker=dict(size=3, color=th.dtype_color(theme, dt), opacity=0.6),
                                       customdata=custom(sub), hovertemplate=tmpl))
    else:
        rest = f[f["group"] != selected_group]
        noise = rest["is_noise"].fillna(False).astype(bool)
        fig.add_trace(go.Scattergl(x=rest.loc[~noise, "x"], y=rest.loc[~noise, "y"], mode="markers",
                                   name="other groups", marker=dict(size=3, color=c["neutral_mark"], opacity=0.5),
                                   customdata=custom(rest[~noise]), hovertemplate=tmpl))
        if noise.any():
            fig.add_trace(go.Scattergl(x=rest.loc[noise, "x"], y=rest.loc[noise, "y"], mode="markers",
                                       name="noise / unassigned", visible="legendonly",
                                       marker=dict(size=3, color=c["muted"], opacity=0.35),
                                       customdata=custom(rest[noise]), hovertemplate=tmpl))
        if selected_group is not None:
            sel = f[f["group"] == selected_group]
            fig.add_trace(go.Scattergl(x=sel["x"], y=sel["y"], mode="markers", name=f"{selected_group} ({len(sel)})",
                                       marker=dict(size=6, color=th.cat(theme, 0),
                                                   line=dict(width=0.5, color=c["surface"])),
                                       customdata=custom(sel), hovertemplate=tmpl))
    if hl:
        sub = f[f.index.isin(hl)]
        fig.add_trace(go.Scattergl(x=sub["x"], y=sub["y"], mode="markers", name=f"search matches ({len(sub)})",
                                   marker=dict(size=10, color=th.cat(theme, 1), symbol="circle-open",
                                               line=dict(width=2, color=th.cat(theme, 1))),
                                   customdata=custom(sub), hovertemplate=tmpl))
    return _base(fig, theme, height, xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                 legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, itemsizing="constant"),
                 margin=dict(l=8, r=8, t=32, b=8), dragmode="pan", clickmode="event")


def dendrogram(native: Dict, theme, selected_group: Optional[str] = None, height: int = 520) -> go.Figure:
    c = th.chrome(theme)
    ic, dc = np.array(native["icoord"]), np.array(native["dcoord"])
    xs, ys = [], []
    for xi, yi in zip(ic, dc):
        xs += list(xi) + [None]
        ys += list(yi) + [None]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=c["muted"], width=1), hoverinfo="skip",
                               showlegend=False))
    leaves = native["leaf_groups"]
    lx = 5 + 10 * np.arange(len(leaves))
    is_sel = np.array([g == selected_group for g in leaves])
    fig.add_trace(go.Scatter(x=lx, y=np.zeros(len(leaves)), mode="markers", showlegend=False,
                             marker=dict(size=np.where(is_sel, 12, 5),
                                         color=np.where(is_sel, th.cat(theme, 0), c["neutral_mark"])),
                             customdata=np.array(leaves), hovertemplate="%{customdata}<extra></extra>"))
    if native.get("cut_height"):
        fig.add_hline(y=native["cut_height"], line=dict(color=th.cat(theme, 1), width=1),
                      annotation_text="cut height", annotation_font_color=c["ink2"])
    return _base(fig, theme, height, xaxis=dict(visible=False), yaxis_title="Merge distance",
                 margin=dict(l=56, r=8, t=16, b=8), clickmode="event")


def hdbscan_icicle(native: Dict, theme, selected_group: Optional[str] = None, height: int = 520) -> go.Figure:
    """HDBSCAN condensed cluster tree: each cluster is a vertical segment from its birth to the λ where it
    splits; segment width grows with cluster size. Clusters HDBSCAN selected as groups are blue; the
    open group is orange. Higher λ = denser."""
    c = th.chrome(theme)
    df = pd.DataFrame(native["clusters"])
    df["node"] = df["node"].astype(int)
    df["parent"] = pd.to_numeric(df["parent"], errors="coerce")
    children = df.dropna(subset=["parent"]).groupby("parent")["node"].apply(list).to_dict()
    root = int(df.loc[df["parent"].isna(), "node"].iloc[0])
    birth = df.set_index("node")["lambda_birth"].to_dict()
    # a cluster ends where its children are born (or at its deepest descendant's birth for leaves)
    end = {n: min(birth[ch] for ch in children[n]) if n in children else None for n in df["node"]}
    leaf_end = df["lambda_birth"].max() * 1.05 if len(df) else 1.0
    xpos, counter = {}, [0]
    def place(n):
        kids = children.get(n, [])
        if not kids:
            xpos[n] = counter[0]; counter[0] += 1
        else:
            for k in sorted(kids, key=lambda k: -df.loc[df["node"] == k, "size"].iloc[0]):
                place(k)
            xpos[n] = float(np.mean([xpos[k] for k in kids]))
    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    place(root)
    rec = df.set_index("node")
    fig = go.Figure()
    lines_x, lines_y = [], []
    for n, kids in children.items():
        for k in kids:  # horizontal connector at the split λ
            lines_x += [xpos[n], xpos[k], None]
            lines_y += [birth[k], birth[k], None]
    fig.add_trace(go.Scatter(x=lines_x, y=lines_y, mode="lines", line=dict(color=c["axis"], width=1),
                             hoverinfo="skip", showlegend=False))
    max_size = max(rec["size"].max(), 1)
    for kind, mask, color in (("other clusters", ~rec["selected"].astype(bool), c["neutral_mark"]),
                              ("selected as groups", rec["selected"].astype(bool), th.cat(theme, 0))):
        sub = rec[mask]
        for n, r in sub.iterrows():
            y1 = end[n] if end[n] is not None else leaf_end
            is_open = r["group"] == selected_group and selected_group
            fig.add_trace(go.Scatter(
                x=[xpos[n], xpos[n]], y=[r["lambda_birth"], y1], mode="lines", showlegend=False,
                line=dict(color=th.cat(theme, 1) if is_open else color, width=1.5 + 10 * np.sqrt(r["size"] / max_size)),
                customdata=[[r["group"] or f"cluster {n}", r["size"], r["stability"], r["group"]]] * 2,
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} features<br>stability %{customdata[2]:.1f}"
                              "<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=th.cat(theme, 0), width=4),
                             name="selected as groups"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=c["neutral_mark"], width=4),
                             name="parent / unselected clusters"))
    return _base(fig, theme, height, xaxis=dict(visible=False), yaxis=dict(title="λ (density; higher = denser)",
                                                                            autorange="reversed"),
                 legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), margin=dict(l=56, r=8, t=36, b=8),
                 clickmode="event")


def profile_strip(values: pd.Series, samples: pd.DataFrame, variable: str, levels: List, theme, ylab: str,
                  height: int = 340, title: Optional[str] = None) -> go.Figure:
    """Per-sample values grouped by a metadata variable: points + level means."""
    c = th.chrome(theme)
    df = pd.DataFrame({"v": values}).join(samples[[variable]], how="inner").dropna()
    fig = go.Figure()
    if df.empty:
        return th.empty_figure(theme, "No samples with this metadata variable", height)
    continuous = pd.api.types.is_numeric_dtype(df[variable]) and df[variable].nunique() > 8
    if continuous:
        fig.add_trace(go.Scatter(x=df[variable], y=df["v"], mode="markers", text=df.index,
                                 marker=dict(size=8, color=th.cat(theme, 0)),
                                 hovertemplate="%{text}<br>%{x}: %{y:.2f}<extra></extra>", showlegend=False))
        xlab = variable
    else:
        order = [lv for lv in levels if lv in set(df[variable])]
        pos = {lv: i for i, lv in enumerate(order)}
        rng = np.random.default_rng(0)
        x = df[variable].map(pos) + rng.uniform(-0.18, 0.18, len(df))
        fig.add_trace(go.Scatter(x=x, y=df["v"], mode="markers", text=df.index, showlegend=False,
                                 marker=dict(size=8, color=th.cat(theme, 0), opacity=0.85,
                                             line=dict(width=1, color=c["surface"])),
                                 hovertemplate="%{text}<br>%{y:.2f}<extra></extra>"))
        means = df.groupby(variable)["v"].mean().reindex(order)
        fig.add_trace(go.Scatter(x=list(range(len(order))), y=means.values, mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ew-open", size=26, color=c["ink"],
                                             line=dict(width=2.5, color=c["ink"])),
                                 hovertemplate="%{text}<br>mean %{y:.2f}<extra></extra>",
                                 text=[str(o) for o in order]))
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=[str(o) for o in order],
                         tickangle=-35 if len(order) > 4 else 0)
        xlab = variable
    return _base(fig, theme, height, xaxis_title=xlab, yaxis_title=ylab, title=title,
                 margin=dict(l=56, r=16, t=36 if title else 16, b=90 if len(levels) > 4 else 48))


def composition_bars(df: pd.DataFrame, theme, fdr: float, height: Optional[int] = None) -> go.Figure:
    """Top terms in a group: bar = features in the group, colored by enrichment significance."""
    c = th.chrome(theme)
    df = df.iloc[::-1]
    sig = df["padj"] < fdr
    fig = go.Figure(go.Bar(
        x=df["k"], y=df["label"], orientation="h",
        marker=dict(color=np.where(sig, th.cat(theme, 0), c["neutral_mark"]), cornerradius=4),
        customdata=np.stack([df["K"], df["fold"], df["padj"], df["entity_id"]], axis=1),
        hovertemplate="%{y}<br>%{x} features in group (of %{customdata[0]} in background)"
                      "<br>fold enrichment %{customdata[1]:.1f}<br>padj %{customdata[2]:.2e}<extra></extra>"))
    return _base(fig, theme, height or max(220, 22 * len(df) + 70), xaxis_title="Features in group with term",
                 margin=dict(l=280, r=16, t=16, b=48), bargap=0.25)


def term_group_bars(df: pd.DataFrame, theme, fdr: float, height: Optional[int] = None) -> go.Figure:
    c = th.chrome(theme)
    df = df.iloc[::-1]
    sig = df["padj"] < fdr
    fig = go.Figure(go.Bar(
        x=df["k"], y=df["group"], orientation="h",
        marker=dict(color=np.where(sig, th.cat(theme, 0), c["neutral_mark"]), cornerradius=4),
        customdata=np.stack([df["n"], df["fold"], df["padj"]], axis=1),
        hovertemplate="%{y}<br>%{x} of the term's features<br>group has %{customdata[0]} annotated features"
                      "<br>fold %{customdata[1]:.1f}, padj %{customdata[2]:.2e}<extra></extra>"))
    return _base(fig, theme, height or max(220, 22 * len(df) + 70), xaxis_title="Term features in group",
                 margin=dict(l=140, r=16, t=16, b=48), bargap=0.25)


def enrichment_heatmap(mat: pd.DataFrame, theme, metric_label: str, height: Optional[int] = None,
                       hover: Optional[np.ndarray] = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=mat.to_numpy(), x=mat.columns, y=mat.index, colorscale=th.sequential_scale(theme), xgap=1, ygap=1,
        colorbar=dict(title=metric_label, thickness=10), hoverongaps=False,
        customdata=hover, hovertemplate="%{y}<br>%{x}<br>" + metric_label + " %{z:.2f}"
                                        + ("<br>%{customdata}" if hover is not None else "") + "<extra></extra>"))
    return _base(fig, theme, height or max(360, 16 * len(mat) + 160),
                 xaxis=dict(tickangle=-60, tickfont=dict(size=9), showgrid=False),
                 yaxis=dict(tickfont=dict(size=10), autorange="reversed", showgrid=False),
                 margin=dict(l=320, r=16, t=16, b=120), clickmode="event")
