"""Colors and Plotly templates for the explorer (light and dark, each validated).

Color is assigned by the job it does:
* categorical (identity): fixed slot order, never cycled; data types own slots 1-3
* sequential (magnitude): one blue ramp, light -> dark
* diverging (signed effects, correlations): blue <-> red with a neutral gray midpoint
* emphasis: selected = slot 1, everything else = muted gray
"""
from __future__ import annotations

import plotly.graph_objects as go

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
SYMBOLS = ["circle", "diamond", "square", "triangle-up", "cross", "x", "star", "hexagon"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

CHROME = {
    "light": {"surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "mid": "#f0efec", "neutral_mark": "#c3c2b7",
              "blue": "#2a78d6", "red": "#e34948"},
    "dark": {"surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "mid": "#383835", "neutral_mark": "#4a4a46",
             "blue": "#3987e5", "red": "#e66767"},
}
DTYPE_SLOT = {"tx": 0, "mx": 1, "px": 2}
DTYPE_LABEL = {"tx": "Transcripts (tx)", "mx": "Metabolites (mx)", "px": "Proteins (px)"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def mode(theme: str | None) -> str:
    return "dark" if theme == "dark" else "light"


def chrome(theme) -> dict:
    return CHROME[mode(theme)]


def cat(theme, i: int) -> str:
    return CATEGORICAL[mode(theme)][i % 8]


def dtype_color(theme, dtype: str) -> str:
    return cat(theme, DTYPE_SLOT.get(dtype, 3))


def diverging_scale(theme) -> list:
    c = chrome(theme)
    return [[0.0, c["blue"]], [0.5, c["mid"]], [1.0, c["red"]]]


def sequential_scale(theme) -> list:
    ramp = SEQUENTIAL if mode(theme) == "light" else SEQUENTIAL_DARK
    return [[i / (len(ramp) - 1), col] for i, col in enumerate(ramp)]


def category_style(theme, levels) -> dict:
    """``{level: (color, symbol)}``: hue slots in fixed order, never generated.

    Scatter marks validate only three hues all-pairs, so with more than three levels a
    marker symbol is added as secondary encoding; the symbol offset after every eight
    levels guarantees no two levels share both hue and symbol.
    """
    levels = list(levels)
    if len(levels) <= 3:
        return {lv: (cat(theme, i), "circle") for i, lv in enumerate(levels)}
    return {lv: (cat(theme, i % 8), SYMBOLS[(i + i // 8) % len(SYMBOLS)]) for i, lv in enumerate(levels)}


def template(theme) -> go.layout.Template:
    c = chrome(theme)
    axis = dict(gridcolor=c["grid"], linecolor=c["axis"], zerolinecolor=c["axis"], tickcolor=c["axis"],
                tickfont=dict(color=c["muted"], size=11), title=dict(font=dict(color=c["ink2"], size=12)),
                showline=True, ticks="outside", ticklen=4)
    return go.layout.Template(layout=dict(
        font=dict(family=FONT, color=c["ink"], size=12),
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        colorway=CATEGORICAL[mode(theme)],
        xaxis=axis, yaxis=axis,
        legend=dict(font=dict(color=c["ink2"], size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["axis"], font=dict(family=FONT, color=c["ink"], size=12)),
        margin=dict(l=56, r=16, t=36, b=48),
        title=dict(font=dict(size=14, color=c["ink"]), x=0, xanchor="left"),
    ))


def empty_figure(theme, message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=template(theme), height=height, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      annotations=[dict(text=message, showarrow=False, font=dict(color=chrome(theme)["muted"], size=13),
                                        x=0.5, y=0.5, xref="paper", yref="paper")])
    return fig


def cytoscape_stylesheet(theme) -> list:
    c = chrome(theme)
    return [
        {"selector": "node", "style": {"width": "data(size)", "height": "data(size)", "background-color": "data(color)",
                                       "border-width": 1, "border-color": c["surface"], "label": "data(label)",
                                       "font-size": 9, "color": c["ink2"], "text-valign": "bottom",
                                       "text-margin-y": 2, "min-zoomed-font-size": 8}},
        {"selector": "edge", "style": {"width": "data(width)", "line-color": "data(color)", "opacity": 0.55,
                                       "curve-style": "haystack"}},
        {"selector": ".selected", "style": {"border-width": 3, "border-color": c["ink"], "z-index": 10,
                                            "font-weight": "bold", "color": c["ink"]}},
        {"selector": ".highlight", "style": {"border-width": 3, "border-color": cat(theme, 1), "z-index": 9}},
        {"selector": "node:selected", "style": {"border-width": 3, "border-color": c["ink"]}},
    ]
