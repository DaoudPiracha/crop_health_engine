"""
vi_dash_app.py — Interactive Dash app for VI time-series exploration.

Click any field polygon on the map to display its NDVI / EVI / NDRE
time series in the sidebar.

Usage:
    python -m engine.vi_analysis.vi_dash_app
"""

import os

import dash
import dash_leaflet as dl
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html

from engine.vi_analysis.vi_analysis import load_vi_log, _block_colors, _rgb_to_hex

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASSET_DIR = "/Users/daoud/PycharmAssets/shahmeer_farms"
BOUNDARIES_FILE = f"{ASSET_DIR}/shahmeer_drawn_named.geojson"

ENGINE_DIR = os.path.dirname(__file__)
_engine_root = os.path.join(ENGINE_DIR, "..")
LOG_FILE = os.path.join(_engine_root, "kharif_shahmeer_field_veg_index_stats.csv")
BLOCKS_FILE = os.path.join(_engine_root, "shahmeer_blocks.csv")
WWF_FILE = os.path.join(_engine_root, "shahmeer_wwf_map.geojson")

VI_OPTIONS = ["ndvi"]
NAME_COL = "Name"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

boundaries: gpd.GeoDataFrame = gpd.read_file(BOUNDARIES_FILE).to_crs("epsg:4326")
blocks_df: pd.DataFrame = pd.read_csv(BLOCKS_FILE)
vi_log: pd.DataFrame = load_vi_log(LOG_FILE)

block_colors: dict = {
    bid: _rgb_to_hex(rgb) for bid, rgb in _block_colors(blocks_df).items()
}

_gdf = boundaries[[NAME_COL, "geometry"]].merge(
    blocks_df, left_on=NAME_COL, right_on="name", how="left"
)
_gdf["color"] = _gdf["block_id"].map(block_colors).fillna("#bbbbbb")
_gdf["block_id"] = _gdf["block_id"].fillna(-1).astype(int)
_gdf["cluster"] = _gdf["cluster"].fillna(-1).astype(int)
_gdf = _gdf.rename(columns={NAME_COL: "field_id"})

wwf_geojson = None
wwf_gdf = None
if os.path.exists(WWF_FILE):
    wwf_gdf = gpd.read_file(WWF_FILE).to_crs("epsg:4326")
    wwf_geojson = wwf_gdf.__geo_interface__
    # Spatial join: attach WWF name to each field
    _joined = gpd.sjoin(_gdf, wwf_gdf[["Name", "geometry"]], how="left", predicate="intersects")
    _joined = _joined[~_joined.index.duplicated(keep="first")]
    _gdf["wwf_name"] = _joined["Name"]
else:
    _gdf["wwf_name"] = None

# ---------------------------------------------------------------------------
# Layer data: one entry per unique color group
# ---------------------------------------------------------------------------

_layers: list[tuple] = []
for i, (color, group) in enumerate(_gdf.groupby("color")):
    fill_opacity = 0.5 if color == "#bbbbbb" else 0.75
    _layers.append((color, fill_opacity, group.__geo_interface__))


_center = boundaries.geometry.unary_union.centroid
MAP_CENTER = [_center.y, _center.x]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_TOGGLE_STYLE = {
    "fontSize": "12px", "background": "#313244", "color": "#cdd6f4",
    "border": "none", "borderRadius": "4px", "padding": "4px 10px",
    "cursor": "pointer", "textAlign": "left",
}

app = dash.Dash(__name__)

_EMPTY_STORE = {"field_a": None, "field_b": None, "next_slot": "a"}

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    children=[
        dcc.Store(id="compare-store", data=_EMPTY_STORE),
        # ── Left: map ──────────────────────────────────────────────────────
        html.Div(
            style={"flex": "1", "position": "relative"},
            children=[
                dl.Map(
                    center=MAP_CENTER,
                    zoom=14,
                    style={"height": "100%", "width": "100%"},
                    children=[
                        dl.TileLayer(
                            url="https://server.arcgisonline.com/ArcGIS/rest/services/"
                                "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                            attribution="Esri World Imagery",
                        ),
                        *[
                            dl.GeoJSON(
                                id={"type": "field-layer", "index": i},
                                data=geojson,
                                style={"fillColor": color, "color": "black",
                                       "weight": 0.8, "fillOpacity": fill_opacity},
                                hoverStyle={"weight": 2, "color": "white", "fillOpacity": 0},
                            )
                            for i, (color, fill_opacity, geojson) in enumerate(_layers)
                        ],
                        dl.GeoJSON(
                            id="wwf-layer",
                            data=wwf_geojson,
                            style={"fillColor": "none", "color": "white",
                                   "weight": 2.5, "fillOpacity": 0},
                        ) if wwf_geojson else None,
                        dl.GeoJSON(id="highlight-a", data=None,
                                   style={"fillColor": "none", "color": "red",
                                          "weight": 2.5, "fillOpacity": 0}),
                        dl.GeoJSON(id="highlight-b", data=None,
                                   style={"fillColor": "none", "color": "yellow",
                                          "weight": 2.5, "fillOpacity": 0}),
                    ],
                ),
            ],
        ),
        # ── Right: sidebar ─────────────────────────────────────────────────
        html.Div(
            style={
                "width": "380px",
                "padding": "16px",
                "boxSizing": "border-box",
                "overflowY": "auto",
                "background": "#1e1e2e",
                "color": "#cdd6f4",
                "display": "flex",
                "flexDirection": "column",
                "gap": "12px",
            },
            children=[
                html.H3("VI Analysis", style={"margin": "0", "color": "#cba6f7"}),

                # Layer toggles
                html.Div(
                    style={"display": "flex", "gap": "8px",
                           "borderBottom": "1px solid #313244", "paddingBottom": "10px"},
                    children=[
                        html.Button("Fields", id="btn-fields", n_clicks=0,
                                    style={**_TOGGLE_STYLE, "background": "#45475a"}),
                        html.Button("Outlines only", id="btn-outlines", n_clicks=0,
                                    style=_TOGGLE_STYLE),
                        html.Button("WWF boundaries", id="btn-wwf", n_clicks=0,
                                    style=_TOGGLE_STYLE),
                        html.Button("Compare", id="btn-compare", n_clicks=0,
                                    style=_TOGGLE_STYLE),
                    ],
                ),

                html.P(
                    id="sidebar-hint",
                    children="Click a field to view its NDVI time series.",
                    style={"color": "#a6adc8", "fontSize": "13px", "margin": "0"},
                ),
                html.Div(id="field-info", style={"fontSize": "13px"}),
                dcc.Graph(
                    id="vi-chart",
                    config={"displayModeBar": False},
                    style={"height": "340px"},
                ),
            ],
        ),
    ],
)


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        margin={"l": 40, "r": 10, "t": 10, "b": 40},
        xaxis={"showgrid": False},
        yaxis={"showgrid": True, "gridcolor": "#313244"},
    )
    return fig


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output({"type": "field-layer", "index": ALL}, "style"),
    Output({"type": "field-layer", "index": ALL}, "hoverStyle"),
    Output("btn-fields", "style"),
    Output("btn-outlines", "style"),
    Input("btn-fields", "n_clicks"),
    Input("btn-outlines", "n_clicks"),
)
def toggle_fields(fields_clicks, outlines_clicks):
    fields_on = (fields_clicks % 2) == 0
    outlines_on = (outlines_clicks % 2) == 1

    hover = {"weight": 2, "color": "red" if outlines_on else "white", "fillOpacity": 0}

    styles = []
    for color, fill_opacity, _ in _layers:
        if not fields_on:
            styles.append({"fillColor": color, "color": "black",
                            "weight": 0, "fillOpacity": 0})
        elif outlines_on:
            styles.append({"fillColor": color, "color": "white",
                            "weight": 1, "fillOpacity": 0})
        else:
            styles.append({"fillColor": color, "color": "black",
                            "weight": 0.8, "fillOpacity": fill_opacity})

    btn_fields_style = {**_TOGGLE_STYLE,
                        "background": "#45475a" if fields_on else "#313244"}
    btn_outlines_style = {**_TOGGLE_STYLE,
                          "background": "#45475a" if outlines_on else "#313244"}
    return styles, [hover] * len(_layers), btn_fields_style, btn_outlines_style


@app.callback(
    Output("wwf-layer", "style"),
    Output("btn-wwf", "style"),
    Input("btn-wwf", "n_clicks"),
)
def toggle_wwf(n_clicks):
    wwf_visible = (n_clicks % 2) == 1
    layer_style = {"fillColor": "none", "color": "white",
                   "weight": 2.5 if wwf_visible else 0, "fillOpacity": 0}
    btn_style = {**_TOGGLE_STYLE,
                 "background": "#45475a" if wwf_visible else "#313244"}
    return layer_style, btn_style


def _field_props(field_id: str) -> dict:
    """Return display properties for a field from the GeoDataFrame."""
    rows = _gdf[_gdf["field_id"] == field_id]
    if rows.empty:
        return {"block_id": "N/A", "cluster": "N/A", "wwf_name": "—"}
    row = rows.iloc[0]
    raw_block = row.get("block_id")
    raw_cluster = row.get("cluster")
    return {
        "block_id": "N/A" if pd.isna(raw_block) or int(raw_block) == -1 else int(raw_block),
        "cluster": "N/A" if pd.isna(raw_cluster) or int(raw_cluster) == -1 else int(raw_cluster),
        "wwf_name": row.get("wwf_name") or "—",
    }


def _field_card(field_id: str, label: str, color: str) -> html.Div:
    p = _field_props(field_id)
    return html.Div([
        html.Span(label, style={"color": color, "fontSize": "11px", "fontWeight": "bold"}),
        html.Br(),
        html.Strong(field_id, style={"color": "#cdd6f4"}),
        html.Br(),
        html.Span(f"Block ID {p['block_id']}  ·  Crop ID {p['cluster']}",
                  style={"color": "#a6adc8", "fontSize": "12px"}),
        html.Br(),
        html.Span("WWF ID: ", style={"color": "#a6adc8", "fontSize": "12px"}),
        html.Span(p["wwf_name"], style={"color": "#cdd6f4", "fontSize": "12px"}),
    ], style={"flex": "1", "minWidth": "0"})


def _ndvi_trace(field_id: str, label: str, color: str, dash: str) -> go.Scatter:
    data = vi_log[vi_log["name"] == field_id].sort_values("date")
    return go.Scatter(
        x=pd.to_datetime(data["date"]),
        y=data["ndvi_mean"],
        mode="lines+markers",
        name=f"{label}: {field_id}",
        line={"color": color, "width": 2, "dash": dash},
        marker={"size": 4},
    )


@app.callback(
    Output("compare-store", "data"),
    Input({"type": "field-layer", "index": ALL}, "clickData"),
    Input("btn-compare", "n_clicks"),
    State("compare-store", "data"),
)
def update_store(all_click_data, compare_clicks, store):
    compare_on = (compare_clicks % 2) == 1
    triggered_id = ctx.triggered_id

    if triggered_id == "btn-compare":
        # Toggling off: drop field_b and reset slot
        if not compare_on:
            return {**store, "field_b": None, "next_slot": "b"}
        return store

    if not ctx.triggered:
        return store
    click_data = ctx.triggered[0]["value"]
    if not click_data or not isinstance(click_data, dict):
        return store

    field_id = click_data.get("properties", {}).get("field_id", "")
    if not field_id:
        return store

    if not compare_on:
        return {"field_a": field_id, "field_b": None, "next_slot": "b"}

    if store.get("next_slot", "a") == "a":
        return {"field_a": field_id, "field_b": store.get("field_b"), "next_slot": "b"}
    else:
        return {"field_a": store.get("field_a"), "field_b": field_id, "next_slot": "a"}


def _field_geojson(field_id: str):
    """Return a GeoJSON FeatureCollection for a single field."""
    if not field_id:
        return None
    rows = _gdf[_gdf["field_id"] == field_id]
    return rows.__geo_interface__ if not rows.empty else None


@app.callback(
    Output("field-info", "children"),
    Output("vi-chart", "figure"),
    Output("btn-compare", "style"),
    Output("sidebar-hint", "children"),
    Output("highlight-a", "data"),
    Output("highlight-b", "data"),
    Input("compare-store", "data"),
    Input("btn-compare", "n_clicks"),
)
def render_selection(store, compare_clicks):
    compare_on = (compare_clicks % 2) == 1
    field_a = store.get("field_a")
    field_b = store.get("field_b")

    btn_style = {**_TOGGLE_STYLE, "background": "#45475a" if compare_on else "#313244"}

    if compare_on:
        next_slot = store.get("next_slot", "a")
        hint = f"Click to set Field {'A' if next_slot == 'a' else 'B'}."
    else:
        hint = "Click a field to view its NDVI time series."

    if not field_a:
        return "", _empty_figure(), btn_style, hint, None, None

    fig = _empty_figure()
    fig.add_trace(_ndvi_trace(field_a, "A", "#a6e3a1", "solid"))

    if field_b:
        fig.add_trace(_ndvi_trace(field_b, "B", "#89dceb", "solid"))

    fig.update_layout(
        legend={"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        xaxis_title="Date",
        yaxis_title="NDVI",
        yaxis={"range": [0, 1], "gridcolor": "#313244"},
    )

    if field_b:
        info = html.Div(
            style={"display": "flex", "gap": "10px"},
            children=[
                _field_card(field_a, "A", "red"),
                html.Div(style={"width": "1px", "background": "#313244"}),
                _field_card(field_b, "B", "yellow"),
            ],
        )
    else:
        info = _field_card(field_a, "A", "red" if compare_on else "#a6e3a1")

    highlight_a = _field_geojson(field_a) if compare_on else None
    highlight_b = _field_geojson(field_b) if compare_on else None
    return info, fig, btn_style, hint, highlight_a, highlight_b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
