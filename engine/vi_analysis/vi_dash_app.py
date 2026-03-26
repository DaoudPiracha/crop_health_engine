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
from dash import ALL, Input, Output, ctx, dcc, html

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

VI_OPTIONS = ["ndvi", "evi", "ndre"]
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

app.layout = html.Div(
    style={"display": "flex", "height": "100vh", "fontFamily": "sans-serif"},
    children=[
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
                    ],
                ),

                html.P(
                    "Click a field on the map to view its vegetation index time series.",
                    style={"color": "#a6adc8", "fontSize": "13px", "margin": "0"},
                ),
                html.Div(id="field-info", style={"fontSize": "13px"}),
                dcc.Checklist(
                    id="vi-selector",
                    options=[{"label": vi.upper(), "value": vi} for vi in VI_OPTIONS],
                    value=["ndvi"],
                    inline=True,
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"marginRight": "12px"},
                    style={"fontSize": "13px"},
                ),
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


@app.callback(
    Output("field-info", "children"),
    Output("vi-chart", "figure"),
    Input({"type": "field-layer", "index": ALL}, "clickData"),
    Input("vi-selector", "value"),
)
def on_field_click(all_click_data, selected_vis):
    if not ctx.triggered:
        return "", _empty_figure()
    click_data = ctx.triggered[0]["value"]
    if not click_data or not isinstance(click_data, dict):
        return "", _empty_figure()

    props = click_data.get("properties", {})
    field_id = props.get("field_id", "")
    raw_block = props.get("block_id")
    raw_cluster = props.get("cluster")
    block_id = "N/A" if raw_block is None or int(raw_block) == -1 else int(raw_block)
    cluster = "N/A" if raw_cluster is None or int(raw_cluster) == -1 else int(raw_cluster)
    wwf_name = props.get("wwf_name") or "—"

    info = html.Div([
        html.Span("Field: ", style={"color": "#a6adc8"}),
        html.Strong(field_id, style={"color": "#cdd6f4"}),
        html.Br(),
        html.Span(f"Block ID {block_id}  ·  Crop ID {cluster}",
                  style={"color": "#a6adc8", "fontSize": "12px"}),
        html.Br(),
        html.Span("WWF ID: ", style={"color": "#a6adc8", "fontSize": "12px"}),
        html.Span(wwf_name, style={"color": "#cdd6f4", "fontSize": "12px"}),
    ])

    field_data = vi_log[vi_log["name"] == field_id].sort_values("date")

    if field_data.empty:
        fig = _empty_figure()
        fig.add_annotation(
            text="No data for this field",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"color": "#a6adc8"},
        )
        return info, fig

    _VI_COLORS = {"ndvi": "#a6e3a1", "evi": "#89b4fa", "ndre": "#fab387"}
    fig = _empty_figure()
    dates = pd.to_datetime(field_data["date"])

    for vi in (selected_vis or []):
        col = f"{vi}_mean"
        if col not in field_data.columns:
            continue
        fig.add_trace(go.Scatter(
            x=dates,
            y=field_data[col],
            mode="lines+markers",
            name=vi.upper(),
            line={"color": _VI_COLORS.get(vi, "#cdd6f4"), "width": 2},
            marker={"size": 4},
        ))

    fig.update_layout(
        legend={"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        xaxis_title="Date",
        yaxis_title="Index value",
        yaxis={"range": [0, 1], "gridcolor": "#313244"},
    )

    return info, fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
