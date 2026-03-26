"""
vi_dash_app.py — Interactive Dash app for VI time-series exploration.

Click any field polygon on the map to display its NDVI / EVI / NDRE
time series in the sidebar. Use the Layers toggle in the sidebar to
show/hide color groups.

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
_gdf["color"] = _gdf["block_id"].map(block_colors).fillna("#555555")
_gdf["block_id"] = _gdf["block_id"].fillna(-1).astype(int)
_gdf["cluster"] = _gdf["cluster"].fillna(-1).astype(int)
_gdf = _gdf.rename(columns={NAME_COL: "field_id"})

# ---------------------------------------------------------------------------
# Build layer data: one entry per unique color group
# ---------------------------------------------------------------------------
# layer_index i  →  (color, fill_opacity, label, geojson_data)

_layers: list[tuple] = []
for i, (color, group) in enumerate(_gdf.groupby("color")):
    fill_opacity = 0.5 if color == "#555555" else 0.75
    label = "Unassigned" if color == "#555555" else f"Group {i}"
    _layers.append((color, fill_opacity, label, group.__geo_interface__))

_hover_style = {"weight": 2, "color": "white", "fillOpacity": 0.35}

_center = boundaries.geometry.unary_union.centroid
MAP_CENTER = [_center.y, _center.x]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

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
                                hoverStyle=_hover_style,
                            )
                            for i, (color, fill_opacity, label, geojson) in enumerate(_layers)
                        ],
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
                html.P(
                    "Click a field on the map to view its vegetation index time series.",
                    style={"color": "#a6adc8", "fontSize": "13px"},
                ),

                # Layer visibility toggle
                html.Details(
                    open=False,
                    children=[
                        html.Summary("Layers", style={"cursor": "pointer",
                                                       "color": "#89b4fa",
                                                       "fontSize": "13px"}),
                        dcc.Checklist(
                            id="layer-toggle",
                            options=[
                                {"label": html.Span(
                                    [
                                        html.Span("■ ", style={"color": color}),
                                        label,
                                    ]
                                ), "value": i}
                                for i, (color, _, label, _) in enumerate(_layers)
                            ],
                            value=list(range(len(_layers))),
                            style={"fontSize": "12px", "marginTop": "6px"},
                        ),
                    ],
                    style={"borderTop": "1px solid #313244", "paddingTop": "8px"},
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
    Input("layer-toggle", "value"),
)
def toggle_layers(visible_indices):
    visible = set(visible_indices or [])
    styles = []
    for i, (color, fill_opacity, _, _) in enumerate(_layers):
        if i in visible:
            styles.append({"fillColor": color, "color": "black",
                            "weight": 0.8, "fillOpacity": fill_opacity})
        else:
            styles.append({"fillColor": color, "color": "black",
                            "weight": 0, "fillOpacity": 0})
    return styles


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
    block_id = props.get("block_id", "—")
    cluster = props.get("cluster", "—")

    info = html.Div([
        html.Span("Field: ", style={"color": "#a6adc8"}),
        html.Strong(field_id, style={"color": "#cdd6f4"}),
        html.Br(),
        html.Span(f"Block {block_id}  ·  Cluster {cluster}",
                  style={"color": "#a6adc8", "fontSize": "12px"}),
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
