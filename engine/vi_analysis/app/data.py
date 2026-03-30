"""
data.py — Data loading and preparation for the VI Dash app.

Call load_data() to get an AppData instance. The module-level `app_data`
singleton is created at import time for use by the Dash app.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from engine.vi_analysis.vi_analysis import block_colors, load_vi_log, rgb_to_hex
from engine.vi_analysis.app.theme import UNASSIGNED_COLOR, z_score_bin_color
from engine.vi_analysis.app.config import ASSET_DIR, CROP, SEASON

# ---------------------------------------------------------------------------
# Paths (derived from config)
# ---------------------------------------------------------------------------

BOUNDARIES_FILE = f"{ASSET_DIR}/{CROP}_drawn_named.geojson"

_engine_root = os.path.join(os.path.dirname(__file__), "..")
LOG_FILE         = os.path.join(_engine_root, f"../{SEASON}_{CROP}_field_veg_index_stats.csv")
BLOCKS_FILE      = os.path.join(_engine_root, f"../{CROP}_blocks.csv")
WWF_FILE         = os.path.join(_engine_root, f"../{CROP}_wwf_map.geojson")
Z_SCORE_FILE     = os.path.join(_engine_root, f"../{CROP}_ndvi_z_scores_norm.csv")
MARKERS_FILE     = os.path.join(_engine_root, f"../{CROP}_whatsapp_markers.geojson")
N_ZSCORE_BINS    = 10

NAME_COL = "Name"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Layer(NamedTuple):
    color: str
    fill_opacity: float
    geojson: dict


@dataclass
class AppData:
    vi_log:            pd.DataFrame
    gdf:               gpd.GeoDataFrame
    layers:            list[Layer]
    z_score_layers:    list[Layer]       # empty list if z-score file absent
    wwf_geojson:       dict | None
    map_center:        list[float]
    field_props_map:   dict[str, dict]   # field_id → {block_id, cluster, wwf_name}
    field_geojson_map: dict[str, dict]   # field_id → geojson FeatureCollection
    markers_geojson:   dict | None       # WhatsApp field markers


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data() -> AppData:
    for path, label in [
        (BOUNDARIES_FILE, "boundaries GeoJSON"),
        (BLOCKS_FILE,     "blocks CSV"),
        (LOG_FILE,        "VI log CSV"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")

    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("epsg:4326")
    blocks_df  = pd.read_csv(BLOCKS_FILE)
    vi_log     = load_vi_log(LOG_FILE)
    vi_log["name"] = vi_log["name"].astype(str)

    # Field GeoDataFrame
    color_map = {bid: rgb_to_hex(rgb) for bid, rgb in block_colors(blocks_df).items()}
    gdf = boundaries[[NAME_COL, "geometry"]].merge(
        blocks_df, left_on=NAME_COL, right_on="name", how="left"
    )
    gdf["color"]    = gdf["block_id"].map(color_map).fillna(UNASSIGNED_COLOR)
    gdf["block_id"] = gdf["block_id"].fillna(-1).astype(int)
    gdf["cluster"]  = gdf["cluster"].fillna(-1).astype(int)
    gdf = gdf.rename(columns={NAME_COL: "field_id"})
    gdf["field_id"] = gdf["field_id"].astype(str)

    # WWF spatial join
    wwf_geojson: dict | None = None
    if os.path.exists(WWF_FILE):
        wwf = gpd.read_file(WWF_FILE).to_crs("epsg:4326")
        wwf_geojson = wwf.__geo_interface__
        joined = gpd.sjoin(gdf, wwf[["Name", "geometry"]], how="left", predicate="intersects")
        joined = joined[~joined.index.duplicated(keep="first")]
        gdf["wwf_name"] = joined["Name"]
    else:
        gdf["wwf_name"] = None

    # Color layers
    layers = [
        Layer(color, 0.5 if color == UNASSIGNED_COLOR else 0.75, group.__geo_interface__)
        for color, group in gdf.groupby("color")
    ]

    # Z-score layers (quantile bins → red-yellow-green gradient)
    z_score_layers: list[Layer] = []
    if os.path.exists(Z_SCORE_FILE):
        z_df = pd.read_csv(Z_SCORE_FILE)
        z_df["field_id"] = z_df["name"].astype(str)
        z_df["bin"] = pd.qcut(z_df["0"], N_ZSCORE_BINS, labels=False, duplicates="drop")
        n_actual_bins = z_df["bin"].nunique()
        for bin_idx, group in z_df.groupby("bin"):
            color   = z_score_bin_color(int(bin_idx), n_actual_bins)
            subset  = gdf[gdf["field_id"].isin(group["field_id"])]
            if not subset.empty:
                z_score_layers.append(Layer(color, 0.75, subset.__geo_interface__))
        # Grey layer for fields with no z-score
        unscored = gdf[~gdf["field_id"].isin(z_df["field_id"])]
        if not unscored.empty:
            z_score_layers.append(Layer(UNASSIGNED_COLOR, 0.5, unscored.__geo_interface__))

    # WhatsApp markers
    markers_geojson: dict | None = None
    if os.path.exists(MARKERS_FILE):
        with open(MARKERS_FILE) as f:
            markers_geojson = json.load(f)

    # Map centre (compute centroid once)
    centroid   = boundaries.geometry.unary_union.centroid
    map_center = [centroid.y, centroid.x]

    # Pre-compute per-field lookup dicts
    field_props_map: dict[str, dict] = {}
    field_geojson_map: dict[str, dict] = {}
    for field_id, group in gdf.groupby("field_id"):
        row = group.iloc[0]
        raw_block   = row["block_id"]
        raw_cluster = row["cluster"]
        field_props_map[field_id] = {
            "block_id": "N/A" if int(raw_block) == -1 else int(raw_block),
            "cluster":  "N/A" if int(raw_cluster) == -1 else int(raw_cluster),
            "wwf_name": row.get("wwf_name") or "—",
        }
        field_geojson_map[field_id] = group.__geo_interface__

    return AppData(
        vi_log=vi_log,
        gdf=gdf,
        layers=layers,
        z_score_layers=z_score_layers,
        wwf_geojson=wwf_geojson,
        map_center=map_center,
        field_props_map=field_props_map,
        field_geojson_map=field_geojson_map,
        markers_geojson=markers_geojson,
    )


# Module-level singleton for the Dash app
app_data: AppData = load_data()
