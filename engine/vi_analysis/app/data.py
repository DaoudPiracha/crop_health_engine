"""
data.py — Data loading and preparation for the VI Dash app.

Call load_data() to get an AppData instance. The module-level `app_data`
singleton is created at import time for use by the Dash app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import geopandas as gpd
import pandas as pd

from engine.vi_analysis.vi_analysis import block_colors, load_vi_log, rgb_to_hex
from engine.vi_analysis.app.theme import UNASSIGNED_COLOR
from engine.vi_analysis.app.config import ASSET_DIR, CROP, SEASON

# ---------------------------------------------------------------------------
# Paths (derived from config)
# ---------------------------------------------------------------------------

BOUNDARIES_FILE = f"{ASSET_DIR}/{CROP}_drawn_named.geojson"

_engine_root = os.path.join(os.path.dirname(__file__), "..")
LOG_FILE    = os.path.join(_engine_root, f"../{SEASON}_{CROP}_field_veg_index_stats.csv")
BLOCKS_FILE = os.path.join(_engine_root, f"../{CROP}_blocks.csv")
WWF_FILE    = os.path.join(_engine_root, f"../{CROP}_wwf_map.geojson")

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
    vi_log:          pd.DataFrame
    gdf:             gpd.GeoDataFrame
    layers:          list[Layer]
    wwf_geojson:     dict | None
    map_center:      list[float]
    field_props_map: dict[str, dict]    # field_id → {block_id, cluster, wwf_name}
    field_geojson_map: dict[str, dict]  # field_id → geojson FeatureCollection


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data() -> AppData:
    boundaries = gpd.read_file(BOUNDARIES_FILE).to_crs("epsg:4326")
    blocks_df  = pd.read_csv(BLOCKS_FILE)
    vi_log     = load_vi_log(LOG_FILE)

    # Field GeoDataFrame
    color_map = {bid: rgb_to_hex(rgb) for bid, rgb in block_colors(blocks_df).items()}
    gdf = boundaries[[NAME_COL, "geometry"]].merge(
        blocks_df, left_on=NAME_COL, right_on="name", how="left"
    )
    gdf["color"]    = gdf["block_id"].map(color_map).fillna(UNASSIGNED_COLOR)
    gdf["block_id"] = gdf["block_id"].fillna(-1).astype(int)
    gdf["cluster"]  = gdf["cluster"].fillna(-1).astype(int)
    gdf = gdf.rename(columns={NAME_COL: "field_id"})

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
        wwf_geojson=wwf_geojson,
        map_center=map_center,
        field_props_map=field_props_map,
        field_geojson_map=field_geojson_map,
    )


# Module-level singleton for the Dash app
app_data: AppData = load_data()
