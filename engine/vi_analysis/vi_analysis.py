import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


# ---------------------------------------------------------------------------
# Reference curves
# ---------------------------------------------------------------------------

def make_corn_reference(n_points: int = 36) -> np.ndarray:
    """Return an interpolated NDVI reference curve for corn over n_points timesteps."""
    keyframes = [0.2, 0.2, 0.4, 0.6, 0.8, 0.75, 0.5, 0.3, 0.2]
    keyframes = [0.2] * 2 + keyframes[:-2]
    x_orig = np.linspace(0, 1, len(keyframes))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_orig, keyframes)


# ---------------------------------------------------------------------------
# Data loading / preparation
# ---------------------------------------------------------------------------

def load_vi_log(csv_path: str) -> pd.DataFrame:
    """Load and clean the vegetation index CSV produced by the core pipeline."""
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="date")
    # Drop duplicate (name, date) rows
    duplicates = df.groupby(["name", "date"]).filter(lambda x: len(x) > 1)
    df = df[~df.index.isin(duplicates.index)]
    return df


def build_time_series(df: pd.DataFrame, veg_idx: str, std_threshold: float = 0.1) -> pd.DataFrame:
    """
    Pivot the log into a (dates × fields) time series for one vegetation index.

    Fields whose mean intra-date std exceeds std_threshold are dropped as too
    heterogeneous for reliable clustering.
    """
    df_clean = df[~df["ndvi_mean"].isna()].copy()

    field_std = df_clean.groupby("name")[f"{veg_idx}_std"].mean()
    fields_to_keep = field_std[field_std < std_threshold].index
    df_clean = df_clean[df_clean["name"].isin(fields_to_keep)]

    ts = df_clean.pivot(index="date", columns="name", values=f"{veg_idx}_mean")
    ts = ts.interpolate(method="linear")
    ts = ts.bfill().ffill()
    return ts


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def kmeans_cluster(time_series_df: pd.DataFrame, n_clusters: int = 6,
                   use_pca: bool = False, n_components: int = 5) -> pd.Series:
    """
    Cluster fields by their full vegetation index time series using K-Means.

    Parameters
    ----------
    time_series_df : DataFrame  (rows=dates, columns=fields)
    n_clusters     : number of clusters
    use_pca        : reduce dimensionality with PCA before clustering
    n_components   : number of PCA components (ignored if use_pca=False)

    Returns
    -------
    Series mapping field name → cluster label
    """
    X = time_series_df.T.values  # (fields, time_steps)

    if use_pca:
        X = PCA(n_components=n_components, random_state=0).fit_transform(X)

    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    return pd.Series(labels, index=time_series_df.columns)


def kshape_cluster(time_series_df: pd.DataFrame, n_clusters: int = 6) -> pd.Series:
    """
    Cluster fields by their full vegetation index time series using k-Shape.

    k-Shape is shape-based and scale-invariant, making it well suited for
    comparing growth curve trajectories regardless of absolute NDVI level.

    Parameters
    ----------
    time_series_df : DataFrame  (rows=dates, columns=fields)
    n_clusters     : number of clusters

    Returns
    -------
    Series mapping field name → cluster label
    """
    X = time_series_df.T.values[:, :, np.newaxis]  # (fields, time_steps, 1)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    labels = KShape(n_clusters=n_clusters, random_state=0).fit_predict(X_scaled)
    return pd.Series(labels, index=time_series_df.columns)


# ---------------------------------------------------------------------------
# Z-score anomaly detection
# ---------------------------------------------------------------------------

def compute_z_scores(time_series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise each field's time series relative to the cross-field mean and std
    at each date. Returns a DataFrame of the same shape with z-score values.
    """
    vi_mean = time_series_df.mean(axis=1)
    vi_std = time_series_df.std(axis=1)
    return time_series_df.sub(vi_mean, axis=0).div(vi_std, axis=0)


def write_z_scores(z_scores_df: pd.DataFrame, crop_id: str, veg_idx: str) -> str:
    """Write the mean z-score per field to CSV. Returns the output path."""
    out_path = f"{crop_id}_{veg_idx}_z_scores_norm.csv"
    z_scores_df.mean(axis=0).to_csv(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Block assignment (spatial connected components within clusters)
# ---------------------------------------------------------------------------

def _union_find_components(nodes: list, edges: list) -> dict:
    """
    Simple union-find. Returns a dict mapping each node to its root/component id.
    """
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in edges:
        union(a, b)

    return {n: find(n) for n in nodes}


def build_blocks(boundaries: gpd.GeoDataFrame, clusters: pd.Series,
                 name_col: str = "Name", buffer_m: float = 3.0) -> pd.DataFrame:
    """
    Assign each field to a block. A block is a set of spatially adjacent
    fields that share the same cluster label.

    Adjacency is determined by buffering each field by buffer_m metres before
    checking for intersection — this handles tiny gaps between field boundaries.
    The boundaries GeoDataFrame must be in a projected (metre-based) CRS for
    this to be meaningful; reproject before calling if needed.

    Parameters
    ----------
    boundaries : GeoDataFrame with field polygons
    clusters   : Series mapping field name → cluster label
    name_col   : column in boundaries that holds the field name
    buffer_m   : buffer distance in metres used for adjacency detection

    Returns
    -------
    DataFrame with columns: name, cluster, block_id
    """
    gdf = boundaries[[name_col, "geometry"]].copy()
    gdf["cluster"] = gdf[name_col].map(clusters)
    gdf = gdf.dropna(subset=["cluster"])
    gdf["cluster"] = gdf["cluster"].astype(int)

    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(buffer_m)

    block_rows = []
    block_counter = 0

    for cluster_id, group in gdf.groupby("cluster"):
        names = group[name_col].tolist()
        buf_group = gdf_buffered[gdf_buffered[name_col].isin(names)]

        # Spatial self-join to find adjacent pairs within this cluster
        joined = gpd.sjoin(
            buf_group[[name_col, "geometry"]],
            buf_group[[name_col, "geometry"]],
            how="inner",
            predicate="intersects",
        )
        # Remove self-matches
        edges = [
            (row[f"{name_col}_left"], row[f"{name_col}_right"])
            for _, row in joined.iterrows()
            if row[f"{name_col}_left"] != row[f"{name_col}_right"]
        ]

        components = _union_find_components(names, edges)

        # Map component root → stable block_id
        root_to_block = {}
        for root in set(components.values()):
            root_to_block[root] = block_counter
            block_counter += 1

        for name in names:
            block_rows.append({
                "name": name,
                "cluster": cluster_id,
                "block_id": root_to_block[components[name]],
            })

    return pd.DataFrame(block_rows)


def dissolve_blocks(boundaries: gpd.GeoDataFrame, blocks_df: pd.DataFrame,
                    name_col: str = "Name") -> gpd.GeoDataFrame:
    """
    Merge adjacent same-cluster fields into single block polygons.

    Returns a GeoDataFrame with one row per block_id containing the dissolved
    geometry and the cluster label.
    """
    gdf = boundaries[[name_col, "geometry"]].merge(
        blocks_df, left_on=name_col, right_on="name"
    )
    dissolved = gdf.dissolve(by="block_id", aggfunc={"cluster": "first"}).reset_index()
    return dissolved


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_clusters(time_series_df: pd.DataFrame, clusters: pd.Series,
                  veg_idx: str, reference_curve: np.ndarray = None) -> None:
    """Plot each cluster's individual field curves and their mean."""
    for cluster_id in sorted(clusters.unique()):
        fields = clusters[clusters == cluster_id].index
        cluster_df = time_series_df[fields]

        print(f"Cluster {cluster_id}: {len(fields)} fields")
        fig, ax = plt.subplots(figsize=(8, 5))
        cluster_df.plot(ax=ax, legend=False, alpha=0.15, color="blue")
        cluster_df.mean(axis=1).plot(ax=ax, color="red", linewidth=3, label="cluster mean")

        if reference_curve is not None:
            ref = pd.Series(reference_curve, index=time_series_df.index)
            ref.plot(ax=ax, color="green", linestyle="--", linewidth=2, label="reference")
            ax.legend()

        ax.set_title(f"Cluster {cluster_id} — {veg_idx}")
        ax.set_xlabel("Date")
        ax.set_ylabel(veg_idx)
        plt.show()


def plot_cluster_map(boundaries: gpd.GeoDataFrame, clusters: pd.Series) -> None:
    """Plot fields on a map coloured by cluster assignment."""
    clusters_df = clusters.reset_index()
    clusters_df.columns = ["name", "cluster"]
    merged = boundaries.merge(clusters_df, left_on="Name", right_on="name")
    merged.plot(column="cluster", cmap="tab10", legend=True)
    plt.title("Field Clusters")
    plt.show()


def plot_z_score_map(boundaries: gpd.GeoDataFrame, clusters: pd.Series,
                     z_score_mean: pd.Series, veg_idx: str) -> None:
    """Plot fields on a map coloured by mean z-score."""
    clusters_df = clusters.reset_index()
    clusters_df.columns = ["name", "cluster"]
    merged = boundaries.merge(clusters_df, left_on="Name", right_on="name")
    merged["z_score"] = merged["name"].map(z_score_mean)
    merged.plot(column="z_score", cmap="RdYlGn", legend=True, edgecolor="black")
    plt.title(f"{veg_idx} Fields Colored by Z-score")
    plt.show()


def block_colors(blocks_df: pd.DataFrame, saturation: float = 0.45,
                  value: float = 0.80, hue_jitter: float = 0.06) -> dict:
    """
    Assign each block a muted colour where:
    - Hue family is determined by cluster (evenly spaced around the colour wheel)
    - Hue is jittered slightly per block within the cluster for distinction
    - Saturation and value are fixed for a consistent muted/pastel look

    Returns a dict mapping block_id → RGB tuple.
    """
    import colorsys

    clusters = sorted(blocks_df["cluster"].unique())
    n_clusters = len(clusters)
    # Evenly space base hues; offset by 0.05 to avoid starting on pure red
    base_hue = {c: (i / n_clusters + 0.05) % 1.0 for i, c in enumerate(clusters)}

    rng = np.random.default_rng(seed=786)
    colors = {}
    for cluster_id, group in blocks_df.groupby("cluster"):
        block_ids = group["block_id"].unique()
        for bid in block_ids:
            jitter = rng.uniform(-hue_jitter, hue_jitter)
            h = (base_hue[cluster_id] + jitter) % 1.0
            colors[bid] = colorsys.hsv_to_rgb(h, saturation, value)
    return colors


def rgb_to_hex(rgb: tuple) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def plot_block_map_interactive(boundaries: gpd.GeoDataFrame, blocks_df: pd.DataFrame,
                                name_col: str = "Name",
                                overlay: gpd.GeoDataFrame = None,
                                overlay_name_col: str = "Name"):
    """
    Interactive version of plot_block_map using leafmap with a basemap.

    If overlay is provided, each field is spatially joined to the overlay so
    that clicking/hovering on a field shows the corresponding overlay polygon
    name (e.g. WWF block name) in the popup.

    Returns the leafmap.Map object so it can be displayed in a notebook
    or saved with m.to_html().
    """
    import leafmap.foliumap as leafmap

    block_colors = {bid: rgb_to_hex(rgb) for bid, rgb in block_colors(blocks_df).items()}

    gdf = boundaries[[name_col, "geometry"]].merge(blocks_df, left_on=name_col, right_on="name")
    gdf["color"] = gdf["block_id"].map(block_colors)
    gdf = gdf.to_crs("epsg:4326")

    # Spatially join WWF names onto fields so they appear in popups
    if overlay is not None and overlay_name_col in overlay.columns:
        wwf = overlay[[overlay_name_col, "geometry"]].to_crs("epsg:4326")
        joined = gpd.sjoin(gdf, wwf, how="left", predicate="intersects")
        # If a field touches multiple WWF polygons take the first match
        joined = joined[~joined.index.duplicated(keep="first")]
        joined = joined.rename(columns={f"{overlay_name_col}_right": "WWF Name"})
        gdf = joined.drop(columns=["index_right", f"{name_col}_left"], errors="ignore")
    else:
        gdf["WWF Name"] = None

    gdf = gdf.rename(columns={"name": "field_id", "cluster": "crop_id", "block_id": "Block ID"})

    unassigned = boundaries[~boundaries[name_col].isin(blocks_df["name"])].copy()
    unassigned = unassigned.to_crs("epsg:4326")

    center = gdf.geometry.unary_union.centroid
    m = leafmap.Map(center=[center.y, center.x], zoom=14)
    m.add_basemap("SATELLITE")

    if not unassigned.empty:
        m.add_gdf(
            unassigned,
            style_function=lambda _: {
                "fillColor": "#555555", "color": "black",
                "weight": 0.5, "fillOpacity": 0.5,
            },
            layer_name="Unassigned fields",
        )

    m.add_gdf(
        gdf,
        style_function=lambda f: {
            "fillColor": f["properties"]["color"], "color": "black",
            "weight": 0.5, "fillOpacity": 0.8,
        },
        layer_name="Blocks",
    )

    if overlay is not None:
        m.add_gdf(
            overlay.to_crs("epsg:4326"),
            style_function=lambda _: {
                "fillColor": "none", "color": "black",
                "weight": 2.5, "fillOpacity": 0,
            },
            layer_name="WWF map",
        )

    return m


def plot_block_map(boundaries: gpd.GeoDataFrame, blocks_df: pd.DataFrame,
                   name_col: str = "Name",
                   overlay: gpd.GeoDataFrame = None) -> None:
    """
    Plot individual field polygons coloured by block.

    Colour hue family = cluster (so same-cluster fields share a visual family),
    slight hue variation per block within a cluster. Muted saturation and fixed
    brightness keep the palette easy on the eye.

    Parameters
    ----------
    overlay : optional GeoDataFrame whose boundaries are drawn on top in red
    """
    block_colors = block_colors(blocks_df)

    gdf = boundaries[[name_col, "geometry"]].merge(blocks_df, left_on=name_col, right_on="name")
    gdf["color"] = gdf["block_id"].map(block_colors)

    # Fields that were filtered out (high std) — not in blocks_df
    unassigned = boundaries[~boundaries[name_col].isin(blocks_df["name"])]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unassigned fields first (behind) in grey
    if not unassigned.empty:
        unassigned.plot(ax=ax, color="#555555", edgecolor="black", linewidth=0.5, alpha=0.5)

    for _, row in gdf.iterrows():
        gpd.GeoDataFrame([row], crs=boundaries.crs).plot(
            ax=ax, color=[row["color"]], edgecolor="black", linewidth=0.5
        )

    # Label each block once at its dissolved centroid
    for bid, group in gdf.groupby("block_id"):
        centroid = group.geometry.unary_union.centroid
        ax.annotate(str(bid), (centroid.x, centroid.y), fontsize=6, ha="center",
                    color="black", fontweight="bold")

    if overlay is not None:
        overlay.to_crs(boundaries.crs).boundary.plot(ax=ax, edgecolor="black", linewidth=2.5)

    ax.set_title("Fields coloured by AI crop similarity")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    season = "kharif"
    crop_id = "shahmeer"
    log_file = f"{season}_{crop_id}_field_veg_index_stats.csv"
    show_z_plots = False
    n_clusters = 6
    clusters_to_keep = [0, 5]           # set to None to keep all clusters
    std_threshold = 0.1
    veg_indices = ["ndvi", "evi", "ndre"]
    # CRS for adjacency detection — must be metre-based for buffer_m to be meaningful
    projected_crs = "epsg:32642"        # UTM zone 42N covers Pakistan

    asset_dir = "/Users/daoud/PycharmAssets/shahmeer_farms"
    boundaries_file = f"{asset_dir}/shahmeer_drawn_named.geojson"

    geo_file = f"./shahmeer_wwf_map.geojson"
    wwf_map = None
    if os.path.exists(geo_file):
        wwf_map = gpd.read_file(geo_file)
        wwf_map.boundary.plot(edgecolor="red", linewidth=0.5)
        plt.title("shahmeer wwf map")
        plt.show()

    if not os.path.exists(boundaries_file):
        raise ValueError(f"Boundaries file not found: {boundaries_file}")
    boundaries = gpd.read_file(boundaries_file)
    boundaries.boundary.plot()
    plt.title("shahmeer AI generated field map")

    plt.show()

    ndvi_log = load_vi_log(log_file)
    corn_ref = make_corn_reference(n_points=36)

    # Use the first veg index to define clusters and blocks, then reuse for all
    primary_idx = veg_indices[0]
    print(f"\n--- Building clusters from {primary_idx} ---")
    ts_primary = build_time_series(ndvi_log, primary_idx, std_threshold=std_threshold)
    clusters = kshape_cluster(ts_primary, n_clusters=n_clusters)

    # Build blocks: adjacent same-cluster fields → one block
    boundaries_proj = boundaries.to_crs(projected_crs)
    blocks_df = build_blocks(boundaries_proj, clusters, buffer_m=3.0)
    dissolved = dissolve_blocks(boundaries_proj, blocks_df)

    print(f"Fields: {len(clusters)}  →  Blocks: {dissolved['block_id'].nunique()}")
    blocks_df.to_csv(f"{crop_id}_blocks.csv", index=False)
    print(f"Field→block mapping written to {crop_id}_blocks.csv")

    plot_block_map(boundaries, blocks_df, overlay=wwf_map)

    m = plot_block_map_interactive(boundaries, blocks_df, overlay=wwf_map)
    m.to_html(f"{crop_id}_block_map.html")
    print(f"Interactive map saved to {crop_id}_block_map.html")

    for veg_idx in veg_indices:
        print(f"\n--- {veg_idx} ---")
        ts = build_time_series(ndvi_log, veg_idx, std_threshold=std_threshold)
        print(f"Fields after std filter: {ts.shape[1]}")

        plot_clusters(ts, clusters, veg_idx, reference_curve=corn_ref)
        plot_cluster_map(boundaries, clusters)

        if clusters_to_keep is not None:
            ts = ts[clusters[clusters.isin(clusters_to_keep)].index]

        ts.plot(title=f"plant {veg_idx} levels", color="blue", alpha=0.05).legend(
            bbox_to_anchor=(1.0, 1.0), fontsize="small"
        )
        plt.show()

        z_scores = compute_z_scores(ts)
        z_scores_recent = z_scores.iloc[-20:]
        out = write_z_scores(z_scores_recent, crop_id, veg_idx)
        print(f"Z-scores written to: {out}")

        if show_z_plots:
            z_scores_recent.plot(color="blue", alpha=0.05)
            plt.show()

        plot_z_score_map(boundaries, clusters, z_scores_recent.mean(axis=0), veg_idx)
        print(f"Plotted {veg_idx}")
