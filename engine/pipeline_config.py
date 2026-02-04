
from typing import List, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    asset_dir: str
    season: str
    crop_id: str

    file_dir: str  # glob root for images (already includes /*/PSScene/)
    boundaries_file: str

    cluster_file: str
    z_score_ts_file: str
    z_score_glob: str

    target_crs: str

    only_visual: bool
    write_to_file: bool

    color_clusters: bool
    color_z_scores: bool
    show_z_ts_plots: bool
    reset_names: bool
    show_images_at_each_ts: bool

    unwanted_ids: List[str]

    bbox_latlon: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
