import logging

from engine.pipeline_config import PipelineConfig
from engine.pipeline.run import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    asset_dir = "/Users/daoud/PycharmAssets/wattoo_farms"

    cfg = PipelineConfig(
        asset_dir=asset_dir,
        season="kharif",
        crop_id="wattoo",
        file_dir=f"{asset_dir}/*/PSScene/",
        boundaries_file=f"{asset_dir}/wattoo_farms.geojson",
        cluster_file="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/sufi/cluster_cire.csv",
        z_score_ts_file="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo_cire_z_scores_ts.csv",
        z_score_glob="/Users/daoud/PycharmProjects/DISA/ai4h_disa/experimental/wattoo/wattoo_cire_z_scores_norm.csv",
        target_crs="epsg:3857",
        only_visual=False,
        write_to_file=True,
        color_clusters=False,
        color_z_scores=True,
        show_z_ts_plots=False,
        reset_names=False,
        show_images_at_each_ts = False,
        unwanted_ids=["301", "302", "304", "153", "176", "170", "175", "172"],
        bbox_latlon=(30.6655, 30.676, 73.675377, 73.6815427),
    )

    run_pipeline(cfg)
