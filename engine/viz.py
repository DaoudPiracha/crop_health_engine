import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    gdf = gpd.read_file('/Users/daoud/PycharmProjects/kisaan_ai/crop_health_scoring/engine/shahmeer_drawn_geo.geojson')
    z_scores = pd.read_csv('/Users/daoud/PycharmProjects/kisaan_ai/crop_health_scoring/engine/shahmeer_cire_z_scores_norm.csv')
    z_scores = z_scores.rename(columns={"0": "z_score"})
    z_scores = z_scores[z_scores['z_score']>-0.5]
    z_scores.to_csv('/Users/daoud/PycharmProjects/kisaan_ai/crop_health_scoring/engine/shahmeer_cire_z_scores_norm_filtered.csv')


    merged = gdf.merge(z_scores, left_on='Name', right_on='name')

    fig, ax = plt.subplots(figsize=(10, 10))

    merged.plot(
        column="z_score",
        cmap="RdYlGn",  # red = bad, green = good (nice for crop health)
        legend=True,
        ax=ax,
        edgecolor="black",
    )

    ax.set_title("Crop Health Z-Scores")
    ax.axis("off")

    plt.show()