import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import re
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from matplotlib import cm
import os

def cluster_time_series_and_plot(time_series_df, n_clusters=6):
    """
    Cluster fields based on their full vegetation index time series
    without normalization.

    Parameters
    ----------
    time_series_df : DataFrame
        rows = dates
        columns = fields
    """

    # Convert to clustering format
    # rows = fields
    # columns = time steps
    X = time_series_df.T.values

    # Run clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)

    clusters = pd.Series(labels, index=time_series_df.columns)
    # Plot each cluster
    for cluster_id in range(n_clusters):

        fields_in_cluster = clusters[clusters == cluster_id].index

        cluster_df = time_series_df[fields_in_cluster]

        print(f"Cluster {cluster_id}: {len(fields_in_cluster)} fields")

        plt.figure(figsize=(8,5))

        cluster_df.plot(
            ax=plt.gca(),
            legend=False,
            alpha=0.15,
            color="blue"
        )

        # plot cluster mean curve
        cluster_df.mean(axis=1).plot(
            ax=plt.gca(),
            color="red",
            linewidth=3
        )

        plt.title(f"Cluster {cluster_id}")
        plt.xlabel("Date")
        plt.ylabel("Vegetation Index")
        plt.show()

    return clusters

if __name__ == '__main__':
    season = 'kharif'
    crop_id = 'shahmeer'
    log_file = f"{season}_{crop_id}_field_veg_index_stats.csv"
    log_file_rao = f"../kharif_{crop_id}_field_veg_index_stats.csv"

    show_z_plots = False


    asset_dir = "/Users/daoud/PycharmAssets/shahmeer_farms"
    boundaries_file = f"{asset_dir}/shahmeer_drawn_named.geojson"

    if os.path.exists(boundaries_file):

        boundaries = gpd.read_file(boundaries_file)
        boundaries.boundary.plot()
        plt.show()
    else:
        raise ValueError('no boundaries file')

    ndvi_log = pd.read_csv(log_file)
    ndvi_log = ndvi_log.sort_values(by='date')
    ndvi_log_select = ndvi_log[ndvi_log[('name')].isin([720, 821])]

    duplicate_groups = ndvi_log.groupby(['name', 'date']).filter(lambda x: len(x) > 1)
    ndvi_log = ndvi_log[~ndvi_log.index.isin(duplicate_groups.index)]

    veg_idx = 'ndre'
    veg_idx_n = ['ndvi', 'cire', 'evi', 'ndre']
    for plot_idx, veg_idx_i in enumerate(veg_idx_n):
        ndvi_log_clean = ndvi_log[~ndvi_log['ndvi_mean'].isna()]
        # ndvi_log_clean = ndvi_log_clean[ndvi_log_clean['ndvi_mean']< 1]
        # ndvi_log_clean = ndvi_log_clean[ndvi_log_clean['evi_mean']< 4]

        # ndvi_log_clean = ndvi_log_clean[ndvi_log_clean['cire_mean']< 10]
        print (len(ndvi_log_clean))
        # commented line below is incorrrect, it should avg the std and see if above a threshold for an id, i.e the value of the id is not homogenous
        field_std = ndvi_log_clean.groupby('name')[f'{veg_idx_i}_std'].mean()

        # Keep only fields that are consistent (below threshold)
        threshold = 0.1
        fields_to_keep_by_std = field_std[field_std < threshold].index

        # Filter the main dataframe
        ndvi_log_clean = ndvi_log_clean[ndvi_log_clean['name'].isin(fields_to_keep_by_std)]
        print (len(ndvi_log_clean))


        time_series_df = ndvi_log_clean.pivot(index='date', columns='name', values=f'{veg_idx_i}_mean')
        time_series_df = time_series_df.interpolate(method='linear')
        time_series_df = time_series_df.fillna(method='bfill').fillna(method='ffill')

        clusters = cluster_time_series_and_plot(time_series_df, n_clusters=20)
        clusters_df = clusters.reset_index()
        clusters_df.columns = ["name", "cluster"]
        boundaries_clustered = pd.merge(boundaries, clusters_df, left_on='Name', right_on='name')

        clusters_to_keep = [2]

        # boundaries_clustered = boundaries_clustered[boundaries_clustered['cluster'].isin(clusters_to_keep)]
        boundaries_clustered.plot(
            column="cluster",
            cmap="tab10",
            legend=True,
            edgecolor="black"
        )

        plt.title("Field Clusters")
        plt.show()


        fields_to_keep = clusters[clusters.isin(clusters_to_keep)].index

        time_series_filtered = time_series_df[fields_to_keep]
        time_series_df = time_series_filtered
        time_series_df.plot(title=f'plant {veg_idx_i} levels', color="blue",
    alpha=0.05).legend(
            bbox_to_anchor=(1.0, 1.0),
            fontsize='small',
        )
        plt.show()

        print (f'plotted {veg_idx_i}')


        vi_mean = time_series_df.mean(axis = 1)
        vi_std =  time_series_df.std(axis = 1)

        z_scores_df = (time_series_df.sub(vi_mean, axis=0)).div(vi_std, axis=0)
        # z_scores_df.index = pd.to_datetime(z_scores_df.index)

        z_scores_df = z_scores_df[-20:]
        z_score_mean_early = z_scores_df.mean(axis=0)
        z_score_mean_early.to_csv(f'{crop_id}_{veg_idx_i}_z_scores_norm.csv')

        if show_z_plots:
            z_scores_df.plot(color = 'blue', alpha = 0.05)
            plt.show()

        boundaries_filtered = boundaries_clustered
        boundaries_filtered['z_score'] = boundaries_filtered['name'].map(z_score_mean_early)

        boundaries_filtered.plot(
            column='z_score',
            cmap='RdYlGn',
            legend=True,
            edgecolor='black'
        )
        plt.title(f'{veg_idx_i} Fields Colored by Z-score')
        plt.show()