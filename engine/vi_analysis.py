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
from sklearn.decomposition import PCA

def cluster_time_series_and_plot(time_series_df, n_clusters=6, use_pca=False, n_components=5):
    """
    Cluster fields based on their full vegetation index time series.
    Optionally reduce dimensionality using PCA.
    """
    X = time_series_df.T.values  # shape: (fields, time_steps)

    if use_pca:
        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X)
        X = X_pca  # use PCA-reduced data for clustering

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
        cluster_df.plot(ax=plt.gca(), legend=False, alpha=0.15, color="blue")
        cluster_df.mean(axis=1).plot(ax=plt.gca(), color="red", linewidth=3)
        plt.title(f"Cluster {cluster_id}")
        plt.xlabel("Date")
        plt.ylabel("Vegetation Index")
        plt.show()

    return clusters

# Install tslearn if not already installed
# pip install tslearn

from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kshape_cluster_time_series(time_series_df, n_clusters=6, reference_curve=None):
    """
    Cluster fields based on their full vegetation index time series using k-Shape.

    Parameters
    ----------
    time_series_df : DataFrame
        rows = dates
        columns = fields
    n_clusters : int
        Number of clusters
    reference_curve : array-like, optional
        A reference curve (e.g., corn NDVI) to plot for comparison
    """

    # Convert to (fields, time_steps, 1) for tslearn
    X = time_series_df.T.values[:, :, np.newaxis]

    # Normalize time series
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    # Run k-Shape
    ks = KShape(n_clusters=n_clusters, random_state=0)
    labels = ks.fit_predict(X_scaled)

    clusters = pd.Series(labels, index=time_series_df.columns)

    # Plot each cluster
    for cluster_id in range(n_clusters):
        fields_in_cluster = clusters[clusters == cluster_id].index
        cluster_df = time_series_df[fields_in_cluster]

        print(f"Cluster {cluster_id}: {len(fields_in_cluster)} fields")

        plt.figure(figsize=(8,5))
        # Individual fields
        cluster_df.plot(ax=plt.gca(), legend=False, alpha=0.15, color="blue")
        # Cluster mean
        cluster_df.mean(axis=1).plot(ax=plt.gca(), color="red", linewidth=3)
        # Reference curve if provided
        if reference_curve is not None:
            ref_curve = pd.Series(reference_curve, index=time_series_df.index)
            ref_curve.plot(ax=plt.gca(), color="green", linestyle="--", linewidth=2, label="Corn reference")
            plt.legend()

        plt.title(f"k-Shape Cluster {cluster_id}")
        plt.xlabel("Date")
        plt.ylabel("Vegetation Index")
        plt.show()

    return clusters

# Example NDVI curve for corn
corn_reference = [0.2, 0.2, 0.4, 0.6, 0.8, 0.75, 0.5, 0.3, 0.2]
corn_reference = [0.2] * 2 + corn_reference[:-2]
x_orig = np.linspace(0, 1, len(corn_reference))
x_orig = np.linspace(0, 1, len(corn_reference))

# Create target x-axis for 36 points
x_new = np.linspace(0, 1, 36)

# Interpolate
corn_interp = np.interp(x_new, x_orig, corn_reference)



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
    veg_idx_n = ['ndvi', 'evi', 'ndre']
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

        # clusters = cluster_time_series_and_plot(time_series_df, use_pca=False, n_clusters=20)
        clusters = kshape_cluster_time_series(time_series_df, n_clusters=6, reference_curve=corn_interp)
        clusters_df = clusters.reset_index()
        clusters_df.columns = ["name", "cluster"]
        boundaries_clustered = pd.merge(boundaries, clusters_df, left_on='Name', right_on='name')

        clusters_to_keep = [0, 5, 6]

        # boundaries_clustered = boundaries_clustered[boundaries_clustered['cluster'].isin(clusters_to_keep)]
        boundaries_clustered.plot(
            column="cluster",
            cmap="tab10",
            legend=True,
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