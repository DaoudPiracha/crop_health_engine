import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import re
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from matplotlib import cm



sowing_date = pd.Timestamp(year=2024, month=7, day=1)  # Start of July
tillering_start = pd.Timestamp(year=2024, month=7, day=18)  # 3 weeks after
stem_elongation_start = pd.Timestamp(year=2024, month=8, day=7)  # 3 weeks after
flowering_start = pd.Timestamp(year=2024, month=9, day=22)  # End of September
maturity_start = pd.Timestamp(year=2024, month=10, day=15)  # Mid-October
harvesting_end = pd.Timestamp(year=2024, month=10, day=28)  # End of October

crop_stage_dates = [("sowing", sowing_date), ('tillering', tillering_start), ('stem elongation', stem_elongation_start),
                    ('flowering', flowering_start), ('maturity', maturity_start), ('harvest end', harvesting_end)]

fields_to_yields = {

    'lp_18': {
        'Feild 20': 80,
        'Field 21': 65,
        'Field 22': 74,
        # 'Field 23': 62,
        # 'Field 26': 62,
        # 'Field 29': 62,
        'field 6': -2,  # water issue

    },
    '17_18': {
        'field 1': 34,
        'field 10a': 123 / 2.81,
        'field 10b': 64 / 1.58,
        'field 10c': 43,
        'field 10e': 39 / 0.89,
        'field 10f': 21 / 0.51,

        # 'field 10d': 52 / 1.07,  # supri

        # 'Field 28': -1,
        # 'field 7': -1,
        # 'field 5': -1,

        'field 6': -2,

    },

    'supri': {
        'Feild 20': 80,

        'field 10d': 52 / 1.07,  # supri
        'field 2': 52,
        'field 6': -2,

    },
    'gold_1_hybrid': {
        'Field 30': 57,  # supri
        'Field 31': 57,  # supri

        'field 2': 52,
        'field 6': -2,

    }

}

fields = {
    'lp_18': {
        'Feild 20': 80,
        'Field 21': 65,
        'Field 22': 74,
        'Field 23': 62,
        'Field 26': 62,
        'Field 29': 62,
        'field 6': -2,  # water issue

    },
    '17_18': {
        'field 1': 34,
        'field 10a': 123 / 2.81,
        'field 10b': 64 / 1.58,
        'field 10c': 43,
        'field 10e': 39 / 0.89,
        'field 10f': 21 / 0.51,

        # 'field 10d': 52 / 1.07,  # supri

        # 'Field 28': -1,
        # 'field 7': -1,
        # 'field 5': -1,

        'field 6': -2,

    },

    'supri': {
        'field 10d': 52 / 1.07,  # supri
        'field 2': 52,
        'field 6': -2,

    },
    'gold_1_hybrid': {
        'Field 30': 57,  # supri
        'Field 31': 57,  # supri

        'field 2': 52,
        'field 6': -2,

    }

}


def plot_ndvi_time_series(ndvi_log, veg_indice='ndvi', variety='lp_18'):
    # Get the unique field IDs
    ndvi_log = ndvi_log.sort_values(by='date')
    ndvi_log = ndvi_log[ndvi_log['date']!='2024-07-15']

    ndvi_log['date'] = pd.to_datetime(ndvi_log['date'])

    field_ids = ndvi_log['name'].unique()

    # Create a plot for each field
    plt.figure(figsize=(10, 6))


    fig, ax = plt.subplots(figsize=(10, 6))
    for field_id in field_ids[:20]:
        if field_id:# in [4340, 3780, 3750, 4125, 4146, 4159 ]: # in fields_to_yields[variety].keys():
            # Filter the data for the current field
            field_data = ndvi_log[ndvi_log['name'] == field_id]

            # Plot NDVI Mean with error bars (std deviation)
            ax.errorbar(field_data['date'], field_data[f'{veg_indice}_mean'], yerr=field_data[f'{veg_indice}_std'],
                         label=f'Field: {field_id}', fmt='-o')

    for stage_name, stage_change_date in crop_stage_dates:
        ax.axvline(stage_change_date, color='red', linestyle='--')  # Adjust the date as needed
        ax.text(
            stage_change_date,  # X-coordinate
            ax.get_ylim()[1] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),  # Position slightly above y-axis
            stage_name,  # Label
            rotation=0,  # Rotate the label for better readability
            # ha='center',
            va='top',
            fontsize=9,
            color='black'
        )
    # Set plot labels and title
    plt.xlabel('Date')
    plt.xticks(rotation=45)  # Rotate x-axis tick labels by 45 degrees
    plt.ylabel(f'{veg_indice} Mean')
    plt.title(f'{veg_indice} Time Series for Each Field')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title="Crop Stages")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def determine_crop_growth_stage(date):
    """
    Determines the growth stage of a rice crop based on the date.
    Sowing is assumed in July, and harvesting is at the end of October.

    Parameters:
    date (datetime): Date to determine the stage for.

    Returns:
    str: Crop growth stage.
    """
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    # Define typical dates for each stage (can be adjusted based on local knowledge)


    # Determine the stage based on the date
    if date < tillering_start:
        return 'Seedling'
    elif tillering_start <= date < stem_elongation_start:
        return 'Tillering'
    elif stem_elongation_start <= date < flowering_start:
        return 'Stem Elongation'
    elif flowering_start <= date < maturity_start:
        return 'Flowering'
    elif maturity_start <= date <= harvesting_end:
        return 'Maturity'
    else:
        return 'Post-Harvest'


def ndvi_alert(row):
    """
    Generates an NDVI-based alert for a single row of data based on growth stage and NDVI value.

    Parameters:
    row (Series): A row of data containing 'ndvi' and 'growth_stage'.

    Returns:
    str: An alert message or 'No alerts' if no issues are detected.
    """
    growth_stage = row['growth_stage']
    ndvi = row['ndvi_mean']

    # Define alert rules based on growth stage
    if growth_stage == 'Seedling':
        if ndvi < 0.2 or ndvi > 0.4:
            return 'NDVI out of range for Seedling stage (Expected: 0.2-0.4)'
    elif growth_stage == 'Tillering':
        if ndvi < 0.4 or ndvi > 0.6:
            return 'NDVI out of range for Tillering stage (Expected: 0.4-0.6)'
    elif growth_stage == 'Stem Elongation':
        if ndvi < 0.6 or ndvi > 0.8:
            return 'NDVI out of range for Stem Elongation stage (Expected: 0.6-0.8)'
    elif growth_stage == 'Flowering':
        if ndvi < 0.7 or ndvi > 0.9:
            return 'NDVI out of range for Flowering stage (Expected: 0.7-0.9)'
    elif growth_stage == 'Maturity':
        if ndvi < 0.3 or ndvi > 0.5:
            return 'NDVI out of range for Maturity stage (Expected: 0.3-0.5)'

    # Default case if no alerts
    return 'No alerts'

# todo: update to use dynamic thresholds
def diagnose_field_issues(row):
    """
    Diagnoses potential issues based on changes in vegetation indices during different growth stages.

    Parameters:
    row (Series): A row of data containing 'growth_stage', 'NDVI', 'EVI', 'NDRE', 'NDWI', and other indices.

    Returns:
    str: Diagnosis message or 'No significant issues detected'.
    """
    growth_stage = row['growth_stage']
    ndvi = round(row['ndvi_mean'], 2)
    evi = round(row['evi_mean'], 2)
    ndre = round(row['ndre_mean'], 2)
    # ndwi = row['ndwi_mean', None]
    # cire = round(row.get('cire_mean', None), 2)  # Optional, if available

    diagnosis = []

    # Rules based on growth stage and indices
    if growth_stage == 'Seedling':
        if ndvi < 0.2 or ndvi > 0.4:
            diagnosis.append(f'NDVI{evi} out of expected range for Seedling (0.2-0.4)')
        if evi < 0.2 or evi > 0.4:
            diagnosis.append(f'EVI{evi} out of expected range for Seedling (0.2-0.4)')
        # if ndwi < 0.4 or ndwi > 0.6:
        #     diagnosis.append('Potential water stress (NDWI out of 0.4-0.6)')

    elif growth_stage == 'Tillering':
        if ndvi < 0.4 or ndvi > 0.6:
            diagnosis.append(f'NDVI{ndvi} out of expected range for Tillering (0.4-0.6)')
        if ndre < 0.4 or ndre > 0.6:
            diagnosis.append(f'NDRE{ndre} suggests potential nutrient imbalance (Expected: 0.4-0.6)')
        # if ndwi < 0.5 or ndwi > 0.7:
        #     diagnosis.append('Possible water stress (NDWI out of 0.5-0.7)')
        if evi < 0.4 or evi > 0.6:
            diagnosis.append(f'EVI{evi} suggests canopy issues (Expected: 0.4-0.6)')

    elif growth_stage == 'Stem Elongation':
        if ndvi < 0.6 or ndvi > 0.8:
            diagnosis.append(f'NDVI{ndvi} out of expected range for Stem Elongation (0.6-0.8)')
        if ndre < 0.6 or ndre > 0.8:
            diagnosis.append(f'NDRE{ndre} suggests nutrient deficiency or excess (Expected: 0.6-0.8)')
        # if ndwi < 0.6 or ndwi > 0.8:
        #     diagnosis.append('Potential water stress (NDWI out of 0.6-0.8)')
        # if cire is not None and (cire < 0.6 or cire > 0.8):
        #     diagnosis.append(f'CIred-edge{cire} suggests chlorophyll imbalance (Expected: 0.6-0.8)')

    elif growth_stage == 'Flowering':
        if ndvi < 0.7 or ndvi > 0.9:
            diagnosis.append(f'NDVI: {ndvi} out of expected range for Flowering (0.7-0.9)')
        if ndre < 0.6 or ndre > 0.8:
            diagnosis.append(f'NDRE: {ndre} suggests potential nutrient issues (Expected: 0.6-0.8)')
        # if ndwi < 0.6 or ndwi > 0.8:
        #     diagnosis.append('Water stress indicated (NDWI out of 0.6-0.8)')
        if evi < 0.6 or evi > 0.8:
            diagnosis.append(f'EVI: {evi} suggests canopy density issues (Expected: 0.6-0.8)')

    elif growth_stage == 'Maturity':
        if ndvi < 0.3 or ndvi > 0.5:
            diagnosis.append(f'NDVI: {ndvi} out of expected range for Maturity (0.3-0.5)')
        if ndre < 0.3 or ndre > 0.5:
            diagnosis.append(f'NDRE: {ndre} indicates nutrient imbalances during Maturity (Expected: 0.3-0.5)')
        # if ndwi < 0.4 or ndwi > 0.6:
        #     diagnosis.append('Potential water imbalance (NDWI out of 0.4-0.6)')

    # Aggregate diagnosis messages
    if diagnosis:
        return '; '.join(diagnosis)
    return 'No significant issues detected'

if __name__ == '__main__':
    # ndvi_log = pd.read_csv('./kot_ishaq_log.csv')
    # log_file = './ki_syed_log2.csv'
    season = 'kharif'
    crop_id = 'raza'
    log_file = f"{season}_{crop_id}_field_logs.csv"
    log_file_rao = f"../kharif_rao_field_logs.csv"

    ndvi_log = pd.read_csv(log_file)
    wattoo_ids = ndvi_log['name'].drop_duplicates().to_list()
    ndvi_log_rao = pd.read_csv(log_file_rao)
    ndvi_log_rao['name'] = ndvi_log_rao['name'].apply(lambda x: 'r' + str(x))
    # ndvi_log = pd.concat([ndvi_log, ndvi_log_rao])

    # file_dir = '/Users/daoud/PycharmAssets/sufi_farms/'

    # kml_gdf = gpd.read_file(file_dir + '6_Nov_visit_clean.geojson')  # Your KML polygons as GeoDataFrame

    # Get the unique field IDs
    ndvi_log = ndvi_log.sort_values(by='date')

    duplicate_groups = ndvi_log.groupby(['name', 'date']).filter(lambda x: len(x) > 1)
    ndvi_log = ndvi_log[~ndvi_log.index.isin(duplicate_groups.index)]

    veg_idx = 'cire'
    veg_idx_n = ['ndvi', 'evi', 'ndre', 'cire']
    # fig, ax = plt.subplots(3,1)
    for plot_idx, veg_idx_i in enumerate(veg_idx_n):

        ndvi_log_clean = ndvi_log[~ndvi_log['ndvi_mean'].isna()]

        time_series_df = ndvi_log_clean.pivot(index='date', columns='name', values=f'{veg_idx_i}_mean')
        time_series_df = time_series_df.interpolate(method='linear')
        time_series_df = time_series_df.fillna(method='bfill').fillna(method='ffill')

        # 276, 233, below farm not as good
        # 138, 375, high above
        # below 391, 398, 328, 403
        below_ids = [392, 391, 397, 396, 399, 391]
        # top_ids = [ 372, 368, 276, 386, 387, 399, 370, 146, 390, 403] #375, 276, 386, 396, 397]
        # top_ids = [276, 369, 106, 128, 370]#102, 97, 159, 160, 144, 143, 374] #[276, 386, 387, 399, 388, 389, 128, 118] #375, 276, 386, 396, 397]
# 138, 92, 159, 101, 126, 121, 128
        #top_ids = [106, 120, 102, 129, 207, 89, 86]# 129] #['r198',  'r201', 'r207', 'r206', 'r118', '106']#[276, 143, 145, 138, 90, 91]
        top_ids = [ 73, 191, 179, 166]
        only_last_k = 100
        time_series_df = time_series_df[-only_last_k:]
        # time_series_df = time_series_df.drop('2025-03-03')
        time_series_df[top_ids].plot(title= f'plant health levels').legend(
    bbox_to_anchor=(1.0, 1.0),
    fontsize='small',
)


    time_series_df.index = pd.to_datetime(time_series_df.index)
    window_size = 1
    time_series_df = time_series_df.rolling(window=window_size).median()
    time_series_df = time_series_df.dropna()

    time_series_array = time_series_df.T.values  # Transpose for clustering (shape: n_ids x time_steps)

    # Perform clustering (as done earlier)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(time_series_array)

    # Attach cluster labels to the original IDs
    cluster_results = pd.DataFrame({'id': time_series_df.columns, 'cluster': clusters})

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    plt.figure(figsize=(12, 6))
    plt.ylim(0, 6)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    series_idx_n = []#[138, 375]

    # 4, 5 look togther
    crop_clusters  = [1]# 2, 3, 4, 5, 6, 7, 8 ,9]# 3, 6, 7, 8, 4, 5, 9]
    for cluster in crop_clusters:# 4, 5, 6, 7, 8 9]: #1, 3, 5, 7, 8, 9]:#,2,5, 8,7, 9]:#, 3, 6, 7, 8]: # 5, 9 are time offset


        # plt.figure(figsize=(12, 6))
        # plt.ylim(0, 5)
        cluster_ids = cluster_results[cluster_results['cluster'] == cluster]['id']

        cluster_to_sowing = {
            0: pd.Timestamp(time_series_df.index[0]),
            1: pd.Timestamp(time_series_df.index[0]), # 2 for k2
            2: pd.Timestamp(time_series_df.index[0]),
            3: pd.Timestamp(time_series_df.index[0]),# 2 for k2
            4: pd.Timestamp(time_series_df.index[0]),
            5: pd.Timestamp(time_series_df.index[0]),
            6: pd.Timestamp(time_series_df.index[0]),
            7: pd.Timestamp(time_series_df.index[0]),# 4 for k2
            8:pd.Timestamp(time_series_df.index[0]),
            9: pd.Timestamp(time_series_df.index[0]), #diff crop?

        }
        cluster_id = 7
        days_since_sowing = (time_series_df.index - cluster_to_sowing[cluster_id]).days



        # plt.title(f'cluster: {cluster}')


        for idx in cluster_ids:
            cluster_to_offset= {
                0: 0,
                1: 0, # same as 8, 0
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,

            }

            offset = cluster_to_offset[cluster]
            series_idx= pd.Series(time_series_df[idx], time_series_df.index)
            series_idx.index = pd.to_datetime(series_idx.index)
            series_idx.index = series_idx.index - cluster_to_sowing[cluster]
            series_idx_resampled = series_idx.resample('D').asfreq()
            series_idx_interpolate = series_idx_resampled.interpolate(method='linear')
            # todo: potentially a need to interpolate
            series_idx_n.append(series_idx_interpolate)
            plt.plot(
                time_series_df.index - cluster_to_sowing[cluster],
                time_series_df[idx],
                label=f"Cluster {cluster}" if idx == cluster_ids.iloc[0] else "",  # Avoid duplicate labels
                color=colors[cluster],
                alpha=0.2,
            )
    plt.show()

    normalized_ts = pd.concat(series_idx_n, axis = 1)
    normalized_ts_filter = normalized_ts[normalized_ts.index>pd.Timedelta("0 days")] # 30 days
    normalized_ts_filter = normalized_ts_filter[normalized_ts_filter.index<pd.Timedelta("80 days")]

    # normalized_ts_filter.plot()
    # normalized_ts_filter_early = normalized_ts_filter.iloc[3:8]  # todo: should be based on date

    mean = normalized_ts_filter.mean(axis=1)
    std = normalized_ts_filter.std(axis=1)
    #
    # plt.errorbar(subcluster_time_series_early.index,
    #          mean, std, color = 'red')
    #
    z_scores_df = (normalized_ts_filter.sub(mean, axis=0)).div(std, axis=0)
    z_scores_df[top_ids].plot()
    # z_scores_df = z_scores_df[wattoo_ids]
    #filter
    z_scores_df.to_csv(f'{crop_id}_cire_z_scores_ts.csv')
    z_score_mean_early = z_scores_df.mean(axis=0)
    # add merger

    z_score_mean_early.to_csv(f'{crop_id}_cire_z_scores_norm.csv')
    sowing_date_ts_df = time_series_df.copy()



    # calculate z score globally
    subcluster_id_n =  [0, 2, 3, 6, 7, 8] # not 2, 4, 7, 9
    cluster_ids = cluster_results[cluster_results['cluster'].isin(subcluster_id_n)]['id'].to_list()
    subcluster_time_series = time_series_df[cluster_ids]
    subcluster_time_series_array = subcluster_time_series.T.values  # Transpose for clustering (shape: n_ids x time_steps)

    # Perform clustering (as done earlier)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    subclusters = kmeans.fit_predict(subcluster_time_series_array)
    subcluster_results = pd.DataFrame({'id': subcluster_time_series.columns, 'cluster': subclusters})

    # Define a colormap

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(12, 6))
    plt.ylim(0, 5)


    # for each cluster find the highest ndvi date, date of fall or smth and adjust stsrt date accordingly



    for cluster in range(n_clusters):




        cluster_ids = subcluster_results[subcluster_results['cluster'] == cluster]['id']
        subcluster_time_series_early = subcluster_time_series.iloc[3:8] # todo: should be based on date
        mean = subcluster_time_series_early.mean(axis = 1)
        std = subcluster_time_series_early.std(axis = 1)
        #
        # plt.errorbar(subcluster_time_series_early.index,
        #          mean, std, color = 'red')
        #

        # old_df
        # z_scores_df = (subcluster_time_series_early.sub(mean, axis=0)).div(std, axis=0)
        # z_scores_df.to_csv('cire_z_score_timeseries.csv')
        # z_score_mean_early = z_scores_df.mean(axis = 0)


        norm = Normalize(vmin=z_score_mean_early.min(), vmax=z_score_mean_early.max())
        cmap = cm.viridis  # Colormap for positive and negative values

        # Step 3: Plot each field's original NDVI time series colored by Z-score
        plt.figure(figsize=(10, 6))
        subcluster_time_series_mid = subcluster_time_series.iloc[:]
        for field in subcluster_time_series_mid.columns:
            if field in z_score_mean_early.index:
                color = cmap(norm(z_score_mean_early[field]))  # Map Z-score to color
                plt.plot(subcluster_time_series_mid.index, subcluster_time_series_mid[field],  color=color,
                         alpha = 0.2, linewidth=1.5)
        plt.show()


        for field in z_scores_df.columns:
            plt.plot(z_scores_df.index, z_scores_df[field], color='blue', alpha=0.2, linewidth=1.5)        # for field in z_scores_df.columns:
        #     plt.plot(z_scores_df.index, z_scores_df[field], color = 'blue', alpha = 0.2, linewidth=1.5)

        for idx in cluster_ids:
            plt.plot(
                z_scores_df.index,
                z_scores_df[idx],
                label=f"Cluster {cluster}" if idx == cluster_ids.iloc[0] else "",  # Avoid duplicate labels
                color=colors[cluster],
                alpha=0.2,
            )
        # z_score_mean_early.to_csv(f'cire_z_scores_cluster_{subcluster_id}.csv')

        z_score_mean_early.to_csv(f'cire_z_scores_cluster_all.csv')

        assert False

        for idx in cluster_ids:
            plt.plot(
                subcluster_time_series_early.index,
                subcluster_time_series_early[idx],
                label=f"Cluster {cluster}" if idx == cluster_ids.iloc[0] else "",  # Avoid duplicate labels
                color=colors[cluster],
                alpha=0.2,
            )
    # cluster_results.to_csv(f'./cluster_{veg_idx}.csv')
    plt.show()
    assert False

    # now find clusters within each cluster

    plot_variety = 'gold_1_hybrid'
    plot_ndvi_time_series(ndvi_log_clean, veg_indice = 'ndvi', variety=plot_variety)
    plot_ndvi_time_series(ndvi_log, veg_indice = 'ndre', variety=plot_variety)
    plot_ndvi_time_series(ndvi_log, veg_indice = 'evi', variety=plot_variety)

    plot_ndvi_time_series(ndvi_log, veg_indice = 'cire', variety=plot_variety)

    field_ids = ndvi_log['name'].unique()
    for field_id in field_ids:
        if field_id == 'Feild 20':
            field_data = ndvi_log[ndvi_log['name'] == field_id]
            field_data['growth_stage'] = field_data['date'].apply(determine_crop_growth_stage)
            field_data['ndvi_alert'] = field_data.apply(ndvi_alert, axis=1)
            field_data['overall alert'] = field_data.apply(diagnose_field_issues, axis=1)
            # field_data['ndre_alert'] = None
            for idx, row in field_data[['date', 'overall alert']].iterrows():
                print (row.values)


