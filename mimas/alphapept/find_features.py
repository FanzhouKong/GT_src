from . import performance
from . import feature_finding
from .constants import averagine_aa, isotopes
import numpy as np
import pandas as pd


@performance.performance_function(compilation_mode="numba")
def extract_xic(
    idx: np.ndarray,
    result_xic_list: np.ndarray,
    rt_idx_list: np.ndarray,
    mz_start_list: np.ndarray,
    mz_end_list: np.ndarray,
    ms1_indices: np.ndarray,
    ms1_mass_list: np.ndarray,
    ms1_int_list: np.ndarray,
):
    rt_idx = rt_idx_list[idx]
    mz_start = mz_start_list[idx]
    mz_end = mz_end_list[idx]
    ms1_index_start = ms1_indices[rt_idx]
    ms1_index_end = ms1_indices[rt_idx + 1]

    ms1_mass = ms1_mass_list[ms1_index_start:ms1_index_end]
    ms1_int = ms1_int_list[ms1_index_start:ms1_index_end]

    int_sum = np.sum(ms1_int[(ms1_mass >= mz_start) & (ms1_mass <= mz_end)])
    result_xic_list[idx] = int_sum


def find_features(ms_file, **kwargs):
    f_settings = {
        "ms1_tol_for_xic": 0.005,  # This is the m/z tolerance for extracting XIC.
        "centroid_tol": 8,  # This is the start cutoff when connect two RT neighbor peak.
        # It equals the delta ppm between the neighbor RT peak.
        # 8 means only peaks with delta ppm less than 8 ppm will be connected.
        "hill_length_min": 5,  # This is the minimum length of a hill.
        # 3 means at least 5 datapoint is needed for a feature.
        "max_gap": 2,  # This is the maximum gap between two datapoints in a feature.
        "hill_smoothing": 1,  # This is the window size for smoothing the hill.
        # The value in i will be used as median([i-window,i+window+1])
        "hill_split_level": 1.3,  # This is the split level for spliting the hill.
        # This is an example:   min_1,   max_1,   min_2,   max_2,   min_3
        #                       |        |        |        |        |
        # It's calculated as minimal(max_1, max_2) / min_2,
        # If this value is higer than 1.3 (i.e., min/max <0.77), it will be considered as two features.
        "hill_check_large": 2,  # This is the window size for checking if a hill is too flat.
        "hill_peak_factor": 1.3,  # This is the factor for checking if a hill is too flat.
        # It will re-smooth with 'hill_smoothing', then if max/board_value < hill_peak_factor, it will be removed.
        "hill_nboot": 150,  # This is the number of bootstrapping for hill stats, which is used for calculating the intra delta_mz
        "hill_nboot_max": 300,  # This is the maximum number of bootstrapping for hill stats, which is used for calculating the intra delta_mz
        "iso_charge_max": 3,  # This is the maximum charge state for isotope pattern.
        "iso_charge_min": 1,  # This is the minimum charge state for isotope pattern.
        "iso_corr_min": 0.6,  # This is the minimum correlation for isotope pattern.
        "iso_mass_range": 5,  # This number times delta_mz will be used as the tolerance for isotope determination.
        "iso_n_seeds": 100,  # This value looks like very big one, unlikely will be reached.
        "iso_split_level": 1.3,  # The value used to split isotope pattern into two isotope group.
        "map_mz_range": 0.5,  # This is the m/z range for mapping MS2 to MS1, this parameter will be updated if the isolation window is provided in mzML.
        "map_rt_range": 1,  # This is the RT range for mapping MS2 to MS1 (in min).
        "map_mob_range": 0.3,  # This is the mobility range for mapping MS2 to MS1.
        "map_n_neighbors": 10,  # This is the number of neighbors for mapping MS2 to MS1.
    }

    f_settings.update(kwargs)
    performance.set_worker_count(1)
    max_gap = f_settings["max_gap"]
    centroid_tol = f_settings["centroid_tol"]
    hill_split_level = f_settings["hill_split_level"]
    iso_split_level = f_settings["iso_split_level"]
    window = f_settings["hill_smoothing"]
    hill_check_large = f_settings["hill_check_large"]
    hill_peak_factor = f_settings["hill_peak_factor"]
    iso_charge_min = f_settings["iso_charge_min"]
    iso_charge_max = f_settings["iso_charge_max"]
    iso_n_seeds = f_settings["iso_n_seeds"]
    hill_nboot_max = f_settings["hill_nboot_max"]
    hill_nboot = f_settings["hill_nboot"]
    iso_mass_range = f_settings["iso_mass_range"]
    iso_corr_min = f_settings["iso_corr_min"]

    ms_file["int_list_ms1"] = np.array(ms_file["int_list_ms1"]).astype(int)
    int_data = ms_file["int_list_ms1"]
    # logging.info('Feature finding on {}'.format(file_name))
    # logging.info(f'Hill extraction with centroid_tol {centroid_tol} and max_gap {max_gap}')

    hill_ptrs, hill_data, path_node_cnt, score_median, score_std = feature_finding.extract_hills(ms_file, max_gap, centroid_tol)

    # logging.info(f'Number of hills {len(hill_ptrs):,}, len = {np.mean(path_node_cnt):.2f}')
    # logging.info(f'Repeating hill extraction with centroid_tol {score_median+score_std*3:.2f}')

    hill_ptrs, hill_data, path_node_cnt, score_median, score_std = feature_finding.extract_hills(ms_file, max_gap, score_median + score_std * 3)

    # logging.info(f'Number of hills {len(hill_ptrs):,}, len = {np.mean(path_node_cnt):.2f}')

    hill_ptrs, hill_data = feature_finding.remove_duplicate_hills(hill_ptrs, hill_data, path_node_cnt)
    # logging.info(f'After duplicate removal of hills {len(hill_ptrs):,}')

    hill_ptrs = feature_finding.split_hills(hill_ptrs, hill_data, int_data, hill_split_level=hill_split_level, window=window)  # hill lenght is inthere already
    # logging.info(f'After split hill_ptrs {len(hill_ptrs):,}')

    hill_data, hill_ptrs = feature_finding.filter_hills(
        hill_data, hill_ptrs, int_data, hill_check_large=hill_check_large, window=window, hill_peak_factor=hill_peak_factor
    )

    # logging.info(f'After filter hill_ptrs {len(hill_ptrs):,}')

    stats, sortindex_, idxs_upper, scan_idx, hill_data, hill_ptrs = feature_finding.get_hill_data(
        ms_file, hill_ptrs, hill_data, hill_nboot_max=hill_nboot_max, hill_nboot=hill_nboot
    )
    # logging.info('Extracting hill stats complete')

    # Save stats
    # pd.DataFrame(stats).to_csv( 'stats.csv', index=False)

    pre_isotope_patterns = feature_finding.get_pre_isotope_patterns(
        stats,
        idxs_upper,
        sortindex_,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        feature_finding.maximum_offset,
        iso_charge_min=iso_charge_min,
        iso_charge_max=iso_charge_max,
        iso_mass_range=iso_mass_range,
        cc_cutoff=iso_corr_min,
    )
    # logging.info('Found {:,} pre isotope patterns.'.format(len(pre_isotope_patterns)))

    isotope_patterns, iso_idx, isotope_charges = feature_finding.get_isotope_patterns(
        pre_isotope_patterns,
        hill_ptrs,
        hill_data,
        int_data,
        scan_idx,
        stats,
        sortindex_,
        averagine_aa,
        isotopes,
        iso_charge_min=iso_charge_min,
        iso_charge_max=iso_charge_max,
        iso_mass_range=iso_mass_range,
        iso_n_seeds=iso_n_seeds,
        cc_cutoff=iso_corr_min,
        iso_split_level=iso_split_level,
        callback=None,
    )
    # logging.info('Extracted {:,} isotope patterns.'.format(len(isotope_charges)))

    feature_table, lookup_idx = feature_finding.feature_finder_report(
        ms_file, isotope_patterns, isotope_charges, iso_idx, stats, sortindex_, hill_ptrs, hill_data
    )
    # logging.info('Report complete.')

    # Add single isotope patterns to the feature table
    isotope_patterns_single = np.setdiff1d(np.arange(stats.shape[0]), isotope_patterns)
    isotope_charges_single = np.ones(len(isotope_patterns_single), dtype=isotope_charges.dtype)
    iso_idx_single = np.arange(len(isotope_patterns_single) + 1, dtype=iso_idx.dtype)
    feature_table_single, lookup_idx_single = feature_finding.feature_finder_report(
        ms_file, isotope_patterns_single, isotope_charges_single, iso_idx_single, stats, sortindex_, hill_ptrs, hill_data
    )

    # Add S/N to the feature table
    feature_table = calculate_sn(feature_table, ms_file, f_settings["ms1_tol_for_xic"])

    feature_table = pd.concat([feature_table, feature_table_single], axis=0, ignore_index=True)
    feature_table = feature_table.sort_values(["rt_start", "mz"])

    # Calculate additional params
    feature_table["rt_length"] = feature_table["rt_end"] - feature_table["rt_start"]
    feature_table["rt_right"] = feature_table["rt_end"] - feature_table["rt_apex"]
    feature_table["rt_left"] = feature_table["rt_apex"] - feature_table["rt_start"]
    feature_table["rt_tail"] = feature_table["rt_right"] / feature_table["rt_left"]
    feature_table["rt"] = feature_table["rt_apex"]

    # Reset the index
    feature_table.reset_index(drop=True, inplace=True)

    # logging.info('Matching features to query data.')

    features_ms2 = pd.DataFrame()
    if "mono_mzs2" in ms_file.keys():
        if "select_windows_ms2" in ms_file.keys():
            # Determine the map_mz_range for MS2
            select_windows_ms2 = ms_file["select_windows_ms2"]
            map_mz_range = np.median(select_windows_ms2[:, 1] - select_windows_ms2[:, 0]) / 2
            if map_mz_range > 0:
                f_settings["map_mz_range"] = map_mz_range
        features_ms2 = feature_finding.map_ms2(feature_table, ms_file, **f_settings)

        # For every feature in MS1, only keep the nearest RT MS2 feature
        features_ms2["rt_offset_abs"] = np.abs(features_ms2["rt_offset"])
        features_ms2_simple = features_ms2.sort_values(["rt_offset_abs"])

        # Keep only the nearest MS2 feature
        features_ms2_simple = features_ms2_simple.groupby("feature_idx").first().reset_index()

        # If one MS2 feature is assigned to multiple MS1 features, keep only the one with the highest intensity
        features_ms2_simple.sort_values(["ms1_int_sum_apex"], ascending=False, inplace=True)
        features_ms2_simple = features_ms2_simple.groupby("query_idx").first().reset_index()

        features_ms2_simple = features_ms2_simple[["feature_idx", "query_idx"]]
        features_ms2_simple.columns = ["feature_idx", "ms2_idx"]

        # Merge the MS2 features back into the MS1 feature table
        feature_table = feature_table.merge(features_ms2_simple, left_index=True, right_on="feature_idx", how="left")
    feature_table = link_msms(feature_table, ms_file)
    return feature_table
def link_msms(feature_table, ms_file):
    msms = []
    for index, row in feature_table.iterrows():

        if row['ms2_idx']==row['ms2_idx']:
            idx_start = ms_file['indices_ms2'][int(row['ms2_idx'])]
            idx_end = ms_file['indices_ms2'][int(row['ms2_idx'])+1]
            mass =ms_file['mass_list_ms2'][idx_start:idx_end]
            intensity = ms_file['int_list_ms2'][idx_start:idx_end]
            msms.append([[x,y] for x, y in zip(mass,intensity)])
            # break
        else:
            msms.append(np.NAN)
    feature_table['msms']=msms
    return feature_table
def calculate_sn(ms1_features, ms_file, ms1_tolerance_in_da):
    # Get RT, and MS1 intensity from HDF5 file
    ms1_rt_list = ms_file["rt_list_ms1"]
    ms1_mass_list = ms_file["mass_list_ms1"]
    ms1_int_list = ms_file["int_list_ms1"]
    ms1_indices = ms_file["indices_ms1"]

    mz_list = ms1_features["mz"].values
    mz_list_start = mz_list - ms1_tolerance_in_da
    mz_list_end = mz_list + ms1_tolerance_in_da

    ###########################################################
    # Extract XIC for feature start
    rt_index = ms1_features["rt_start_idx"].values.astype(np.int32)
    result_xic_1 = np.zeros(len(rt_index), dtype=np.float32)
    extract_xic(range(len(result_xic_1)), result_xic_1, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    result_xic_2 = np.zeros(len(rt_index), dtype=np.float32)
    rt_index -= 1
    rt_index[rt_index < 0] = 0
    extract_xic(range(len(result_xic_2)), result_xic_2, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    feature_start_xic = np.minimum(result_xic_1, result_xic_2)

    ###########################################################
    # Extract XIC for feature end
    rt_index = ms1_features["rt_end_idx"].values.astype(np.int32)
    result_xic_1 = np.zeros(len(rt_index), dtype=np.float32)
    extract_xic(range(len(result_xic_1)), result_xic_1, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    result_xic_2 = np.zeros(len(rt_index), dtype=np.float32)
    rt_index += 1
    rt_index[rt_index >= len(ms1_rt_list)] = len(ms1_rt_list) - 1
    extract_xic(range(len(result_xic_2)), result_xic_2, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    feature_end_xic = np.minimum(result_xic_1, result_xic_2)

    ###########################################################
    # Extract XIC for feature apex
    rt_index = ms1_features["rt_apex_idx"].values.astype(np.int32)
    result_xic_1 = np.zeros(len(rt_index), dtype=np.float32)
    extract_xic(range(len(result_xic_1)), result_xic_1, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    result_xic_2 = np.zeros(len(rt_index), dtype=np.float32)
    rt_index += 1
    rt_index[rt_index >= len(ms1_rt_list)] = len(ms1_rt_list) - 1
    extract_xic(range(len(result_xic_2)), result_xic_2, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    result_xic_3 = np.zeros(len(rt_index), dtype=np.float32)
    rt_index -= 2
    rt_index[rt_index < 0] = 0
    extract_xic(range(len(result_xic_3)), result_xic_3, rt_index, mz_list_start, mz_list_end, ms1_indices, ms1_mass_list, ms1_int_list)

    feature_apex_xic = np.maximum(result_xic_1, result_xic_2, result_xic_3)

    ###########################################################
    # Calculate S/N
    feature_noise = feature_start_xic + feature_end_xic
    feature_sn = feature_apex_xic / feature_noise * 2
    feature_sn[feature_noise == 0] = 1000

    ###########################################################
    # Save S/N to HDF5 file
    ms1_features["sn_ratio"] = feature_sn
    return ms1_features
