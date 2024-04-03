# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_io.ipynb.

# %% auto 0
__all__ = ['get_peaks', 'get_centroid', 'gaussian_estimator', 'centroid_data', 'get_local_intensity', 'get_most_abundant',
           'load_thermo_raw', 'load_bruker_raw', 'one_over_k0_to_CCS', 'import_sciex_as_alphapept', 'load_sciex_raw',
           'check_sanity', 'extract_mzml_info', 'load_mzml_data', 'extract_mq_settings', 'parse_mq_seq',
           'list_to_numpy_f32', 'HDF_File', 'MS_Data_File', 'index_ragged_list', 'raw_conversion']

# %% ../nbs/02_io.ipynb 3
from numba import njit
import numpy as np

@njit
def get_peaks(int_array: np.ndarray) -> list:
    """Detects peaks in an array.

    Args:
        int_array (np.ndarray): An array with intensity values.

    Returns:
        list: A regular Python list with all peaks.
            A peak is a triplet of the form (start, center, end)

    """
    peaklist = []
    gradient = np.diff(int_array)
    start, center, end = -1, -1, -1

    for i in range(len(gradient)):

        grad = gradient[i]

        if (end == -1) & (center == -1):  # No end and no center yet
            if grad <= 0:  # If decreasing, move start point
                start = i
            else:  # If increasing set as center point
                center = i

        if (end == -1) & (
            center != -1
        ):  # If we have a centerpoint and it is still increasing set as new center
            if grad >= 0:
                center = i
            else:  # If it is decreasing set as endpoint
                end = i

        if end != -1:  # If we have and endpoint and it is going down
            if grad < 0:
                end = i  # Set new Endpoint
            else:  # if it stays the same or goes up set a new peak
                peaklist.append((start + 1, center + 1, end + 1))
                start, center, end = end, -1, -1  # Reset start, center, end

    if end != -1:
        peaklist.append((start + 1, center + 1, end + 1))

    return peaklist

# %% ../nbs/02_io.ipynb 8
from numba import njit

@njit
def get_centroid(
    peak: tuple,
    mz_array: np.ndarray,
    int_array: np.ndarray
) -> tuple:
    """Wrapper to estimate centroid center positions.

    Args:
        peak (tuple): A triplet of the form (start, center, end)
        mz_array (np.ndarray): An array with mz values.
        int_array (np.ndarray): An array with intensity values.

    Returns:
        tuple: A tuple of the form (center, intensity)
    """
    start, center, end = peak
    mz_int = np.sum(int_array[start + 1 : end])

    peak_size = end - start - 1

    if peak_size == 1:
        mz_cent = mz_array[center]
    elif peak_size == 2:
        mz_cent = (
            mz_array[start + 1] * int_array[start + 1]
            + mz_array[end - 1] * int_array[end - 1]
        ) / (int_array[start + 1] + int_array[end - 1])
    else:
        mz_cent = gaussian_estimator(peak, mz_array, int_array)

    return mz_cent, mz_int

@njit
def gaussian_estimator(
    peak: tuple,
    mz_array: np.ndarray,
    int_array: np.ndarray
) -> float:
    """Three-point gaussian estimator.

    Args:
        peak (tuple): A triplet of the form (start, center, end)
        mz_array (np.ndarray): An array with mz values.
        int_array (np.ndarray): An array with intensity values.

    Returns:
        float: The gaussian estimate of the center.
    """
    start, center, end = peak

    m1, m2, m3 = mz_array[center - 1], mz_array[center], mz_array[center + 1]
    i1, i2, i3 = int_array[center - 1], int_array[center], int_array[center + 1]

    if i1 == 0:  # Case of sharp flanks
        m = (m2 * i2 + m3 * i3) / (i2 + i3)
    elif i3 == 0:
        m = (m1 * i1 + m2 * i2) / (i1 + i2)
    else:
        l1, l2, l3 = np.log(i1), np.log(i2), np.log(i3)
        m = (
            ((l2 - l3) * (m1 ** 2) + (l3 - l1) * (m2 ** 2) + (l1 - l2) * (m3 ** 2))
            / ((l2 - l3) * (m1) + (l3 - l1) * (m2) + (l1 - l2) * (m3))
            * 1
            / 2
        )

    return m

# %% ../nbs/02_io.ipynb 14
@njit
def centroid_data(
    mz_array: np.ndarray,
    int_array: np.ndarray
) -> tuple:
    """Estimate centroids and intensities from profile data.

    Args:
        mz_array (np.ndarray): An array with mz values.
        int_array (np.ndarray): An array with intensity values.

    Returns:
        tuple: A tuple of the form (mz_array_centroided, int_array_centroided)
    """
    peaks = get_peaks(int_array)

    mz_array_centroided = np.zeros(len(peaks))
    int_array_centroided = np.zeros(len(peaks))


    for i in range(len(peaks)):
        mz_, int_ = get_centroid(peaks[i], mz_array, int_array)
        mz_array_centroided[i] = mz_
        int_array_centroided[i] = int_

    return mz_array_centroided, int_array_centroided

# %% ../nbs/02_io.ipynb 16
from .chem import calculate_mass
from tqdm import tqdm
import numpy as np
from numba.typed import List
from numba import njit
import gzip
import sys
import os
import logging

@njit
def get_local_intensity(intensity, window=10):
    """
    Calculate the local intensity for a spectrum.

    Args:
        intensity (np.ndarray): An array with intensity values.
        window (int): Window Size
    Returns:
        nop.ndarray: local intensity
    """

    local_intensity = np.zeros(len(intensity))

    for i in range(len(intensity)):
        start = max(0, i-window) 
        end = min(len(intensity), i+window)
        local_intensity[i] = intensity[i]/np.max(intensity[start:end])
        
    return local_intensity

def get_most_abundant(
    mass: np.ndarray,
    intensity: np.ndarray,
    n_max: int,
    window: int = 10,
) -> tuple:
    """Returns the n_max most abundant peaks of a spectrum.

    Args:
        mass (np.ndarray): An array with mz values.
        intensity (np.ndarray): An array with intensity values.
        n_max (int): The maximum number of peaks to retain.
            Setting `n_max` to -1 returns all peaks.
        window (int): Use local maximum in a window

    Returns:
        tuple: the filtered mass and intensity arrays.

    """
    if n_max == -1:
        return mass, intensity
    if len(mass) < n_max:
        return mass, intensity
    else:
        
        if window > 0:
            sortindex = np.argsort(get_local_intensity(intensity, window))[::-1][:n_max]
        else:
            sortindex = np.argsort(intensity)[::-1][:n_max]
        
        sortindex.sort()

    return mass[sortindex], intensity[sortindex]

# %% ../nbs/02_io.ipynb 19
def load_thermo_raw(
    raw_file_name: str,
    n_most_abundant: int,
    use_profile_ms1: bool = False,
    callback: callable = None,
) -> tuple:
    """Load raw thermo data as a dictionary.

    Args:
        raw_file_name (str): The name of a Thermo .raw file.
        n_most_abundant (int): The maximum number of peaks to retain per MS2 spectrum.
        use_profile_ms1 (bool): Use profile data or centroid it beforehand. Defaults to False.
        callback (callable): A function that accepts a float between 0 and 1 as progress. Defaults to None.

    Returns:
        tuple: A dictionary with all the raw data and a string with the acquisition_date_time

    """
    from alphapept.pyrawfilereader import RawFileReader
    rawfile = RawFileReader(raw_file_name)

    spec_indices = range(rawfile.FirstSpectrumNumber, rawfile.LastSpectrumNumber + 1)

    scan_list = []
    rt_list = []
    mass_list = []
    int_list = []
    ms_list = []
    prec_mzs_list = []
    mono_mzs_list = []
    charge_list = []

    for idx, i in enumerate(spec_indices):
        try:
            ms_order = rawfile.GetMSOrderForScanNum(i)
            rt = rawfile.RTFromScanNum(i)

            if ms_order == 2:
                prec_mz = rawfile.GetPrecursorMassForScanNum(i, 0)

                mono_mz, charge = rawfile.GetMS2MonoMzAndChargeFromScanNum(i)
            else:
                prec_mz, mono_mz, charge = 0,0,0

            if use_profile_ms1:
                if ms_order == 2:
                    masses, intensity = rawfile.GetCentroidMassListFromScanNum(i)
                    masses, intensity = get_most_abundant(masses, intensity, n_most_abundant)
                else:
                    masses, intensity = rawfile.GetProfileMassListFromScanNum(i)
                    masses, intensity = centroid_data(masses, intensity)

            else:
                masses, intensity = rawfile.GetCentroidMassListFromScanNum(i)
                if ms_order == 2:
                    masses, intensity = get_most_abundant(masses, intensity, n_most_abundant)

            scan_list.append(i)
            rt_list.append(rt)
            mass_list.append(np.array(masses))
            int_list.append(np.array(intensity, dtype=np.int64))
            ms_list.append(ms_order)
            prec_mzs_list.append(prec_mz)
            mono_mzs_list.append(mono_mz)
            charge_list.append(charge)
        except KeyboardInterrupt as e:
            raise e
        except SystemExit as e:
            raise e
        except Exception as e:
            logging.info(f"Bad scan={i} in raw file '{raw_file_name}': {e}")

        if callback:
            callback((idx+1)/len(spec_indices))

    scan_list_ms1 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    rt_list_ms1 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    mass_list_ms1 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    int_list_ms1 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    ms_list_ms1 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 1]

    scan_list_ms2 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    rt_list_ms2 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mass_list_ms2 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    int_list_ms2 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    ms_list_ms2 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mono_mzs2 = [mono_mzs_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    charge2 = [charge_list[i] for i, _ in enumerate(ms_list) if _ == 2]

    prec_mass_list2 = [
        calculate_mass(mono_mzs_list[i], charge_list[i])
        for i, _ in enumerate(ms_list)
        if _ == 2
    ]

    check_sanity(mass_list)

    query_data = {}

    query_data["scan_list_ms1"] = np.array(scan_list_ms1)
    query_data["rt_list_ms1"] = np.array(rt_list_ms1)
    query_data["mass_list_ms1"] = np.array(mass_list_ms1, dtype=object)
    query_data["int_list_ms1"] = np.array(int_list_ms1, dtype=object)
    query_data["ms_list_ms1"] = np.array(ms_list_ms1)

    query_data["scan_list_ms2"] = np.array(scan_list_ms2)
    query_data["rt_list_ms2"] = np.array(rt_list_ms2)
    query_data["mass_list_ms2"] = mass_list_ms2
    query_data["int_list_ms2"] = int_list_ms2
    query_data["ms_list_ms2"] = np.array(ms_list_ms2)
    query_data["prec_mass_list2"] = np.array(prec_mass_list2)
    query_data["mono_mzs2"] = np.array(mono_mzs2)
#     TODO: Refactor charge2 to be consistent: charge_ms2
    query_data["charge2"] = np.array(charge2)

    acquisition_date_time = rawfile.GetCreationDate()

    rawfile.Close()

    return query_data, acquisition_date_time

# %% ../nbs/02_io.ipynb 21
def load_bruker_raw(raw_file, most_abundant, callback=None, **kwargs):
    """
    Load bruker raw file and extract spectra
    """
    import alphatims.bruker
    from .constants import mass_dict
    from .io import list_to_numpy_f32, get_most_abundant

    data = alphatims.bruker.TimsTOF(raw_file)
    prec_data = data.precursors
    frame_data = data.frames
    frame_data = frame_data.set_index('Id')
    
    import sqlalchemy as db
    import pandas as pd

    tdf = os.path.join(raw_file, 'analysis.tdf')
    engine = db.create_engine('sqlite:///{}'.format(tdf))

    global_metadata = pd.read_sql_table('GlobalMetadata', engine)
    global_metadata = global_metadata.set_index('Key').to_dict()['Value']
    acquisition_date_time = global_metadata['AcquisitionDateTime']
    

    M_PROTON = mass_dict['Proton']

    prec_data['Mass'] = prec_data['MonoisotopicMz'].values * prec_data['Charge'].values - prec_data['Charge'].values*M_PROTON

    query_data = {}

    query_data['prec_mass_list2'] = prec_data['Mass'].values
    query_data['prec_id2'] = prec_data['Id'].values
    query_data['mono_mzs2'] = prec_data['MonoisotopicMz'].values
    query_data['rt_list_ms2'] = frame_data.loc[prec_data['Parent'].values]['Time'].values / 60 #convert to minutes
    query_data['scan_list_ms2'] = prec_data['Parent'].values
    query_data['charge2'] = prec_data['Charge'].values

    query_data['mobility2'] = data.mobility_values[
        data.precursors.ScanNumber.values.astype(np.int64)
    ]
    (
        spectrum_indptr,
        spectrum_tof_indices,
        spectrum_intensity_values,
    ) = data.index_precursors(
        centroiding_window=5,
        keep_n_most_abundant_peaks=most_abundant
    )
    # TODO: Centroid spectra and trim
    query_data["alphatims_spectrum_indptr_ms2"] = spectrum_indptr[1:]
    query_data["alphatims_spectrum_mz_values_ms2"] = data.mz_values[spectrum_tof_indices]
    query_data["alphatims_spectrum_intensity_values_ms2"] = spectrum_intensity_values

    return query_data, acquisition_date_time

# %% ../nbs/02_io.ipynb 23
import numpy as np

def one_over_k0_to_CCS(
    one_over_k0s: np.ndarray,
    charges: np.ndarray,
    mzs: np.ndarray,
) -> np.ndarray:
    """Retrieve collisional cross section (CCS) values from (mobility, charge, mz) arrays.

    Args:
        one_over_k0s (np.ndarray): The ion mobilities (1D-np.float).
        charges (np.ndarray): The charges (1D-np.int).
        mzs (np.ndarray): The mz values (1D-np.float).

    Returns:
        np.ndarray: The CCS values.

    """
    from alphapept.ext.bruker import timsdata

    ccs = np.empty(len(one_over_k0s))
    ccs[:] = np.nan

    for idx, (one_over, charge, mz) in enumerate(zip(one_over_k0s, charges, mzs)):
        try:
            ccs[idx] = timsdata.oneOverK0ToCCSforMz(one_over, int(charge), mz)
        except ValueError:
            pass
    return ccs



# %% ../nbs/02_io.ipynb 29
def check_sanity(mass_list: np.ndarray) -> None:
    """Sanity check for mass list to make sure the masses are sorted.

    Args:
        mass_list (np.ndarray): The mz values (1D-np.float).

    Raises:
        ValueError: When the mz values are not sorted.

    """
    if not all(
        mass_list[0][i] <= mass_list[0][i + 1] for i in range(len(mass_list[0]) - 1)
    ):
        raise ValueError("Masses are not sorted.")


def extract_mzml_info(input_dict: dict) -> tuple:
    """Extract basic MS coordinate arrays from a dictionary.

    Args:
        input_dict (dict): A dictionary obtained by iterating over a Pyteomics mzml.read function.

    Returns:
        tuple: The rt, masses, intensities, ms_order, prec_mass, mono_mz, charge arrays retrieved from the input_dict.
            If the `ms level` in the input dict does not equal 2, the charge, mono_mz and prec_mass will be equal to 0.

    """
    from alphapept.chem import calculate_mass
    rt = float(input_dict.get('scanList').get('scan')[0].get('scan start time'))  # rt_list_ms1/2
    masses = input_dict.get('m/z array')
    intensities = input_dict.get('intensity array').astype(int)
    ms_order = input_dict.get('ms level')
    prec_mass = mono_mz = charge = 0
    if ms_order == 2:
        try:
            charge = int(input_dict.get('precursorList').get('precursor')[0].get('selectedIonList').get('selectedIon')[0].get(
                'charge state'))
        except TypeError:
            charge = 0
        mono_mz = input_dict.get('precursorList').get('precursor')[0].get('selectedIonList').get('selectedIon')[0].get(
                'selected ion m/z')
        prec_mass = calculate_mass(mono_mz, charge)
    return rt, masses, intensities, ms_order, prec_mass, mono_mz, charge


def load_mzml_data(
    filename: str,
    n_most_abundant: int,
    callback: callable = None,
    **kwargs
) -> tuple:
    """Load data from an mzml file as a dictionary.

    Args:
        filename (str): The name of a .mzml file.
        n_most_abundant (int): The maximum number of peaks to retain per MS2 spectrum.
        callback (callable): A function that accepts a float between 0 and 1 as progress. Defaults to None.

    Returns:
        tuple: A dictionary with all the raw data, a string with the acquisition_date_time and a string with the vendor.

    """
    from pyteomics import mzml
    import os
    import re
    import logging
    import datetime
    import pathlib
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    try:
        reader = mzml.read(filename, use_index=True)
        spec_indices = np.array(range(1, len(reader) + 1))
    except OSError:
        logging.info('Could not open the file. Please, specify the correct path to the file.')
        sys.exit(1)

    scan_list = []
    rt_list = []
    mass_list = []
    int_list = []
    ms_list = []
    prec_mzs_list = []
    mono_mzs_list = []
    charge_list = []
    
    vendor = "Unknown"
    
    for idx, i in enumerate(spec_indices):
        try:
            spec = next(reader)

            if idx == 0:
                ext = re.findall(r"File:\".+\.(\w+)\"", spec['spectrum title'])[0]
                if ext.lower() == 'raw':
                    vendor = "Thermo"

            scan_list.append(i)
            rt, masses, intensities, ms_order, prec_mass, mono_mz, charge = extract_mzml_info(spec)
            
            sortindex = np.argsort(masses)
            
            masses = masses[sortindex]
            intensities = intensities[sortindex]
            
            if ms_order == 2:
                masses, intensities = get_most_abundant(masses, intensities, n_most_abundant)
            rt_list.append(rt)
            
            #Remove zero intensities
            to_keep = intensities>0
            masses = masses[to_keep]
            intensities = intensities[to_keep]
            
            mass_list.append(masses)
            int_list.append(intensities)
            ms_list.append(ms_order)
            prec_mzs_list.append(prec_mass)
            mono_mzs_list.append(mono_mz)
            charge_list.append(charge)
            
        except KeyboardInterrupt as e:
            raise e
        except SystemExit as e:
            raise e
        except Exception as e:
            logging.info(f"Bad scan={i} in mzML file '{filename}' {e}")

        if callback:
            callback((idx+1)/len(spec_indices))

    check_sanity(mass_list)

    scan_list_ms1 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    rt_list_ms1 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    mass_list_ms1 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    int_list_ms1 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    ms_list_ms1 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 1]

    scan_list_ms2 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    rt_list_ms2 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mass_list_ms2 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    int_list_ms2 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    ms_list_ms2 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    prec_mass_list2 = [prec_mzs_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mono_mzs2 = [mono_mzs_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    charge_ms2 = [charge_list[i] for i, _ in enumerate(ms_list) if _ == 2]

    prec_mass_list2 = [
        calculate_mass(mono_mzs_list[i], charge_list[i])
        for i, _ in enumerate(ms_list)
        if _ == 2
    ]

    query_data = {}

    query_data["scan_list_ms1"] = np.array(scan_list_ms1)
    query_data["rt_list_ms1"] = np.array(rt_list_ms1)
    query_data["mass_list_ms1"] = np.array(mass_list_ms1)
    query_data["int_list_ms1"] = np.array(int_list_ms1)
    query_data["ms_list_ms1"] = np.array(ms_list_ms1)

    query_data["scan_list_ms2"] = np.array(scan_list_ms2)
    query_data["rt_list_ms2"] = np.array(rt_list_ms2)
    query_data["mass_list_ms2"] = mass_list_ms2
    query_data["int_list_ms2"] = int_list_ms2
    query_data["ms_list_ms2"] = np.array(ms_list_ms2)
    query_data["prec_mass_list2"] = np.array(prec_mass_list2)
    query_data["mono_mzs2"] = np.array(mono_mzs2)
    query_data["charge2"] = np.array(charge_ms2)

    fname = pathlib.Path(filename)
    acquisition_date_time = datetime.datetime.fromtimestamp(fname.stat().st_mtime).strftime('%Y-%m-%dT%H:%M:%S')

    return query_data, acquisition_date_time, vendor

# %% ../nbs/02_io.ipynb 32
import xml.etree.ElementTree as ET

def __extract_nested(child):
    """
    Helper function to extract nested entries
    """
    if len(child) > 0:
        temp_dict = {}
        for xx in child:
            temp_dict[xx.tag] = __extract_nested(xx)
        return temp_dict
    else:
        if child.text == 'True':
            info = True
        elif child.text == 'False':
            info = False
        else:
            info = child.text
        return info

def extract_mq_settings(path: str) -> dict:
    """Function to return MaxQuant values as a dictionary for a given xml file.

    Args:
        path (str): File name of an xml file.

    Returns:
        dict: A dictionary with MaxQuant info.

    Raises:
        ValueError: When path is not a valid xml file.

    """
    if not path.endswith('.xml'):
        raise ValueError("Path {} is not a valid xml file.".format(path))

    tree = ET.parse(path)
    root = tree.getroot()

    mq_dict = {}

    for child in root:

        mq_dict[child.tag] = __extract_nested(child)

    return mq_dict

# %% ../nbs/02_io.ipynb 35
def parse_mq_seq(peptide: str) -> str:
    """Replaces maxquant convention to alphapept convention.

    ToDo: include more sequences

    Args:
        peptide (str): A peptide sequence from MaxQuant.

    Returns:
        str: A parsed peptide sequence compatible with AlphaPept.

    """
    peptide = peptide[1:-1] #Remove _

    peptide = peptide.replace('(Acetyl (Protein N-term))','a')
    peptide = peptide.replace('M(Oxidation (M))','oxM')
    peptide = peptide.replace('C','cC') #This is fixed and not indicated in MaxQuant

    return peptide

# %% ../nbs/02_io.ipynb 39
def list_to_numpy_f32(
    long_list: list
) -> np.ndarray:
    """Function to convert a list to np.float32 array.

    Args:
        long_list (list): A regular Python list with values that can be converted to floats.

    Returns:
        np.ndarray: A np.float32 array.

    """
    np_array = (
        np.zeros(
            [len(max(long_list, key=lambda x: len(x))), len(long_list)],
            dtype=np.float32,
        )
        - 1
    )
    for i, j in enumerate(long_list):
        np_array[0 : len(j), i] = j

    return np_array

# %% ../nbs/02_io.ipynb 43
import h5py
import os
import time
VERSION_NO="2023_07_09"


class HDF_File(object):
    '''
    A generic class to store and retrieve on-disk
    data with an HDF container.
    '''

    @property
    def original_file_name(self):
        return self.read(
            attr_name="original_file_name"
        )  # See below for function definition

    @property
    def file_name(self):
        return self.__file_name

    @property
    def directory(self):
        return os.path.dirname(self.file_name)

    @property
    def creation_time(self):
        return self.read(
            attr_name="creation_time"
        )  # See below for function definition

    @property
    def last_updated(self):
        return self.read(
            attr_name="last_updated"
        )  # See below for function definition

    @property
    def version(self):
        return self.read(
            attr_name="version"
        )  # See below for function definition

    @property
    def is_read_only(self):
        return self.__is_read_only

    @property
    def is_overwritable(self):
        return self.__is_overwritable

    def read(self):
        pass

    def write(self):
        pass

    def __init__(
        self,
        file_name: str,
        is_read_only: bool = True,
        is_new_file: bool = False,
        is_overwritable: bool = False,
    ):
        """Create/open a wrapper object to access HDF data.

        Args:
            file_name (str): The file_name of the HDF file.
            is_read_only (bool): If True, the HDF file cannot be modified. Defaults to True.
            is_new_file (bool): If True, an already existing file will be completely removed. Defaults to False.
            is_overwritable (bool): If True, already existing arrays will be overwritten. If False, only new data can be appended. Defaults to False.

        """
        self.__file_name = os.path.abspath(file_name)
        if is_new_file:
            is_read_only = False
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            with h5py.File(self.file_name, "w") as hdf_file:
                current_time = time.asctime()
                hdf_file.attrs["creation_time"] = current_time
                hdf_file.attrs["original_file_name"] = self.__file_name
                hdf_file.attrs["version"] = VERSION_NO
                hdf_file.attrs["last_updated"] = current_time
        else:
            with h5py.File(self.file_name, "r") as hdf_file:
                self.check()
        if is_overwritable:
            is_read_only = False
        self.__is_read_only = is_read_only
        self.__is_overwritable = is_overwritable

    def __eq__(self, other):
        return self.file_name == other.file_name

    def __hash__(self):
        return hash(self.file_name)

    def __str__(self):
        return f"<HDF_File {self.file_name}>"

    def __repr__(self):
        return str(self)

    def check(
        self,
        version: bool = True,
        file_name: bool = True,
    ) -> list:
        """Check if the `version` or `file_name` of this HDF_File have changed.

        Args:
            version (bool): If False, do not check the version. Defaults to True.
            file_name (bool): If False, do not check the file_name. Defaults to True.

        Returns:
            list: A list of warning messages stating any issues.

        """
        warning_messages = []
        if version:
            current_version = VERSION_NO
            creation_version = self.version
            if creation_version != current_version:
                warning_messages.append(
                    f"{self} was created with version "
                    f"{creation_version} instead of {current_version}."
                )
        if file_name:
            if self.file_name != self.original_file_name:
                warning_messages.append(
                    f"The file name of {self} has been changed from"
                    f"{self.original_file_name} to {self.file_name}."
                )
        return warning_messages

# %% ../nbs/02_io.ipynb 45
import pandas as pd
