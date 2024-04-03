from mimas.alphapept.find_features import find_features
from mimas.alphapept.load_mzml_data import load_mzml_data

file_name = "/lab/metabolomics/yli/corrected/PoolQC01_MX612252_PosLIPIDS_postLongevity_Female013.corrected.mzml"
ms_file_data = load_mzml_data(file_name)
features_all, features_mapped = find_features(ms_file_data)
