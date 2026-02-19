import os
import glob

from .config import DATA_DIR


class SizeConsts:
    """Constants for sizes. 
       Used for normalizing the inputs of a neural network.
    """
    ONE_BIT = 1
    ONE_BYTE = 8 * ONE_BIT
    ONE_KB = 1024 * ONE_BYTE
    ONE_MB = 1024 * ONE_KB
    ONE_GB = 1024 * ONE_MB

    ONE_Kb = 1000 * ONE_BIT
    ONE_Mb = 1000 * ONE_Kb
    ONE_Gb = 1000 * ONE_Mb

    GB_TO_MB_SCALE = ONE_Gb / ONE_Mb

    BPS_TO_GBPS = lambda x: x / SizeConsts.ONE_Gb
    GBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Gb

    GBPS_TO_MBPS = lambda x: x * SizeConsts.GB_TO_MB_SCALE

    BPS_TO_MBPS = lambda x: x / SizeConsts.ONE_Mb
    MBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Mb

    BPS_TO_KBPS = lambda x: x / SizeConsts.ONE_Kb
    KBPS_TO_BPS = lambda x: x * SizeConsts.ONE_Kb

    LUMOSCORE_256SPINE_2048GPU = lambda x: x / 1e5


def normalize_size(x):
    """Normalize the input size.
       To prevent excessively large input data from impacting training, 
       we use BPS_TO_GBPS to normalize the capacities and flows of edges. 
       If your data is very small, you can modify this, 
       such as changing it to BPS_TO_MBPS.
    """
    return SizeConsts.LUMOSCORE_256SPINE_2048GPU(x)


def get_dir(props, is_test):
    """Get the train or test directory for the given topology."""
    postfix = "test" if is_test else "train"
    return os.path.join(DATA_DIR, props.topo_name, postfix)


def get_hists_from_folder(folder):
    """Get the list of histogram files from the given folder."""
    hists = glob.glob(folder + "/*.hist")
    return hists


def get_train_test_files(props):
    """Get the train and test files for the given properties."""
    data_train_folder = get_dir(props, is_test=False)
    data_test_folder = get_dir(props, is_test=True)
    if props.train_hist_names != 'all':
        train_hist_files = []
        hist_names = props.train_hist_names.split(',')
        for hist_name in hist_names:
            if os.path.exists(f'{data_train_folder}/{hist_name}.hist'):
                train_hist_files.append(f'{data_train_folder}/{hist_name}.hist')
    else:
        train_hist_files = get_hists_from_folder(data_train_folder)

    if props.test_hist_names != 'all':
        test_hist_files = []
        hist_names = props.test_hist_names.split(',')
        for hist_name in hist_names:
            if os.path.exists(f'{data_test_folder}/{hist_name}.hist'):
                test_hist_files.append(f'{data_test_folder}/{hist_name}.hist')
    else:
        test_hist_files = get_hists_from_folder(data_test_folder)

    return train_hist_files, test_hist_files


def print_to_txt(result, path):
    """Print the result to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        for item in result:
            f.write('%s\n' % float(item))
