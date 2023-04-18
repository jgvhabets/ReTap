
from os.path import join, exists
from os import makedirs
import datetime as dt

from utils.data_management import get_directories_from_cfg, save_class_pickle
from feature_extraction.feat_extraction_classes import singleTrace  
#  FeatureSet, check later # mandatory for pickle import
def run_ft_extraction(
    sel_acc_blocks,
    cfg_filename='configs.json',
    save_fts_pickled=False,
):
    """
    Perform feature extraction
    """
    singleTrace(file_path)



