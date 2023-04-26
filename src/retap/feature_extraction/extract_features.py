
from os.path import join, splitext, exists
from os import listdir, makedirs
import datetime as dt
import json
from numpy import ndarray, int64, float64

from utils.data_management import get_directories_from_cfg, save_class_pickle
from feature_extraction.feat_extraction_classes import singleTrace  
#  FeatureSet, check later # mandatory for pickle import
def run_ft_extraction(acc_block_names, cfg_filename='configs.json',
                      verbose=False,):
    """
    Perform feature extraction
    """
    # features to return
    feats_out = {}
    # find available tapping block files
    paths = get_directories_from_cfg(cfg_filename=cfg_filename)
    tap_block_path = join(paths['results'], 'extracted_tapblocks')
    found_files = listdir(tap_block_path)
    # select only files from this itiration (defined by acc_block_names)
    for f in found_files:
        f_in_names = any([f.startswith(n) for n in acc_block_names])
        if not f_in_names: continue
        # if filename corresponds to one of the trace names
        trace = singleTrace(join(tap_block_path, f))
        trace_key = splitext(f)[0]  # take trace name
        if trace_key.endswith('_250Hz'): trace_key = trace_key[:-6]

        feats_out[trace_key] = trace.fts

        # save features as json
        write_dict_to_json(
            path=join(paths['results'], 'features'),
            fname=f'features_{trace_key}',
            dict_to_write=vars(trace.fts)
        )
        
    return feats_out


def write_dict_to_json(
    dict_to_write, path, fname
):
    """
    Prepares dictionary content in order to be
    serialized by JSON function (to enable
    writing to .json)
    """
    # check path existence
    if not exists(path): makedirs(path)
    
    new_dict = {}
    # convert dict content to json-writable content
    for k in dict_to_write:
        v = dict_to_write[k]
        if isinstance(v, ndarray):
            v = v.tolist()
        
        if isinstance(v, list):
            new_list = []
            for v2 in v:
                if isinstance(v2, ndarray): v2 = v2.tolist()
                elif isinstance(v2, float64): v2 = float(v2)
                elif isinstance(v2, int64): v2 = int(v2)

                if isinstance(v2, list):
                    new_list2 = []
                    for v3 in v2:
                        if isinstance(v3, float64): v3 = float(v3)
                        elif isinstance(v3, int64): v3 = int(v3)
                        new_list2.append(v3)
                    v2 = new_list2

                new_list.append(v2)
            
            new_dict[k] = new_list

        elif isinstance(v, int64):
            new_dict[k] = int(v)

        elif isinstance(v, float64):
            new_dict[k] = float(v)

        else:
            new_dict[k] = v


    # write to json
    with open(join(path, fname + '.json'), 'w') as jsonfile:
        json.dump(new_dict, jsonfile)
