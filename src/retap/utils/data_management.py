"""
Utilisation functions to manage files and data
within ReTap-toolbox's functionality
"""

# import public packages and functions
import os
from os.path import join, exists, splitext, dirname
from os import makedirs, pardir
import sys
import json
from pandas import read_excel
import numpy as np
import pickle
from joblib import load


def set_workingdirectory(goal_path='retap'):
    wd = os.getcwd()
    if not wd.endswith(goal_path):
        if wd.endswith('ReTap'):
            wd = os.path.join(os.getcwd(), 'src', 'retap')
        while not wd.endswith(goal_path):
            wd = os.path.dirname(wd)
        os.chdir(wd)  # set wd to retap
        sys.path.append(os.getcwd())


def read_cfg_file(cfg_filename: str = 'configs.json'):
    """
    Reads Configurations file.
    File should be in ReTap/data/settings, and is default called
    configs.json. Use the template 'configs_template.json' to create
    your own configurations file.
    """
    # check json filetype, and add if extension is missing
    if splitext(cfg_filename)[1] != '.json': cfg_filename += '.json'
    # check existence of file
    
    if exists(cfg_filename):
        cfg_path = cfg_filename
    
    elif not exists(cfg_filename): 
        cfg_path = join('data', 'settings', cfg_filename)

        if not exists(cfg_path):
            set_workingdirectory(goal_path='ReTap')
            cfg_path = join(os.getcwd(), cfg_path)
        
    assert exists(cfg_path), f'cfg_filename ({cfg_filename}) not (in ReTap/data/settings)'
    
    with open(cfg_path, 'r') as json_data:
        cfg = json.load(json_data)

    return cfg


def get_directories_from_cfg(cfg_filename = 'default',):
    """
    Input:
        - cfg_filename: defaults to configs.jsons in read_cfg_file()
            insert custom name if used
    
    Returns:
        - paths: dict containing 'raw', 'results' and 'figures' folders
    """
    # use either default (configs.json) or custom inserted filename
    if cfg_filename == 'default': cfg = read_cfg_file()
    else: cfg = read_cfg_file(cfg_filename)

    paths = {}
    raw_path = cfg['raw_acc_folder']

    assert exists(raw_path), (
        'Raw accelerometer folder in config_json: '
        f'{raw_path} does not exist'
    )
            
    paths['raw'] = raw_path
    par_dir = dirname(raw_path)
    paths['results'] = join(par_dir, 'retap_results')
    paths['figures'] = join(par_dir, 'retap_figures')

    for dir in paths.keys():
        if not exists(paths[dir]):
            makedirs(paths[dir])
            print(f'\t{paths[dir]} is created')

    return paths
    

def load_clf_model(clf_fname: str = 'ReTap_RF_15taps.P'):
    """
    Reads classifier for prediction.
    File should be in ReTap/data/models.
    """
    # check json filetype, and add if extension is missing
    assert splitext(clf_fname)[1] == '.P', 'clf_model should be pickle'
    # check existence of file
    set_workingdirectory(goal_path='ReTap')
    clf_path = join('data', 'models', clf_fname)
    assert exists(clf_path), 'clf_model not found in ReTap/data/models'

    clf = load(clf_path)

    return clf


def get_unique_subs(path):

    files = os.listdir(path)
    subs = [
        f.split('_')[0][-3:] for f in files
        if f[:3].lower() == 'sub'
    ]
    subs = list(set(subs))

    return subs


def get_file_selection(
    path, sub, state,
    joker_string = None
):
    sel_files = []
        
    for f in os.listdir(path):

        if not np.array([
            f'sub{sub}' in f.lower() or
            f'sub{sub[1:]}' in f.lower() or
            f'sub-{sub}' in f.lower()
        ]).any(): continue

        if type(joker_string) == str:

            if joker_string not in f:

                continue

        if state.lower() in f.lower():

            sel_files.append(f)
    
    return sel_files


def get_arr_key_indices(ch_names, hand_code, cfg_fname=None,
                        filename=None):
    """
    creates dict with acc-keynames and indices

    Supprted channel-names:
    - acc-channels are called X, Y, Z, first [X, Y, Z]
        are for the RIGHT-finger, second [X, Y, Z] are
        for the LEFT-finger.
    - acc-channels all called aux
    - acc_channels called [L_X, L_Y, L_Z, R_X, R_Y, R_Z]
    - custom acc-channels names defined in configs.json, should
        be as {'custom_L_X': 'L_X', 'custom_L_Y': 'L_Y', etc}
    """
    # check for given custom naming
    use_custom_naming = False
    if isinstance(cfg_fname, str):
        cfg = read_cfg_file(cfg_fname)
        if cfg['use_custom_channel_naming']:
            custom_coding = cfg['custom_acc_channel_naming']
            use_custom_naming = True

    # empty dict to store
    dict_out = {}

    # set laterality of file for later analysis flow
    if hand_code == 'bilat': file_side = 'bilat'
    elif 'L' in hand_code: file_side = 'left'
    elif 'R' in hand_code: file_side = 'right'

    # name acc which are called XYZ or aux without laterality
    aux_count = 0

    for i, key in enumerate(ch_names):
        
        # use custom naming if defined
        if use_custom_naming:
            if key in list(custom_coding.keys()):
                dict_out[custom_coding[key]] = i

        # L_X, L_Y, etc
        elif key in ['L_X', 'L_Y', 'L_Z', 'R_X', 'R_Y', 'R_Z']:
            dict_out[key] = i
        
        # standard BER acc-coding is X-Y-Z (first right, then left)
        elif key in ['X', 'Y', 'Z']:

            if f'R_{key}' in dict_out.keys():
                # if right exists, make LEFT
                dict_out[f'L_{key}'] = i
            
            else:
                # start with RIGHT keys (first in TMSi files)
                dict_out[f'R_{key}'] = i
        
        elif 'aux' in key.lower():
            # assuming that 3 aux-channels are present, and contain resp. x, y, z
            
            if 'iso' in key.lower(): continue  # ISO is no ACC-channel in TMSi

            if 'L' in hand_code: aux_keys = ['L_X', 'L_Y', 'L_Z']
            elif 'R' in hand_code: aux_keys = ['R_X', 'R_Y', 'R_Z']
            else: raise ValueError('Laterality of tapping hand is not defined'
                                   f' in filename, nor in channel-name for {filename}')
                
            dict_out[aux_keys[aux_count]] = i
            aux_count += 1

    return dict_out, file_side


def save_class_pickle(
    class_to_save,
    path,
    filename,
    extension='.P',
):

    if not os.path.exists(path): os.makedirs(path)
    
    pickle_path = os.path.join(
        path, filename + extension
    )

    with open(pickle_path, 'wb') as f:
        pickle.dump(class_to_save, f)
        f.close()

    return print(f'inserted class saved as {pickle_path}')


def load_class_pickle(
    file_to_load,
):
    """
    Loads saved Classes. When running this code
    the class-definitions have to be called before
    executign this code.

    So, for example:

    from tap_extract_fts.main_featExtractionClass import FeatureSet, singleTrace
    from retap_utils import utils_dataManagement as utilsDataMan

    deriv_path = os.path.join(utilsDataMan.get_local_proj_dir(), 'data', 'derivatives')
    loaded_class = utilsDataMan.load_class_pickle(os.path.join(deriv_path, 'classFileName.P'))


    Input:
        - file_to_load: string including path,
            filename, and extension
    
    Returns:
        - output: variable containing the class
    """

    with open(file_to_load, 'rb') as f:
        output = pickle.load(f)
        f.close()

    return output