"""
Utilisation functions to manage files and data
within ReTap-toolbox's functionality
"""

# import public packages and functions
import os
from os.path import join, exists, splitext
from os import makedirs
import json
from pandas import read_excel
import numpy as np
from dataclasses import dataclass
from array import array
import pickle


def read_cfg_file(cfg_file: str = 'configs.json'):
    # check json filetype, and add if extension is missing
    if splitext(cfg_file)[1] != '.json': cfg_file += '.json'
    # check existence of file
    if not exists(cfg_file): 
        cfg_file = join('data', 'settings', cfg_file)

    assert exists(cfg_file), 'inserted cfg_filename does not exist (in data/settings)'
    
    with open(cfg_file, 'r') as json_data:
        cfg = json.load(json_data)

    return cfg


def get_directories_from_cfg():
    cfg = read_cfg_file('config_jh')

    print(cfg)
    paths = {}
    if cfg['main_directory']:
        assert exists(cfg['main_directory']), ('Main directory in config_json: '
                                               f'{cfg["main_directory"]} does not exist')
            
        paths['raw'] = join(cfg['main_directory'], 'raw_acc')
        paths['results'] = join(cfg['main_directory'], 'results')
        paths['figures'] = join(cfg['main_directory'], 'figures')

        for dir in paths.keys():
            if not exists(paths[dir]):
                makedirs(paths[dir])
                print(f'\t{paths[dir]} is created')


    else:
        paths['raw'] = cfg['single_directories']['find_raw_data']
        paths['results'] = cfg['single_directories']['save_results']
        paths['figures'] = cfg['single_directories']['save_figures']

        # check correctness
        for dir in paths.keys():
            assert exists(paths[dir]), ('Incorrect single_directories'
                                        f'{dir} path given in cfg-file')

    return paths
    


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


def get_arr_key_indices(ch_names, hand_code):
    """
    creates dict with acc-keynames and indices

    assumes that acc-channels are called X, Y, Z
    and that the first three are for the left-finger,
    last three for the right-finger

    TODO: add Cfg variable to indicate
    """
    dict_out = {}

    # set laterality of file for later analysis flow
    if hand_code == 'bilat': file_side = 'bilat'
    elif 'L' in hand_code: file_side = 'left'
    elif 'R' in hand_code: file_side = 'right'

    # name acc which are called XYZ or aux without laterality
    aux_count = 0

    for i, key in enumerate(ch_names):
        # standard BER acc-coding is X-Y-Z (first right, then left)
        if key in ['X', 'Y', 'Z']:

            if f'R_{key}' in dict_out.keys():
                # if right exists, make LEFT
                dict_out[f'L_{key}'] = i
            
            else:
                # start with RIGHT keys (first in TMSi files)
                dict_out[f'R_{key}'] = i
        
        elif 'aux' in key.lower():

            if 'iso' in key.lower(): continue  # ISO is no ACC-channel in TMSi

            if 'L' in hand_code: aux_keys = ['L_X', 'L_Y', 'L_Z']
            elif 'R' in hand_code: aux_keys = ['R_X', 'R_Y', 'R_Z']

            dict_out[aux_keys[aux_count]] = i
            aux_count += 1

    return dict_out, file_side


@dataclass(init=True, repr=True)
class triAxial:
    """
    Select accelerometer keys

    TODO: add Cfg variable to indicate
    user-specific accelerometer-key
    """
    data: array
    key_indices: dict

    def __post_init__(self,):

        try:
            self.left = self.data[
                self.key_indices['L_X']:
                self.key_indices['L_Z'] + 1  # +1 to include last index while slicing
            ]
        except KeyError:
            print('No left indices')

        try:
            self.right = self.data[
                self.key_indices['R_X']:
                self.key_indices['R_Z'] + 1
            ]
        except KeyError:
            print('No right indices')


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