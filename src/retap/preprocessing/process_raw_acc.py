"""
Run finding and splitting blocks of 10-seconds
from command line
"""



# import public packages
from os import listdir, makedirs, getcwd
from os.path import join, splitext, exists
from dataclasses import field, dataclass
from typing import Any, Dict
from collections import defaultdict
from numpy import ndarray

# import own functions
from utils import data_management
import utils.tmsi_poly5reader as poly5_reader
import preprocessing.finding_blocks as find_blocks
from preprocessing.single_block_preprocessing import preprocess_acc


@dataclass(init=True, repr=True)
class ProcessRawAccData:
    """
    Function to process raw accelerometer traces

    Input:

    Raises:
        - ValueError if hand-side (left or right) is not
            defined in neither filename or channelnames
    """
    goal_fs: int = 250
    cfg_filename: str = 'configs.json'
    use_single_file: Any = None
    STORE_CSV: bool = True
    OVERWRITE: bool = True
    feasible_extensions: list = field(default_factory=lambda: ['.Poly5', '.csv'])
    unilateral_coding_list: list = field(
        default_factory=lambda: [
            'LHAND', 'RHAND',
            'FTL', 'FTR',
            'LFTAP', 'RFTAP',
            
        ]
    )
    current_trace_list : list = field(default_factory=lambda: [])
    verbose: bool = False
    

    def __post_init__(self,):
        # IDENTIFY FILES TO PROCESS
        paths = data_management.get_directories_from_cfg(self.cfg_filename)
        # use given file
        if self.use_single_file:
            sel_files = [self.use_single_file,]
        # default consider all files in raw data path
        else:
            raw_path = paths['raw']
            sel_files = listdir(raw_path)

            if self.verbose: print(f'files selected from {raw_path}: {sel_files}')
        
        # Abort if not file found
        if len(sel_files) == 0:
            return print(f'WARNING: ABORTED, no files found in {raw_path}')

        for f in sel_files:
            # check if extension is supported
            if splitext(f)[1] not in self.feasible_extensions:
                print(f'WARNING: File ({f}) skipped, extension not supported')
                continue

            # LOAD FILE
            if splitext(f)[1] == '.Poly5':
                self.raw = poly5_reader.Poly5Reader(join(raw_path, f))
                fs = self.raw.sample_rate
                # set hand-code to bilateral (default)
                hand_code = 'bilat'
                # check if file contains unilateral data
                for code in self.unilateral_coding_list:
                    if code.upper() in f.upper():
                        hand_code = code.upper()

                key_ind_dict, file_side = data_management.get_arr_key_indices(
                    self.raw.ch_names, hand_code, filename=f,
                )
                if len(key_ind_dict) == 0:
                    print(f'WARNING: No ACC-keys found in keys: {self.raw.ch_names}')
                    continue

                # select present acc (aux) variables
                file_data_class = triAxial(
                    data=self.raw.samples,
                    key_indices=key_ind_dict,
                )
            
            # TODO: add functionality for other datatypes to start with
            # elif splitext(f)[1] == '.csv':
            # end in file_data_class creation with left/right present
                # select present acc (aux) variables
                # file_data_class = data_management.triAxial(
                #     data=self.raw.samples,
                #     key_indices=key_ind_dict,
                # )
                # include fs=..
            
            # create TRACE CODE based on filename
            TRACE_CODE = splitext(f)[0]

            for acc_side in vars(file_data_class).keys():
                # skip attr in class not representing acc-side
                if acc_side not in ['left', 'right']: continue
                
                # prevent left-calculations on right-files and viaversa
                if hand_code != 'bilat':
                    if acc_side != file_side: continue

                # define paths and naming
                blocks_fig_path = join(paths['figures'],
                                       'block_detection')
                blocks_csv_path = join(paths['results'],
                                       'extracted_tapblocks')
                csv_fname = f'{TRACE_CODE}_{acc_side}'

                ### PREPROCESS ###
                
                procsd_arr, _ = preprocess_acc(
                    dat_arr=getattr(file_data_class, acc_side),
                    fs=fs,
                    goal_fs=self.goal_fs,
                    to_detrend=True,
                    to_check_magnOrder=True,
                    to_check_polarity=True,
                    to_remove_outlier=True,
                    verbose=self.verbose,
                )
                # replace arr in class with processed data
                setattr(file_data_class, acc_side, procsd_arr)

                self.data = file_data_class  # store in class to work with in notebook

                temp_acc, temp_ind = find_blocks.find_active_blocks(
                    acc_arr=getattr(file_data_class, acc_side),
                    fs=self.goal_fs,
                    verbose=self.verbose,
                    to_plot=True,
                    plot_orig_fname=f,
                    figsave_dir=blocks_fig_path,
                    figsave_name=(f'{TRACE_CODE}_'
                                  f'{acc_side}_blocks_detected'),
                    to_store_csv=self.STORE_CSV,
                    csv_dir=blocks_csv_path,
                    csv_fname=csv_fname,
                )
                self.current_trace_list.append(csv_fname)




@dataclass(init=True, repr=True)
class triAxial:
    """
    Select accelerometer keys

    TODO: add Cfg variable to indicate
    user-specific accelerometer-key
    """
    data: ndarray
    key_indices: Dict[str, int] = field(
        default_factory=lambda: defaultdict(lambda: {
        'L_X': 0, 'L_Z': 2, 'R_X': 3, 'R_Z': 5}
    ))

    def __post_init__(self,):

        try:
            self.left = self.data[
                self.key_indices['L_X']:
                self.key_indices['L_Z'] + 1  # +1 to include last index while slicing
            ]
        except KeyError:
            print('WARNING: No left indices')

        try:
            self.right = self.data[
                self.key_indices['R_X']:
                self.key_indices['R_Z'] + 1
            ]
        except KeyError:
            print('WARNING: No right indices')
    
