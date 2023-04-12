"""
Run finding and splitting blocks of 10-seconds
from command line
"""



# import public packages
from os import listdir, makedirs
from os.path import join, splitext, exists
from dataclasses import field, dataclass
from typing import List
import numpy as np
from array import array
from typing import Any
from scipy.signal import resample_poly


# import own functions
from utils import data_management, tmsi_poly5reader
import preprocessing.finding_blocks as find_blocks
from preprocessing.single_block_preprocessing import run_preproc_acc


@dataclass(init=True, repr=True)
class rawAccData:
    """
    """
    raw_path: str
    joker_string: Any = None
    goal_fs: int = 250
    STORE_CSV=True  # NOT SAVING AT THE MOMENT
    # switched_sides: List = field(default_factory=list)
    # unilateral_coding_list: list = field(
    #     default_factory=lambda: [
    #         'LHAND', 'RHAND',
    #         'FTL', 'FTR',
    #         'LFTAP', 'RFTAP',
            
    #     ]
    # )
    

    def __post_init__(self,):
        # IDENTIFY FILES TO PROCESS
        csv_path = data_management.get_directories_from_cfg()['raw']
        sel_files = listdir(csv_path)
        # TODO: create function to select on extensions (splitext)

        print(f'files selected: {sel_files}')
        
        print(self.unilateral_coding_list)
        print(type(self.unilateral_coding_list))
        # Abort if not file found
        if len(sel_files) == 0:
            return print(f'No files found for {self.sub} {self.state}')

        for f in sel_files:
            # LOAD FILE
            self.raw = tmsi_poly5reader.Poly5Reader(
                join(self.uncut_path, f)
            )
            hand_code = 'bilat'
            # check if file contains unilateral data
            for code in self.unilateral_coding_list:
                if code.upper() in f.upper():
                    hand_code = code.upper()

            key_ind_dict, file_side = data_management.get_arr_key_indices(
                self.raw.ch_names, hand_code
            )
            if len(key_ind_dict) == 0:
                print(f'No ACC-keys found in keys: {self.raw.ch_names}')
                continue
            print(f'selected channels: {key_ind_dict}'
                  f'\n\tout of {self.raw.ch_names}')

            # select present acc (aux) variables
            file_data_class = data_management.triAxial(
                data=self.raw.samples,
                key_indices=key_ind_dict,
            )

            for acc_side in vars(file_data_class).keys():

                if acc_side in ['left', 'right']:
                    # prevent left-calculations on right-files and viaversa
                    if hand_code != 'bilat':
                        if acc_side != file_side:
                            if self.sub not in self.switched_sides:
                                continue
                            else:  # go on w/ non-matching sides, but invert sides for naming of csv's and plots
                                if acc_side == 'left': save_side = 'R'
                                elif acc_side == 'right': save_side = 'L'
                                f = f + '*'
                        else:
                            if self.sub in self.switched_sides:
                                continue
                            else:  # matching sides, correct left-right acc-sides
                                save_side = acc_side[0].upper()
                    
                    else:  # files recorded unilateral
                        save_side = acc_side[0].upper()

                    # PREPROCESS
                    # resample if necessary
                    if self.raw.sample_rate > self.goal_fs:
                        resampled = resample_poly(
                            x=getattr(file_data_class, acc_side),
                            up=1, axis=-1,
                            down=int(self.raw.sample_rate / self.goal_fs), 
                        )
                        setattr(file_data_class, acc_side, resampled)
                    
                    # preprocess data in class
                    procsd_arr, _ = run_preproc_acc(
                        dat_arr=getattr(file_data_class, acc_side),
                        fs=self.goal_fs,
                        to_detrend=True,
                        to_check_magnOrder=True,
                        to_check_polarity=True,
                        to_remove_outlier=True,
                    )
                    # replace arr in class with processed data
                    setattr(file_data_class, acc_side, procsd_arr)

                    self.data = file_data_class  # store in class to work with in notebook

                    blocks_fig_path = join(
                        data_management.get_directories_from_cfg('figures'),
                        'block_detection_submission'
                    )
                    blocks_csv_path = join(
                        data_management.get_directories_from_cfg('results'),
                        'extracted_tapblocks'
                    )

                    temp_acc, temp_ind = find_blocks.find_active_blocks(
                        acc_arr=getattr(file_data_class, acc_side),
                        fs=self.goal_fs,
                        verbose=True,
                        to_plot=True,
                        plot_orig_fname=f,
                        figsave_dir=blocks_fig_path,
                        figsave_name=(
                            f'{self.sub}_{self.state}_'
                            f'{save_side}_blocks_detected'
                        ),
                        to_store_csv=self.STORE_CSV,
                        csv_dir=blocks_csv_path,
                        # csv_fname=f'{self.sub_csv_code}{self.sub}_'
                        #           f'{self.state}_{save_side}',
                        #           # save_side replaced acc_side[0].upper() to correct for swapped acc-sides
                    )




    