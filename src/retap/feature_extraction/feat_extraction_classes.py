"""
Utilisation functions for Feature Extraction

part of (updrsTapping-repo)
ReTap-Toolbox
"""

# import public packages and functions
import os
from dataclasses import dataclass, field
from typing import Any
from itertools import product
from pandas import read_csv
from numpy import logical_and, isnan, loadtxt

from feature_extraction.tapping_extract_features import tapFeatures
from feature_extraction.tapping_time_detect import find_tap_timings
from preprocessing.single_block_preprocessing import preprocess_acc



@dataclass(repr=True, init=True,)
class singleTrace:
    """
    Class to store meta-data, acc-signals,
    and features of one single 10-sec tapping trace

    Input:
        - filepath: filename of file containing data from
            single tapping block. 
            - To identify sample frequency, the filename has to end
                on e.g. xxx_250Hz.csv, if this fails the sampling freq
                (fs) defaults to 250 Hz
            - if filename contains 'RAW' the model assumes that
                the data is not preprocessed and it will
                preprocess the data here. In this case the sampling
                frequency should be given as well.
    """
    filepath: str
    goal_fs: int = 250
    
    def __post_init__(self,):
        # load and store tri-axial ACC-signal
        if os.path.splitext(self.filepath)[1] == '.csv':
            acc_data = read_csv(self.filepath, index_col=False)
            # delete index col without heading if present
            if 'Unnamed: 0' in acc_data.keys():
                del(acc_data['Unnamed: 0'])
                acc_data.to_csv(self.filepath, index=False)

            acc_data = acc_data.values.T  # only np-array as acc-signal

        elif os.path.splitext(self.filepath)[1] == '.txt':
            acc_data = loadtxt(self.filepath, delimiter='\t')

        # define trace-ID
        self.trace_id = os.path.splitext(self.filepath)[0]

         # extract sample freq if given
        try:
            freq_part = self.filepath.lower().split('hz')[0]
            freq_part = freq_part.split('_')[-1]
            self.fs = int(freq_part)
        except:
            self.fs = 250
            print('WARNING: SAMPLING FREQUENCY NOT DEFINED IN NAME'
                  ' -> 250 Hz is assumed')
        # check if raw data is given and needs preprocessing
        if 'RAW' in self.trace_id:
            acc_data, _ = preprocess_acc(
                dat_arr=acc_data,
                fs=self.goal_fs,
                to_detrend=True,
                to_check_magnOrder=True,
                to_check_polarity=True,
                to_remove_outlier=True,
            )

        # set data to attribute (3 rows, n-samples columns)
        setattr(self, 'acc_sig', acc_data)
        
        # Find Single Taps in Acc-trace
        tap_idx, impact_idx, _ = find_tap_timings(acc_triax=self.acc_sig,
                                                  fs=self.fs,)
        # store taps and features in current Class
        setattr(self, 'impact_idx', impact_idx)
        self.fts = tapFeatures(
            triax_arr=self.acc_sig,
            fs=self.fs,
            impacts=self.impact_idx,
            tap_lists=tap_idx,
            updrsSubScore=self.tap_score,
            max_n_taps_incl=self.max_n_taps_incl,
        )

