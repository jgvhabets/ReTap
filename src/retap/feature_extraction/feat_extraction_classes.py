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
import numpy as np

from feature_extraction.tapping_time_detect import find_tap_timings
import preprocessing.single_block_preprocessing as preprocess
import feature_extraction.kinematic_features as kin_feats
import feature_extraction.feature_post_processing as postExtrCalc


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
    max_n_taps_incl: int = 15
    
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
            acc_data, _ = preprocess.preprocess_acc(
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
        # # store taps and features in current Class
        # setattr(self, 'impact_idx', impact_idx)
        self.fts = tapFeatures(
            triax_arr=self.acc_sig,
            fs=self.fs,
            impacts=impact_idx,
            tap_lists=tap_idx,
            max_n_taps_incl=self.max_n_taps_incl,
        )



@dataclass(init=True, repr=True, )
class tapFeatures:
    """
    Extract features from detected taps in acc-tapping trace

    Input:
        - triax_arr: 2d-array with tri-axial acc-signal
        - fs (int): sample freq in Hz
        - impacts (array): array containing indices of
            impact (closing finger) moments
        - tap_lists: list of taps, every tap has an own array
            with 7 timestamps (in n samples) representing the
            moments during a tap (resulting from continuous
            tapping detect function)
        - max_n_taps_incl: integer defining the number of taps
            consider during ft extraction, defaults to zero,
            and is not considered when being zero.
    """
    triax_arr: Any
    fs: int
    impacts: Any
    tap_lists: dict = field(default_factory=dict)
    max_n_taps_incl: int = 0
    
    def __post_init__(self,):

        if len(self.tap_lists) == 0:  # no taps detected
            return

        if np.isnan(self.triax_arr).any():
            setattr(self,
                    'triax_arr',
                    preprocess.remove_acc_nans(self.triax_arr))


        ax = preprocess.find_main_axis(self.triax_arr, method='minmax',)

        # FEATURES BASED ON FULL TRACE

        # total number of taps (not depending on first 10 taps)
        self.total_nTaps = len(self.impacts)
        
        # total tap-frequency (not depending on first 10 taps)
        self.freq = self.total_nTaps / (
            self.triax_arr.shape[1] / self.fs)
        
        # single tap durations
        self.tap_durations = np.diff(self.impacts) / self.fs

        # total RMS of trace normalised by duration in seconds
        svm_trace = kin_feats.signalvectormagn(self.triax_arr)
        # check and remove nans in trace svm
        if np.isnan(svm_trace).any():
            svm_trace = svm_trace[~np.isnan(svm_trace)]
        rms_trace = kin_feats.calc_RMS(svm_trace)
        norm_rms_trace = rms_trace / (max(self.triax_arr.shape) / self.fs)
        setattr(self, 'trace_RMSn', norm_rms_trace)

        # total entropy of trace (acc to Mahadevan 2020)
        norm_svm = svm_trace / max(svm_trace)
        rounded_norm_svm = np.around(norm_svm, 4)
        entr_trace = kin_feats.calc_entropy(rounded_norm_svm)
        setattr(self, 'trace_entropy', entr_trace)


        if self.max_n_taps_incl > 0:
            setattr(self, 'tap_lists', self.tap_lists[:self.max_n_taps_incl])
        
        # FEATURES BASED ON SINGLE TAPS

        self.intraTapInt = kin_feats.intraTapInterval(
            self.tap_lists, self.fs
        )

        self.tapRMS = kin_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='taps',
            ax=ax,
        )

        self.tapRMSnrm = kin_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='taps',
            ax=ax,
            to_norm=True,
            fs=self.fs,
        )

        self.impactRMS = kin_feats.RMS_extraction(
            self.tap_lists,
            self.triax_arr,
            acc_select='svm',
            unit_to_assess='impacts',
            ax=ax,
            fs=self.fs,
        )

        self.raise_velocity = kin_feats.velocity_raising(
            self.tap_lists, self.triax_arr, ax=ax,
        )  # currently only velocity raising based on svm
        
        self.jerkiness_taps = kin_feats.jerkiness(
            accsig=self.triax_arr,
            fs=self.fs,
            tap_indices=self.tap_lists,
            unit_to_assess='taps',
            smooth_samples=0,
        )

        self.jerkiness_trace = kin_feats.jerkiness(
            accsig=self.triax_arr,
            fs=self.fs,
            tap_indices=self.tap_lists,
            unit_to_assess='trace',
            smooth_samples=0,
        )

        self.tap_entropy = kin_feats.entropy_per_tap(
            accsig=self.triax_arr,
            tap_indices=self.tap_lists,
        )

        ### POST-EXTRACTION ANALYSIS
        fts_to_postExtr_calc = [
            'tapRMS',
            'tapRMSnrm',
            'impactRMS',
            'raise_velocity',
            'intraTapInt',
            'jerkiness_taps',
            'tap_entropy'
        ]

        for ft in fts_to_postExtr_calc:
            # get mean
            setattr(
                self,
                f'mean_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='mean',
                )
            )
            # get coef of variation
            setattr(
                self,
                f'coefVar_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='coefVar',
                )
            )
            # get interquartile range
            setattr(
                self,
                f'IQR_{ft}',
                postExtrCalc.aggregate_arr_fts(
                    ft_array=getattr(self, ft),
                    method='IQR',
                )
            )
            # get decrement over start-mean and end-mean
            setattr(
                self,
                f'decr_{ft}',
                postExtrCalc.ft_decrement(
                    ft_array=getattr(self, ft),
                    method='diff_in_mean',
                    n_taps_mean=3,
                )
            )
            # get slope (absolute slope for entropy and intraTap)
            if ft == 'tap_entropy' or ft == 'intraTapInt':
                # give absolute slope values for entropy
                setattr(
                    self,
                    f'slope_{ft}',
                    abs(postExtrCalc.ft_decrement(
                        ft_array=getattr(self, ft),
                        method='regr_slope',
                    ))
                )
            else:
                setattr(
                    self,
                    f'slope_{ft}',
                    postExtrCalc.ft_decrement(
                        ft_array=getattr(self, ft),
                        method='regr_slope',
                    )
                )

        # clear up space
        self.triax_arr = 'cleaned up'