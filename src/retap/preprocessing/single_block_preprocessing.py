'''
Functions to preprocess raw-accelerometer data of
single tapping traces
'''

# Import public packages and functions
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import variation
from scipy.signal import resample_poly

# Import own functions
from retap.preprocessing.finding_impacts import find_impacts

def preprocess_acc(
    dat_arr,
    fs: int,
    goal_fs: int = 250,
    to_detrend: bool=True,
    to_remove_outlier=True,
    to_check_magnOrder: bool=True,
    to_check_polarity: bool=True,
    main_axis_method: str='minmax',
    verbose: bool=True
):
    """
    Preprocess accelerometer according to defined steps

    Input:
        - dat_arr
        - fs

    Returns:
        - dat_arr
        - main_ax_index
    """
    # resample if necessary
    assert fs >= goal_fs, (
        f'Sampling Frequency should be >= {goal_fs} Hz'
        f', now detected {fs}')
    print(dat_arr.shape)
    print(dat_arr)
    dat_arr = dat_arr.astype(np.float64)

    if fs > goal_fs:
        dat_arr = resample_poly(x=dat_arr, up=1, axis=-1,
            down=int(fs / goal_fs), )
        fs = goal_fs  # correct fs
        
    main_ax_index = find_main_axis(dat_arr, method=main_axis_method,)

    if to_check_magnOrder: dat_arr = check_order_magnitude(
        dat_arr, main_ax_index)

    if to_detrend: dat_arr = detrend_bandpass(dat_arr, fs)

    if to_check_polarity: dat_arr = check_polarity(
        dat_arr, main_ax_index, fs, verbose=False)

    if to_remove_outlier: dat_arr = remove_outlier(
        dat_arr, main_ax_index, fs, verbose)  # replaces outliers with nans
    

    return dat_arr, main_ax_index


def find_main_axis(dat_arr, method: str = 'minmax',):
    """
    Select acc-axis which recorded tapping the most

    Input:
        - dat_arr (arr): triaxial acc signals
    Returns:
        - main_ax_index (int): [0, 1, or 2], axis
            with most tapping activity detected
    """
    assert len(dat_arr.shape) == 2, print(
        'shape of inserted data is not 2-dimensional'
    )

    methods = ['minmax', 'variance']
    if method not in methods:
        raise ValueError('given method incorrect')

    if dat_arr.shape[0] > dat_arr.shape[1]:
        dat_arr = dat_arr.T

    if method == 'minmax':
        maxs = np.nanmax(dat_arr, axis=1)
        mins = abs(np.nanmin(dat_arr, axis=1))
        main_ax_index = np.argmax(maxs + mins)
    
    elif method == 'variance':
        var = [
            variation(dat_arr[i, :]) for i in range(
                dat_arr.shape[0]
            )
        ]
        main_ax_index = np.argmax(var)

    return main_ax_index


def detrend_bandpass(
    dat_array, fs: int, lowcut: int=1, highcut: int=100, order=5
):
    """
    Apply bandpass filter to detrend drift in acc-data, effect
    is based on highpass effect.
    """
    nyq = fs / 2
    b, a = butter(
        order,
        [lowcut / nyq, highcut / nyq],
        btype='bandpass'
    )
    filt_dat = filtfilt(b,a, dat_array)

    return filt_dat


def remove_outlier(
    dat_arr, main_ax_index, fs,
    verbose=True,
):
    """
    Removes large outliers, empirical threshold testing
    resulted in using a percentile multiplication.
    Replaces outliers and the half second around them
    with np.nan's.
    """
    main_ax = dat_arr[main_ax_index]
    halfBuff = int(fs * .3)
    thresh = 10 * np.percentile(main_ax, 99)

    outliers = np.logical_or(
        main_ax < -thresh, main_ax > thresh)
    if np.sum(outliers) == 0: return dat_arr

    # always report outlier removal when samples are actually removed
    # (function returns above if no outliers are present; previously
    # removal was silent under verbose=False, the pipeline default)
    print(
        f'{np.sum(outliers)} outlier-timepoints to remove '
        f'(threshold: {thresh:.4f})'
    )

    # create boolean to remove
    # vectorized version of the previous per-sample loop
    # (identical result: mark the open interval (i-halfBuff, i+halfBuff)
    # around every outlier sample i); the old loop was O(n_samples *
    # n_outliers) and could hang for minutes on long recordings
    remove_i = np.zeros(len(main_ax), dtype=bool)
    for i in np.where(outliers)[0]:
        remove_i[max(0, i - halfBuff + 1):i + halfBuff] = True

    # replace with nan
    dat_arr[:, remove_i] = np.nan

    return dat_arr


def interpolate_nan_gaps(dat_arr, fs, verbose: bool = True):
    """
    Repair NaN gaps by linear interpolation instead of the
    previous delete-and-concatenate (which compressed the time base:
    every interval spanning a deleted gap became silently too short,
    and taps at gap edges could merge into pseudo-taps).

    The interpolated samples are NOT real data - no taps can be
    detected inside a gap - their only purpose is to keep the sample
    clock intact so that all indices and inter-tap intervals outside
    the gaps remain correct. Edge-NaNs are filled with the nearest
    valid value.

    Input:
        - dat_arr (array): 3 x n_samples, possibly containing NaNs
        - fs (int): sample frequency
    Returns:
        - dat_arr with NaNs replaced
    """
    dat_arr = np.atleast_2d(dat_arr)
    nan_cols = np.isnan(dat_arr).any(axis=0)
    if not nan_cols.any():
        return dat_arr

    n_nan = int(nan_cols.sum())
    # longest contiguous gap (in seconds), for reporting
    runs, run = [], 0
    for c in nan_cols:
        run = run + 1 if c else (runs.append(run) or 0) if run else 0
    if run: runs.append(run)
    if verbose: print(
        f'\tinterpolating {n_nan} NaN-samples '
        f'({n_nan / fs:.2f} s total, longest gap '
        f'{max(runs) / fs:.2f} s) - interpolated stretches contain no '
        'real data, no taps are detectable inside them'
    )
    x = np.arange(dat_arr.shape[1])
    for ax in range(dat_arr.shape[0]):
        row = dat_arr[ax]
        good = ~np.isnan(row)
        if good.sum() < 2: continue
        dat_arr[ax] = np.interp(x, x[good], row[good])

    return dat_arr


def remove_outlier_block(dat_arr, fs, verbose: bool = True):
    """
    Outlier removal at BLOCK level with a movement-based threshold,
    replacing the whole-recording variant for the extracted block data
    (config-key "outlier_removal": "per_block").

    The published whole-recording threshold (10x the 99th percentile of
    the full trace) collapses towards the noise floor in long,
    rest-dominated recordings, so the strongest (most informative) tap
    impacts were flagged as outliers and destroyed. Within a tapping
    block the 99th percentile of |acc| reflects actual tap amplitude,
    so 10x that value only removes true non-tap artifacts.

    Flagged samples (plus 0.3 s on each side, as published) are set to
    NaN on all axes. The NaNs are stored explicitly in the block-csv
    and repaired by interpolation at feature-extraction time, where
    features overlapping a gap are excluded (interpolated samples
    carry no real data, they only preserve the sample clock).

    Input:
        - dat_arr (array): 3 x n_samples of ONE tapping block
        - fs (int): sample frequency
    Returns:
        - dat_arr: cleaned block (may contain NaN)
    """
    dat_arr = np.atleast_2d(np.asarray(dat_arr, dtype=float))
    main_ax = dat_arr[find_main_axis(dat_arr)]
    thresh = 10 * np.nanpercentile(np.abs(main_ax), 99)

    outliers = np.abs(main_ax) > thresh
    if outliers.any():
        if verbose: print(
            f'\t{int(outliers.sum())} outlier-timepoints removed within '
            f'block (block-level threshold: {thresh:.4f})'
        )
        halfBuff = int(fs * .3)
        remove_i = np.zeros(len(main_ax), dtype=bool)
        for i in np.where(outliers)[0]:
            remove_i[max(0, i - halfBuff + 1):i + halfBuff] = True
        dat_arr[:, remove_i] = np.nan

    # NaNs (from the removal above, or already present in the source
    # data) are intentionally KEPT here: they are stored in the
    # block-csv and interpolated at feature-extraction time, so that
    # features overlapping a gap can be excluded there

    return dat_arr


def check_order_magnitude(dat_arr, main_ax_index):
    """
    Checks and corrects if the order of magnitude of
    the acc-signal is in range [0 - 10] m/s/s, equal
    to range in g (9.81 m/s/s).
    Occasionaly sensor recordings are in order 1e-6,
    or 1e6.
    """
    if np.percentile(dat_arr[main_ax_index], 99) < 1e-2:
    
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] / 1e-6
    
    elif np.percentile(dat_arr[main_ax_index], 99) > 1e2:
    
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * 1e-6

    return dat_arr


def check_polarity(
    dat_arr, main_ax_index: int, fs: int,
    verbose: bool = False):
    """
    Check whether accelerometer was placed correctly.
    Correct is defined as when upwards movement is
    recorded as positive acceleration.
    """
    
    main_ax = dat_arr[main_ax_index]

    impacts = find_impacts(main_ax, fs)

    if len(impacts) == 0:
        print('No impacts-peaks detected in polarity preprocess function')
        return dat_arr

    count = 0    
    for pos in impacts:
        area_pre = main_ax[
            pos - int(fs / 10):pos - int(fs / 50)]
        posRMS = sum([area_pre[area_pre > 0]][0] ** 2)
        negRMS = sum([area_pre[area_pre < 0]][0] ** 2)

        if  posRMS > negRMS:
            count += 1

    if (count / impacts.shape[0]) > .5:
        if verbose: print('Pos/Neg switched')
        for i in range(dat_arr.shape[0]):
            dat_arr[i, :] = dat_arr[i, :] * -1

    return dat_arr


def remove_acc_nans(acc_arr):
    """"
    remove NaNs from 1- or 3-axial ACC array
    """
    acc_arr = np.atleast_2d(acc_arr)

    if min(acc_arr.shape) > 1:  # triaxial
        # get timepoints with any nans in all 3 axes
        sel = ~np.isnan(acc_arr).any(axis=0)
        acc_arr = acc_arr[:, sel]

    else:
        sel = ~np.isnan(acc_arr)
        acc_arr = acc_arr[:, sel]
    
    return acc_arr