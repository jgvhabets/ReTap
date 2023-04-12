'''
Functions to calculate tapping features
'''
# Import general packages and functions
import numpy as np
from scipy.ndimage import uniform_filter1d


def nan_ft_array_base(
    tap_indices: list,
    length_minus_one: bool = False):
    """"
    Create empy nan-array with length
    of number of taps to store features in

    Input:
        - tap_indices: list of arraus with indices
            of movement moments per tap
        - length_minus_one: create length number of
            taps minus one, for features representing
            differences between taps 
    """
    if length_minus_one:
        ft_base = np.array([np.nan] * (len(tap_indices) - 1))
    else:
        ft_base = np.array([np.nan] * len(tap_indices))

    return ft_base

def signalvectormagn(acc_arr):
    """
    Input:
        - acc_arr (array): triaxial array
            with x-y-z axes (3 x n_samples)
    
    Returns:
        - svm (array): uniaxial array wih
            signal vector magn (1, n_samples)
    """
    if acc_arr.shape[0] != 3: acc_arr = acc_arr.T
    assert acc_arr.shape[0] == 3, ('Array must'
    'be tri-axial (x, y, z from accelerometer)')
  
    svm = np.sqrt(
        acc_arr[0] ** 2 +
        acc_arr[1] ** 2 +
        acc_arr[2] ** 2
    )

    return svm


def intraTapInterval(
    tap_indices: list,
    fs: int,
    moment: str = 'impact'
):
    """
    Calculates intratap interval.

    Input:
        - tap_indices: list with arrays of tap-moment
            indices, representing respectively:
            [startUP, fastestUp, stopUP, startDown,
            fastestDown, impact, stopDown]
            (result of updrsTapDetector())
        - fs (int): sample freq
        - moment: timing of ITI calculation, from
            impact-imapct, or start-start
    
    Returns:
        - iti (arr): array with intratapintervals
            in seconds
    """
    assert moment.lower() in ['impact', 'start', 'end'
    ], print(
        f'moment ({moment}) should be start or impact'
    )
    if moment.lower() == 'start': idx = 0
    elif moment.lower() == 'impact': idx = -2
    elif moment.lower() == 'end': idx = -1
    
    iti = nan_ft_array_base(tap_indices, length_minus_one=True)

    for n in np.arange(len(tap_indices) - 1):
        # take distance between two impact-indices
        distance = tap_indices[n + 1][idx] - tap_indices[n][idx]
    
        iti[n] = distance / fs  # from samples to seconds
    
    return iti


def calc_RMS(acc_signal):
    """
    Calculate RMS over total
    acc-signal (can be uni-axis or svm)
    """
    S = np.square(acc_signal)
    MS = S.mean()
    RMS = np.sqrt(MS)

    return RMS


def RMS_extraction(
    tap_indices: list,
    triax_arr,
    acc_select: str,
    ax,
    to_norm: bool = False,
    unit_to_assess: str='taps',
    impact_window: float=.25,
    fs: int = None
):
    """
    Calculates RMS of full acc-signal per tap.

    Input:
        - tap_indices: list of arrays with indices
            representing tap [startUP, fastestUp,
            stopUP, startDown, fastestDown, impact,
            stopDown] (result of updrsTapDetector())
        - triax_arr (array)
        - acc_select: svm or axis, defines which acc-
            signal is used
        - ax: index of main-axis
        - select (str): if full -> return RMS per
            full tap; if impact -> return RMS
            around impact
        - to_norm: bool defining to normalise RMS
            to duration of originating timeframe
        - impact_window (float): in seconds, total
            window around impact to calculate RMS
        - fs (int): sample frequency, required for
            impact-RMS and normalisation
    
    Returns:
        - RMS (arr)
    """
    assess_options = ['run', 'taps', 'impacts']
    assert unit_to_assess in assess_options, ('unit_'
        f'to_asses not in {assess_options} in RMS()')

    if np.logical_or(to_norm, unit_to_assess == 'impacts'):
        assert type(fs) == int, ('Fs has to be integer'
        )

    assert acc_select in ['svm', 'axis'], (
        f'acc_select is incorrect ({acc_select}'
        '), should be "svm" or "axis".'
    )

    if acc_select == 'axis': sig = triax_arr[ax]
    elif acc_select == 'svm': sig = signalvectormagn(triax_arr)

    if unit_to_assess == 'run':
        RMS = calc_RMS(sig)
        if to_norm: RMS /= (len(sig) / fs)  # normalise RMS against duration in sec

        return RMS
    
    else:
        RMS = nan_ft_array_base(tap_indices)

        for n, tap in enumerate(tap_indices):

            tap = tap.astype(int)  # np.nan as int is -999999...

            if unit_to_assess == 'taps':
                sel1 = int(tap[0])
                sel2 = int(tap[-1])
                if np.isnan(sel2): sel2 = int(tap[-2])

            elif unit_to_assess == 'impacts':
                sel1 = int(tap[-2] - int(fs * impact_window / 2))
                sel2 = int(tap[-2] + int(fs * impact_window / 2))

            if np.logical_or(sel1 == np.nan, sel2 == np.nan):
                print('tap skipped, missing indices')
                continue
            
            tap_sig = sig[sel1:sel2]
            
            RMS[n] = calc_RMS(tap_sig)
        
            if to_norm: RMS[n] /= (len(tap_sig) / fs)  # normalise RMS against duration in sec
        
        return RMS


def velocity_raising(tap_indices, triax_arr, ax):
    """
    Calculates velocity approximation via
    area under the curve of acc-signal within
    upwards part of a tap.

    Input:
        - tap_indices
        - triax_arr
        - ax (int): main tap axis index
    
    Returns:
        # - upVelo_uniax (arr)  # currently only velocity based on svm
        - upVelo_triax (arr)
    """
    # ax = triax_arr[ax]
    svm = signalvectormagn(triax_arr)

    # upVelo_uniax = velo_calc_auc(tap_indices, ax)
    upVelo_triax = velo_calc_auc(tap_indices, svm)
    
    return upVelo_triax


def velo_calc_auc(tap_indices, accSig,):
    """
    Calculates max velocity during finger-raising
    based on the AUC from the first big pos peak
    in one tap until the acceleration drops below 0

    Input:
        - tap_indices: dict with lists resulting
            [startUP, fastestUp, stopUP, startDown,
            fastestDown, impact, stopDown]
            (result of updrsTapDetector())
        - accSig (array): uniax acc-array (one ax or svm)
    
    Returns:
        - out (array): one value or nan per tap in tap_indices
    """
    out = []

    for n, tap in enumerate(tap_indices):

        if ~np.isnan(tap[1]):  # crossing 0 has to be known
            # take acc-signal [start : fastest point] of rise
            line = accSig[int(tap[0]):int(tap[1])]
            areas = []
            for s, y in enumerate(line[1:]):
                areas.append(np.mean([y, line[s]]))
            if sum(areas) == 0:
                print('\nSUM 0',n, line[:30], tap[0], tap[1])
            out.append(sum(areas))
    
    return np.array(out)


def jerkiness(
    accsig,
    fs: int,
    tap_indices: list,
    unit_to_assess: str,
    n_hop: int = 1,
    smooth_samples: int = 0,
):
    """
    Detects the number of small changes in
    direction of acceleration.
    Hypothesized is that best tappers, have
    the smoothest acceleration-trace and
    therefore lower numbers of small
    slope changes

    Inputs:
        - accsig (array): tri-axial acceleration
            signal from e.g. 10-s tapping
        - fs: sample freq
        - tap_indices: list with arrays of tap-timing-indices
        - unit_to_assess: calculated per tap or per
            whole trace
        - n_hop (int): the number of samples used
            to determine the difference between
            two points
        - smooth_samples: number of samples to
            smooth signal over
    
    Returns:
        - trace_count: total sum of differential
            changes in all thee axes, returned per
            tap or per whoel trace. Both are norma-
            lised against the tap/trace duration
    """
    assert unit_to_assess in ['trace', 'taps'], print(
        f'given unit_to_asses ({unit_to_assess}) is incorrect'
    )

    if unit_to_assess == 'trace':

        trace_count = 0
        for ax in np.arange(accsig.shape[0]):

            axdiff = np.diff(accsig[ax])
            for i in np.arange(axdiff.shape[0] - n_hop):
                if (axdiff[i + n_hop] * axdiff[i]) < 0:  # removed if -1 < axdiff...
                    trace_count += 1
        # normalise for duration of trace
        duration_trace = accsig.shape[1] / fs
        trace_count = trace_count / duration_trace

        return trace_count


    elif unit_to_assess == 'taps':

        if smooth_samples > 0:
            accsig = uniform_filter1d(accsig, smooth_samples) 

        trace_count = []

        for tap in tap_indices:

            if np.logical_or(
                np.isnan(tap[0]),
                np.isnan(tap[-1])
            ):
                continue

            elif len(tap) == 0:
                continue
            
            else:
                tap_acc = accsig[:, int(tap[0]):int(tap[-1])]
                tap_duration = (tap[-1] - tap[0]) / fs  # in seconds
                count = 0

                for ax in np.arange(accsig.shape[0]):
                    axdiff = np.diff(tap_acc[ax])

                    for i in np.arange(axdiff.shape[0] - n_hop):
                        # count if consecutive diff-values are pos and neg
                        if (axdiff[i + n_hop] * axdiff[i]) < 0:  # removed if -1 < axdiff...
                            count += 1
                
                count = count / tap_duration  # normalise to n jerks per sec
                trace_count.append(count)

    return np.array(trace_count)  # return as array for later calculations


def entropy_per_tap(
    accsig, tap_indices: list,
):
    entr_list = []
    svm = signalvectormagn(accsig)
    for tap in tap_indices:

        if np.logical_or(
            np.isnan(tap[0]),
            np.isnan(tap[-1])
        ):
            continue

        elif len(tap) == 0:
            continue
        
        else:
            tap_svm = svm[int(tap[0]):int(tap[-1])]
            ent = calc_entropy(tap_svm)
            entr_list.append(ent)

    return np.array(entr_list)


from math import log, e

def calc_entropy(signal, base=None):
  """
  Computes entropy of label distribution.
  
  adjusted from: https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
  """

  len_signal = len(signal)

  if len_signal <= 1:
    return 0

  _, counts = np.unique(signal, return_counts=True)
  probs = counts / len_signal
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)

  return ent

