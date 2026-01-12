'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import find_peaks, peak_widths
from pandas import DataFrame

# Import own functions
from retap.preprocessing.single_block_preprocessing import find_main_axis
from retap.feature_extraction.kinematic_features import signalvectormagn
from retap.preprocessing.single_block_preprocessing import remove_acc_nans


def find_tap_timings(acc_triax, fs: int,):
    """
    Detect the moments of finger-raising and -lowering
    during a fingertapping task.
    Function detects the axis with most variation and then
    first detects several large/small pos/neg peaks, then
    the function determines sample-wise in which part of a
    movement or tap the acc-timeseries is, and defines the
    exact moments of finger-raising, finger-lowering, and
    the in between stopping moments. 

    Input:
        - acc_triax (arr): tri-axial accelerometer data-
            array containing x, y, z, shape: [3 x nsamples].
        - main_ax_i (int): index of axis which detected
            strongest signal during tapping (0, 1, or 2)
        - fs (int): sample frequency in Hz
    
    Return:
        - tapi (list of lists): list with full-recognized taps,
            every list is one tap. Every list contains 6 moments
            of the tap: [startUP, fastestUp, stopUP, startDown, 
            fastestDown, impact, stopDown]
        - impacts (list of lists): only containing indices of impact
            moments
        - acc_triax (array): (corrected) tri-axial acc signal
    """
    # data checks
    # if data is DataFRame convert to np array
    if type(acc_triax) == DataFrame: acc_triax = acc_triax.values()
    # transpose if needed
    if np.logical_and(acc_triax.shape[1] == 3,
                      acc_triax.shape[0] > acc_triax.shape[1]):
        acc_triax = acc_triax.T

    if np.isnan(acc_triax).any():
        acc_triax = remove_acc_nans(acc_triax)
        # get timepoints with any nans in all 3 axes
        sel = ~np.isnan(acc_triax).any(axis=0)
        acc_triax = acc_triax[:, sel]
    
    # use main axis to find positive and negative peaks
    main_ax_i = find_main_axis(acc_triax)
    sig = acc_triax[main_ax_i]
    sigdf = np.diff(sig)
    
    # Thresholds for movement detection
    posThr = np.nanmean(sig)
    negThr = -np.nanmean(sig)
    
    # Find peaks to help movement detection
    peaksettings = {'peak_dist': 0.1,
                    'cutoff_time': .25,}
    
    posPeaks = find_peaks(
        sig,
        height=(posThr, np.nanmax(sig)),
        distance=fs * .05,
    )[0]
    negPeak = find_peaks(
        -1 * sig,
        height=-.5e-7,
        distance=fs * peaksettings['peak_dist'] * .5,
        prominence=abs(np.nanmin(sig)) * .05,
    )[0]
    
    # use svm for impact finding
    svm = signalvectormagn(acc_triax)
    impacts = find_impacts(svm, fs)  # svm-impacts are more robust, regardless of main ax

    # delete impact-indices from posPeak-indices
    for i in impacts:
        idel = np.where(posPeaks == i)
        posPeaks = np.delete(posPeaks, idel)
    

    # Lists to store collected indices and timestamps
    tapi = []  # list to store indices of tap
    empty_timelist = np.array([np.nan] * 7)
    # [startUP, fastestUp, stopUP, startDown, fastestDown, impact, stopDown]
    tempi = empty_timelist.copy()
    state = 'lowRest'
    post_impact_blank = int(fs / 1000 * 15)  # last int defines n ms
    blank_count = 0
    end_last_tap_n = 0  # needed for backup filling of tap-start-index

    # Sample-wise movement detection        
    for n, y in enumerate(sig[:-1]):

        if n in impacts:

            state = 'impact'
            tempi[5] = n
        
        elif state == 'impact':
            if blank_count < post_impact_blank:
                blank_count += 1
                continue
            
            else:
                if sigdf[n] > 0:
                    blank_count = 0
                    tempi[6] = n
                    # always set first index of tap
                    if np.isnan(tempi[0]):
                        # if not detected, than use end of last tap
                        tempi[0] = end_last_tap_n + 5

                    tapi.append(np.array(tempi))  # add detected tap-indices as array
                    end_last_tap_n = tempi[6]  # update last impact n to possible fill next start-index

                    tempi = empty_timelist.copy()  # start with new empty list
                    state='lowRest'  # reset state
                    

        elif state == 'lowRest':
            # debugging to get start of tap every time in
            if np.logical_and(
                y > posThr,  # try with half the threshold to detect start-index
                sigdf[n] > np.percentile(sigdf, 50)  # was 75th percentile 
            ):                
                state='upAcc1'
                tempi[0] = n  # START OF NEW TAP, FIRST INDEX
                
        elif state == 'upAcc1':
            if n in posPeaks:
                state='upAcc2'

        elif state == 'upAcc2':
            if y < 0:  # crossing zero-line, start of decelleration
                tempi[1] = n  # save n as FASTEST MOMENT UP
                state='upDec1'

        elif state=='upDec1':
            if n in posPeaks:  # later peak found -> back to up-accel
                state='upAcc2'
            elif n in negPeak:
                state='upDec2'

        elif state == 'upDec2':
            if np.logical_or(y > 0, sigdf[n] < 0):
                # if acc is pos, or goes into acceleration
                # phase of down movement
                state='highRest'  # end of UP-decell
                tempi[2]= n  # END OF UP !!!

        elif state == 'highRest':
            if np.logical_and(
                y < negThr,
                sigdf[n] < 0
            ):
                state='downAcc1'
                tempi[3] = n  # START OF LOWERING            

        elif state == 'downAcc1':
            if np.logical_and(
                y > 0,
                sigdf[n] > 0
            ):
                state='downDec1'
                tempi[4] = n  # fastest down movement
    
    tapi = tapi[1:]  # drop first tap due to starting time

    return tapi, impacts, acc_triax


def find_impacts(uni_arr, fs):
    """
    Function to detect the impact moments in
    (updrs) finger (or hand) tapping tasks.
    Impact-moment is defined as the moment
    where the finger (or hand) lands on the
    thumb (or the leg) after moving down,
    also the 'closing moment'.
    For NOW (07.07.22) work with v2!

    PM: *** include differet treshold values for good and
    bad tappers; or check for numbers of peaks
    detected and re-do with lower threshold in case
    of too little peaks ***

    Input:
        - ax_arr: 1d-array of the acc-axis
            which recorded most variation /
            has the largest amplitude range.
        - fs (int): sample freq in Hz
    
    Returns:
        - impacts: impact-positions of method v1
    """
    thresh = np.nanmax(uni_arr) * .2
    arr_diff = np.diff(uni_arr)
    df_thresh = np.nanmax(arr_diff) * .2  # was .35 (14.12)
    
    pos_peaks = find_peaks(
        uni_arr,
        height=(thresh, np.nanmax(uni_arr)),
        distance=fs / 6,  # was not defined (14.12)
    )[0]

    # select peaks with surrounding pos- or neg-DIFF-peak
    impact_pos = [np.logical_or(
        any(arr_diff[i - 3:i + 3] < -df_thresh),
        any(arr_diff[i - 3:i + 3] > df_thresh)
    ) for i in pos_peaks]
    
    impacts = pos_peaks[impact_pos]
    
    impacts = delete_too_close_peaks(
        acc_ax=uni_arr, peak_pos=impacts,
        min_distance=fs / 6,
    )

    return impacts


def delete_too_wide_peaks(
    acc_ax, peak_pos, max_width
):
    impact_widths = peak_widths(
        acc_ax, peak_pos, rel_height=0.5)[0]
    sel = impact_widths < max_width
    peak_pos = peak_pos[sel]

    return peak_pos


def delete_too_close_peaks(
    acc_ax, peak_pos, min_distance
):
    """
    Deletes tapping peaks which are too close on each
    other.

    Input:
        - acc_ax (array): uni-axial acc-signal
            from which the peaks are detected
        - peak_pos (array): containing the samples
            on which peaks are detected
        - fs (int): sample frequency
        - min_distance (int): peaks closer too each other
            than this minimal distance (in samples)
            will be removed
    Returns:
        - peak_pos: array w/ selected peak-positions
    """
    pos_diffs = np.diff(peak_pos)
    del_impacts = []
    acc_ax = np.diff(acc_ax)  # use diff as decision selection
    for n, df in enumerate(pos_diffs):
        if df < min_distance:
            
            pos1, pos2 = peak_pos[n], peak_pos[n + 1]
            peak1, peak2 = acc_ax[pos1], acc_ax[pos2]
            if peak1 >= peak2:
                del_impacts.append(n + 1)

            else:
                del_impacts.append(n)
            
            for hop in [2, 3]:  # check distances to 2nd, 3rd
                try:
                    pos1, pos2 = peak_pos[n], peak_pos[n + hop]
                    
                    if (pos2 - pos1) < min_distance:
                        peak1, peak2 = acc_ax[pos1], acc_ax[pos2]

                        if peak1 >= peak2:
                            del_impacts.append(n + hop)
                        else:
                            del_impacts.append(n)
                except IndexError:
                    pass
    
    peak_pos = np.delete(peak_pos, del_impacts)

    return peak_pos