'''Feature Extraction Preparation Functions'''

# Import public packages and functions
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import uniform_filter1d
from pandas import DataFrame

# Import own functions
from retap.preprocessing.single_block_preprocessing import find_main_axis
from retap.feature_extraction.kinematic_features import signalvectormagn
from retap.preprocessing.single_block_preprocessing import remove_acc_nans


def find_tap_timings(acc_triax, fs: int, backfill: bool = True,):
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

                    tempi_arr = np.array(tempi)
                    # complete tap-moments the state machine missed
                    # (fills ONLY NaN slots, existing values are never
                    # changed; see backfill_tap_moments; set config-key
                    # "backfill_timestamps" false to disable)
                    if backfill and np.isnan(tempi_arr[1:5]).any():
                        tempi_arr = backfill_tap_moments(tempi_arr, sig, fs)
                    tapi.append(tempi_arr)  # add detected tap-indices as array
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


def backfill_tap_moments(tap, sig, fs, smooth_samples: int = 3):
    """
    Complete tap-phase timestamps which the sample-wise state
    machine left as NaN, computed post hoc from the signal between the
    two always-known anchors startUP (slot 0) and impact (slot 5).
    Controlled by config-key "backfill_timestamps" (default true; set
    false to reproduce the originally published pipeline, in which
    undetermined timestamps remain NaN).

    The slots keep their published physical definitions, but are found
    globally on the closed tap segment instead of during the sample-wise
    walk (which cannot recover from a missed transition):
        - fastestUp   (slot 1): maximum of the within-tap velocity
            profile (cumulative integral of acceleration); equivalent
            to the +/- acceleration zero-crossing of the state machine
        - fastestDown (slot 4): minimum of the velocity profile after
            fastestUp
        - stopUP      (slot 2): first sample after fastestUp where
            acceleration returns to >= 0 (end of up-deceleration)
        - startDown   (slot 3): last sample before fastestDown where
            acceleration is >= 0 (start of down-acceleration)

    ONLY NaN slots are filled - values found by the state machine are
    never changed. A candidate that would violate the physiological
    ordering (startUP < fastestUp < stopUP <= startDown < fastestDown
    < impact, also with respect to already-filled slots) is discarded
    and the slot remains NaN. fastestUp is additionally only accepted
    when the velocity maximum is verifiable: the window must span at
    least 10 samples, the maximum must lie at least 4 samples from
    both window edges (beyond the smoothing-kernel width), and the
    upward excursion must make up at least 15% of the within-window
    velocity range - otherwise the window demonstrably contains no
    raise phase (e.g. startUP fired late) and all slots stay NaN.

    Input:
        - tap (array): 7 tap-moment sample-indices, possibly with NaNs
        - sig (array): main-axis acc signal the indices refer to
        - fs (int): sample frequency
        - smooth_samples: light smoothing (in samples) applied before
            landmark detection, to avoid noise-driven zero-crossings

    Returns:
        - tap (array): same array with NaN slots 1-4 filled where a
            consistent candidate was found
    """
    tap = np.array(tap, dtype=float)

    # anchors must be known and in order
    if np.isnan(tap[0]) or np.isnan(tap[5]):
        return tap
    t0, t5 = int(tap[0]), int(tap[5])
    # a window shorter than 10 samples (40 ms @ 250 Hz) cannot contain
    # a resolvable raise-plus-deceleration; leave all slots NaN
    if t5 - t0 < 10 or t0 < 0 or t5 > len(sig):
        return tap

    seg = np.asarray(sig[t0:t5], dtype=float)
    if np.isnan(seg).any():
        return tap
    if smooth_samples > 1:
        seg = uniform_filter1d(seg, smooth_samples)

    # within-tap velocity profile (arbitrary units; only extrema matter)
    v = cumulative_trapezoid(seg, initial=0)

    # candidate landmarks (indices relative to t0)
    cand = {1: None, 2: None, 3: None, 4: None}
    i1 = int(np.argmax(v))
    # accept fastestUp only when the velocity maximum is verifiable:
    # (a) it lies clear of both window edges (>= 4 samples, i.e.
    #     beyond the reach of the 3-sample smoothing kernel), and
    # (b) the upward excursion makes up >= 15% of the within-window
    #     velocity range (a shape criterion: the speed scale divides
    #     out, so slow-but-clean raises pass at any velocity).
    # Windows whose integrated velocity never meaningfully rises
    # (late startUP, downward-dominated segment) contain no resolvable
    # raise phase: fastestUp stays NaN, as in the published pipeline,
    # instead of defaulting to the window edge with a spurious
    # near-zero raise velocity.
    v_range = v.max() - v.min()
    if 4 <= i1 <= len(seg) - 4 and v_range > 0 and v[i1] >= 0.15 * v_range:
        cand[1] = i1
        i4 = i1 + int(np.argmin(v[i1:]))
        if i1 < i4 <= len(seg) - 1:
            cand[4] = i4
            # stopUP / startDown: defined on the velocity profile.
            # At the movement apex the velocity has decayed to ~zero:
            # stopUP = first sample after fastestUp where v enters a
            # small band around zero (raise finished), startDown =
            # last sample before fastestDown where v has not yet left
            # it downwards (drop not yet started). With a hover phase
            # these bracket the v~0 plateau; without one they collapse
            # onto the zero-crossing (hover duration ~0, stopUP <=
            # startDown by monotonicity of v between the extrema).
            vmax, vmin = v[i1], v[i4]
            if vmax > vmin:
                eps = 0.05 * (vmax - vmin)
                rel = np.where(v[i1 + 1:i4 + 1] <= eps)[0]
                if len(rel): cand[2] = i1 + 1 + int(rel[0])
                rel = np.where(v[i1 + 1:i4] >= -eps)[0]
                if len(rel): cand[3] = i1 + 1 + int(rel[-1])

    # fill ONLY NaN slots, and only if consistent with already-known
    # neighbouring slots (state-machine values are never modified)
    for slot in (1, 2, 3, 4):
        if not np.isnan(tap[slot]) or cand[slot] is None:
            continue
        val = t0 + cand[slot]
        known_before = [tap[s] for s in range(0, slot) if not np.isnan(tap[s])]
        known_after = [tap[s] for s in range(slot + 1, 6) if not np.isnan(tap[s])]
        lower = max(known_before) if known_before else None
        upper = min(known_after) if known_after else None
        # ordering: strictly increasing; equality is ONLY allowed
        # between stopUP (2) and startDown (3) - the no-hover case
        lo_ok = (lower is None or val > lower
                 or (slot == 3 and not np.isnan(tap[2])
                     and val >= tap[2] and val > max(
                         [tap[s] for s in (0, 1) if not np.isnan(tap[s])],
                         default=-np.inf)))
        up_ok = (upper is None or val < upper
                 or (slot == 2 and not np.isnan(tap[3])
                     and val <= tap[3] and val < min(
                         [tap[s] for s in (4, 5) if not np.isnan(tap[s])],
                         default=np.inf)))
        if lo_ok and up_ok:
            tap[slot] = val

    return tap


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
    # NOTE (evaluated and rejected): percentile-capped threshold
    # references were tested against 39 validation blocks and rejected -
    # capping the slope reference admitted false-positive detections
    # (diff distributions are naturally heavy-tailed), and inert caps
    # added an undefendable deviation from the published method.
    # Thresholds below are the PUBLISHED definitions. Known limitation:
    # a single large artifact inside a block inflates both references
    # and can suppress detection of all true taps in that block
    # (signature: implausibly low tap count + extreme max/p99.5 ratio).
    thresh = np.nanmax(uni_arr) * .2
    arr_diff = np.diff(uni_arr)
    df_thresh = np.nanmax(arr_diff) * .2  # was .35 (14.12)
    
    pos_peaks = find_peaks(
        uni_arr,
        height=(thresh, np.nanmax(uni_arr)),
        distance=fs / 6,  # was not defined (14.12)
    )[0]

    # select peaks with surrounding pos- or neg-DIFF-peak
    # clamp window start to 0 (a negative Python slice index counts
    # from the array END, returning an empty window for peaks in the first
    # 3 samples, which silently rejected them)
    impact_pos = [np.logical_or(
        any(arr_diff[max(0, i - 3):i + 3] < -df_thresh),
        any(arr_diff[max(0, i - 3):i + 3] > df_thresh)
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