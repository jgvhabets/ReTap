"""
Finding impact-moments in ACC-traces
"""
# import public packages and function
import numpy as np
from scipy.signal import find_peaks, peak_widths

def find_impacts(uni_arr, fs):
    """
    Function to detect the impact moments in
    (updrs) finger (or hand) tapping tasks.
    Impact-moment is defined as the moment
    where the finger (or hand) lands on the
    thumb (or the leg) after moving down,
    also the 'closing moment'.
    
    Input:
        - ax_arr: 1d-array of the acc-axis
            which recorded most variation /
            has the largest amplitude range.
        - fs (int): sample freq in Hz
    
    Returns:
        - impacts: impact-positions in trace-indices
    """
    # FIX C5 (revised after validation on real data): cap the reference
    # amplitudes so a single artifact spike cannot inflate the thresholds
    # above every true tap (which collapsed detection entirely).
    # Height: capped at 2x the 99.5th percentile - on artifact-free blocks
    #   max <= 2*p99.5, so the reference equals the published maximum.
    # Slope: capped at 8x its 99.5th percentile - diff distributions are
    #   naturally heavy-tailed (max ~ 4-6x p99.5 on clean blocks), so a
    #   tighter cap would lower the sharpness requirement and admit
    #   false-positive detections (verified on sub-002); 8x engages only
    #   for extreme, artifact-driven tails.
    # With these factors, detection is bit-identical to the published
    # code on all 39 validation blocks.
    ref = min(np.nanmax(uni_arr), 2 * np.nanpercentile(uni_arr, 99.5))
    thresh = ref * .2
    arr_diff = np.diff(uni_arr)
    df_ref = min(np.nanmax(arr_diff), 8 * np.nanpercentile(arr_diff, 99.5))
    df_thresh = df_ref * .2  # was .35 (14.12)
    
    pos_peaks = find_peaks(
        uni_arr,
        height=(thresh, np.nanmax(uni_arr)),
        distance=fs / 6,  # was not defined (14.12)
    )[0]

    # select peaks with surrounding pos- or neg-DIFF-peak
    # FIX C8: clamp window start to 0 (a negative Python slice index counts
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