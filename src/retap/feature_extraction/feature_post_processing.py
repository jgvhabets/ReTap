"""
Feature calculations functions
"""

# Import public packages and functions
import numpy as np



def ft_decrement(
    ft_array: list,
    method: str,
    n_taps_mean: int = 5,
):
    """
    Calculates the proportional decrement within
    feature values per tap.
    Positive decrement means increase in feature,
    negative decrement means decrease over time.

    If less than 10 tap-values available, zeroes
    are returned (no nan's for prediction analysis
    functionality)

    Inputs:
        - ft_array: feature values (one per tap), to
            calculate the decrement over
        - method: method to calculate decrement:
            - diff_in_mean calculates the normalised
                difference between first and last taps
            - regr_slope takes the normalised slope
                of a fitted linear regression line
        - n_taps_mean: numer of taps taking to average
            beginning and end taps (only in means method)
    """
    avail_methods = ['diff_in_mean', 'regr_slope']
    assert method in avail_methods, ('method for '
        f'decrement calc should be in {avail_methods}'
    )

    # if there is no array of feature-score given, return nan
    if not isinstance(ft_array, np.ndarray):
        return 0

    if len(ft_array) < 8:
            
        return 0

    # loop over arrays with amp-values
    if method == 'diff_in_mean':

        startMean = np.nanmean(ft_array[:n_taps_mean])
        endMean = np.nanmean(ft_array[-n_taps_mean:])

        if np.isnan(startMean): return 0

        # decrement is difference between end and start
        # normalised against 90-perc of max amplitude
        decr = (endMean - startMean) / startMean

        return decr

    elif method == 'regr_slope':
        ft_array = ft_array[~np.isnan(ft_array)]  # exclude nans
        try:
            slope, intercept = np.polyfit(
                np.arange(len(ft_array)), ft_array, 1)
        except:
            if len(ft_array) == 1:
                slope = 0
            else:
                raise ValueError('Error in np.polyfit() clope-calc')

        return slope



def aggregate_arr_fts(
    method, ft_array
):
    """
    Aggregate array-features (calculated
    per tap in block) to one value per
    block.
    """
    assert method in [
        'mean', 'median', 'stddev', 'sum', 'variance',
        'coefVar', 'IQR'
    ], f'Inserted method "{method}" is incorrect'

    # if there is no array of feature-score given, return nan
    if not isinstance(ft_array, np.ndarray):
        return np.nan
    
    if np.isnan(ft_array).any():

        ft_array = ft_array[~np.isnan(ft_array)]


    if method == 'allin1':

        if np.isnan(ft_array).any():
            ft_array = ft_array[~np.isnan(ft_array)]

        return ft_array  # all in one big list

    elif method == 'mean':
        
        return np.nanmean(ft_array)
    
    elif method == 'median':
        
        return np.nanmedian(ft_array)

    elif method == 'stddev':

        ft_array = normalize_var_fts(ft_array)
        
        return np.nanstd(ft_array)

    elif method == 'sum':
        
        return np.nansum(ft_array)

    elif method == 'coefVar':

        # ft_array = normalize_var_fts(ft_array)
        cfVar = np.nanstd(ft_array) / np.nanmean(ft_array)
        # taking nan's into account instead of variation()

        return cfVar
    
    elif method == 'variance':

        ft_array = normalize_var_fts(ft_array)

        return np.var(ft_array)

    elif method == 'IQR':
        
        ft_array = ft_array[~np.isnan(ft_array)]

        if len(ft_array) < 4: return np.nan 

        qr25 = np.percentile(ft_array, 25)
        qr75 = np.percentile(ft_array, 75)
        IQR = qr75 - qr25

        return IQR


def normalize_var_fts(values):

    ft_max = np.nanmax(values)
    ft_out = values / ft_max

    return ft_out


def z_score_array(
    array, save_params=False, use_mean=None, use_sd=None,
):
    """
    if save_params is True: returns z_array, STD_mean, STD_sd
    """

    if isinstance(array, list): array = np.array(array)

    # if given, use predefined parameters
    if use_mean: STD_mean = use_mean
    else: STD_mean = np.nanmean(array)
    if use_sd: STD_sd = use_sd
    else: STD_sd = np.nanstd(array)

    z_array = (array - STD_mean) / STD_sd

    if save_params: return z_array, STD_mean, STD_sd
    
    return z_array


def nan_array(dim: list):
    """Create 2 or 3d np array with nan's"""
    if len(dim) == 2:
        arr = np.array(
            [[np.nan] * dim[1]] * dim[0]
        )
    else:
        arr = np.array(
            [[[np.nan] * dim[2]] * dim[1]] * dim[0]
        ) 

    return arr


def get_means_std_errs(score_lists):
    """
    used to plot feature course over time (figure 3)
    """
    mean_dict, err_dict = {}, {}

    for score in score_lists.keys():
        # get maximum array length for score and create empty nan array
        max_len = max([len(l) for l in score_lists[score]])
        values = np.array([[np.nan] * max_len] * len(score_lists[score]))
        # fill array with value scores per trace (list)
        for i, l in enumerate(score_lists[score]):
            values[i, :len(l)] = l
        # calculate mean value per observations (1st, 2nd, 3rd, etc)
        mean = np.nanmean(values, axis=0)
        mean_dict[score] = mean
        # calculate std-error (defaults to zero if no std dev/err (only 1 value))
        sd = np.nanstd(values, axis=0)
        n_obs = np.array([sum(~np.isnan(values[:, i])) for i in range(values.shape[1])])
        errs = sd / np.sqrt(n_obs)  # std-err = std-dev / sqrt(data-size)
        err_dict[score] = errs
    
    return mean_dict, err_dict

# ### SMOOTHING FUNCTION WITH NP.CONVOLVE

# sig = accDat['40'].On
# dfsig = np.diff(sig)

# kernel_size = 10
# kernel = np.ones(kernel_size) / kernel_size
# sigSm = np.convolve(sig, kernel, mode='same')
# dfSm = np.convolve(dfsig, kernel, mode='same')

# count = 0
# for i, df in enumerate(dfSm[1:]):
#     if df * dfSm[i] < 0: count += 1

# plt.plot(sigSm)
# plt.plot(dfSm)

# plt.xlim(1000, 1500)


# print(count)