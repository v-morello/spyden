import numpy as np


def noise_std_iqr(x):
    """ 
    Estimate white noise standard deviation from the inter-quartile range of 
    the data. Robust to outliers but NOT to low-frequency noise or 
    interference.
    """
    # If data is of shape (k_1, ..., k_n)
    # stats has shape (num_percentiles, k_1, ..., k_n-1)
    stats = np.percentile(x, (25, 75), axis=-1)
    sigma = (stats[1] - stats[0]) / 1.3489795003921634
    return sigma


def noise_std_diff(x):
    """ 
    Estimate white noise standard deviation from the sequence of consecutive
    differences (numpy.diff) of the data. Robust to both outliers and to low
    frequency noise / signals as long as their variations from one sample to
    the next is much smaller than the true white noise standard deviation s.

    If x is the sum of an undesirable signal low frequency signal f and of
    Gaussian noise with mean m and variance s^2, and y = diff(y), then:
    Var(y) = 2 * s^2 + Var(diff(f))

    where diff(f) ~= f' for a low-frequency signal, 
    and Var(diff(f)) <= max((f')^2) 

    As long as |max(f')| << s, the variance of y is not significantly affected.
    Rather than calculating the variance of y directly, we estimate it from 
    the IQR of y, to make it robust to outliers.
    """
    delta = np.diff(x, axis=-1)

    # We could return delta.std(axis=-1) / sqrt(2), but that would be sensitive
    # to outliers. We use the IQR instead.
    stats = np.percentile(delta, (25, 75), axis=-1)
    sigma = (stats[1] - stats[0]) / 1.3489795003921634
    return sigma / 2**0.5


NOISE_STD_METHODS = {
    'iqr': noise_std_iqr,
    'diff': noise_std_diff
}


def noise_std(data, method):
    """
    Parameters
    ----------
    data: ndarray
        Last dimension is phase
    method: str

    Returns
    -------
    sigma: ndarray
        Estimated standard deviation of every profile in data
    """
    func = NOISE_STD_METHODS.get(method)
    return func(data)