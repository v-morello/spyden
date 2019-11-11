import numpy as np


def noise_std(data, method='iqr'):
    """
    data: ndarray
        Last dimension is phase

    mu: ndarray
    sigma: ndarray
    """
    functions = {
        'iqr': noise_std_iqr
    }
    func = functions.get(method)
    return func(data)


def noise_std_iqr(data):
    """ """
    # If data is of shape (k_1, ..., k_n)
    # stats has shape (num_percentiles, k_1, ..., k_n-1)
    stats = np.percentile(data, (25, 75), axis=-1)
    sigma = (stats[1] - stats[0]) / 1.3489795003921634
    return sigma
