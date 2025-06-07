import numpy as np


def noise_std_iqr(x):
    """ 
    Estimate white noise standard deviation from the inter-quartile range of 
    the data. Robust to outliers but NOT red noise.
    """
    # If data is of shape (k_1, ..., k_n)
    # stats has shape (num_percentiles, k_1, ..., k_n-1)
    stats = np.percentile(x, (25, 75), axis=-1)
    sigma = (stats[1] - stats[0]) / 1.3489795003921634
    return sigma


def noise_std_diffcov(x):
    """ 
    Estimate white noise standard deviation from the sequence of consecutive
    differences (numpy.diff) of the data. Robust to red noise but NOT outliers.

    If x is the sum of a red noise process with variance s_r^2 and of
    Gaussian white noise with mean m and variance s_w^2, and y = diff(x),
    then:
    E[Cov(y[:-1], y[1:])] = 
        [ 2 s_w^2 + s_r^2    - s_w^2           ]
        [         - s_w^2      2 s_w^2 + s_r^2 ]

    A red noise process with variance s_r^2 is such that the difference 
    between consecutive samples of that process is normally distributed
    with zero mean and variance s_r^2.
    """
    def func(line):
        y = np.diff(line)
        c = np.cov(y[:-1], y[1:])
        sw2 = -c[0, 1]
        return sw2 ** 0.5

    if x.ndim == 1:
        return func(x)
    elif x.ndim == 2:
        return np.asarray([func(line) for line in x])
    else:
        raise ValueError("input must be 1D or 2D")


def noise_mean_median(x):
    return np.median(x, axis=-1)


NOISE_STD_METHODS = {
    'iqr': noise_std_iqr,
    'diffcov': noise_std_diffcov
}


NOISE_MEAN_METHODS = {
    'median': noise_mean_median
}


def get_mean_method(name):
    try:
        func = NOISE_MEAN_METHODS[name]
    except KeyError:
        choices = list(NOISE_MEAN_METHODS.keys())
        raise ValueError("Noise mean estimation method must be one of: {!r}".format(choices))
    return func


def get_std_method(name):
    try:
        func = NOISE_STD_METHODS[name]
    except KeyError:
        choices = list(NOISE_STD_METHODS.keys())
        raise ValueError("Noise stddev estimation method must be one of: {!r}".format(choices))
    return func


def noise_mean(data, method='median'):
    """
    Estimate the mean of the white noise background in each line of data.

    Parameters
    ----------
    data: ndarray
        1D or 2D array, last dimension is phase
    method: str
        Name of the estimation method. Choices are:
        'median': use the median along the phase dimension
        (default: 'median')

    Returns
    -------
    mu: float or ndarray
        Estimated noise mean in every profile in data. float if data is 1D,
        ndarray otherwise.
    """
    func = get_mean_method(method)
    return func(data)


def noise_std(data, method='iqr'):
    """
    Estimate the standard deviation of the white noise background in 
    each line of data.

    Parameters
    ----------
    data: ndarray
        1D or 2D array, last dimension is phase
    method: str
        Name of the estimation method. Choices are:
        'iqr': Estimate from the interquartile range of the data along the
            phase dimension. Robust to outliers but NOT to red noise.
        'diffcov': Estimate from the covariance of consecutive differences
            (numpy.diff) of the data along the phase dimension. Robust to
            red noise but NOT outliers.
        (default: 'iqr')

    Returns
    -------
    sigma: float or ndarray
        Estimated noise standard deviation in every profile in data. float if
        data is 1D, ndarray otherwise.
    """
    func = get_std_method(method)
    return func(data)