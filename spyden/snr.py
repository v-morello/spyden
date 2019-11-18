import numpy as np
from spyden import noise_mean, noise_std, Template, TemplateBank
from spyden.cpad import cpadpow2


def snratio(data, temp, mu='median', sigma='iqr'):
    """
    Compute the signal-to-noise ratio map of input data using one or multiple
    pulse templates. The background noise standard deviation is either 
    estimated from the data, or can be specified manually.

    Parameters
    ----------
    data: ndarray
        Array of single pulses 1-D or 2-D. Last dimension must be phase.
    temp: Template or TemplateBank
        Noise-free pulse template(s)
    mu: str or float, optional
        Either the method name to evaluate the background noise mean, or
        specify its value directly. See noise_mean() for
        a description of the different methods.
        (default: 'median')
    sigma: str or float, optional
        Either the method name to evaluate the background noise standard 
        deviation, or specify its value directly. See noise_std() for
        a description of the different methods.
        (default: 'iqr')

    Returns
    -------
    snr: ndarray
        The 3-D signal-to-noise ratio map. The shape of this array is always
        (num_profiles, num_templates, num_bins)
        snr[i, j, k] = S/N of profile i, measured with template j at phase
            bin index k.
    mean: ndarray
        The mean of every profile
    std: ndarray
        The estimated white noise standard deviation for every profile
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")

    if not data.ndim in (1, 2):
        raise ValueError("data must be have 1 or 2 dimensions")

    if not isinstance(temp, (Template, TemplateBank)):
        raise ValueError("temp must be a Template or TemplateBank")

    p = data.shape[-1] # number of phase bins
    x = data.reshape(-1, p) # reshape data to 2D if it is 1D

    nprof = x.shape[0]
    ntemp = 1 if isinstance(temp, Template) else len(temp)

    if isinstance(mu, float):
        mean = np.full(nprof, mu)
    elif isinstance(mu, str):
        mean = noise_mean(data, method=mu)
    else:
        raise ValueError("mu must be either a valid noise mean estimation method name, or a float")

    if isinstance(sigma, float):
        std = np.full(nprof, sigma)
    elif isinstance(sigma, str):
        std = noise_std(data, method=sigma)
    else:
        raise ValueError("sigma must be either a valid noise stddev estimation method name, or a float")

    ### Normalise and pad input
    x = (x - mean.reshape(-1, 1)) / std.reshape(-1, 1)
    x = cpadpow2(x).astype(np.float32)

    ### Get S/N using circular convolution theorem
    fx = np.fft.rfft(x).reshape(nprof, 1, -1)

    # this does padding to right number of bins
    # and the correct time-reversal of templates
    y = temp.prepared_data(x.shape[-1]).astype(np.float32)
    fy = np.fft.rfft(y).reshape(1, ntemp, -1)
    snr = np.fft.irfft(fx * fy)
    return snr, mean, std