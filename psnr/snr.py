import numpy as np
from psnr import noise_std, Template, TemplateBank
from psnr.cpad import cpadpow2
from psnr.noisestats import NOISE_STD_METHODS


def snratio(data, temp, sigma='diff'):
    """
    Compute the signal-to-noise ratio map of input data using one or multiple
    pulse templates.

    Parameters
    ----------
    data: ndarray
        Array of single pulses 1-D or 2-D. Last dimension must be phase.
    temp: Template or TemplateBank
        Noise-free pulse template(s)
    sigma: str or float, optional
        Either the method name to evaluate the background noise standard 
        deviation, or specify its value directly.
        (default: 'diff')

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

    if isinstance(sigma, float):
        pass
    elif isinstance(sigma, str):
        if not sigma in NOISE_STD_METHODS:
            valid_choices = list(NOISE_STD_METHODS.keys())
            raise ValueError("Invalid noise estimation method: must be one of {}".format(valid_choices))
    else:
        raise ValueError("sigma must be either a valid noise estimation method name, or a float")

    p = data.shape[-1] # number of phase bins
    x = data.reshape(-1, p) # reshape data to 2D if it is 1D

    nprof = x.shape[0]
    ntemp = 1 if isinstance(temp, Template) else len(temp)

    ### Noise stats
    mean = x.mean(axis=1)
    
    if isinstance(sigma, float):
        # Case where the standard deviation was specified by the user
        std = np.full(nprof, sigma)
    else:
        # Estimate from data
        std = noise_std(x, method=sigma)

    ### Normalise and pad input
    x = (x - mean.reshape(-1, 1)) / std.reshape(-1, 1)
    x = cpadpow2(x)

    ### Get S/N using circular convolution theorem
    fx = np.fft.rfft(x).reshape(nprof, 1, -1)

    # this does padding to p bins and correct time-reversal of templates
    y = temp.prepared_data(p)
    fy = np.fft.rfft(y).reshape(1, ntemp, -1)
    snr = np.fft.irfft(fx * fy)
    return snr, mean, sigma