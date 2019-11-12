import logging

import numpy as np
from psnr import cpadpow2, noise_std, Template, TemplateBank


log = logging.getLogger('psnr')


def get_snr(data, temp, sigma='iqr'):
    """
    Parameters
    ----------
    data: ndarray
        Array of single pulses, last dimension must be phase
    temp: Template or TemplateBank
        Noise-free pulse template(s)
    sigma: str or float
        Either the method to evaluate the background noise standard deviation,
        or specify its value directly.

    Returns
    -------
    snr: ndarray
        shape (k_1, ..., k_n-1, ntemp, nbins)
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")

    if not isinstance(temp, (Template, TemplateBank)):
        raise ValueError("temp must be a Template or TemplateBank")

    if not isinstance(sigma, (str, float, int)):
        raise ValueError("sigma must be a string or a number")

    p = data.shape[-1] # number of phase bins

    nprof = np.prod(data.shape[:-1]) if data.ndim > 1 else 1
    ntemp = 1 if isinstance(temp, Template) else temp.ntemp

    # Reshape data to 2D until calculations are done
    x = data.reshape(nprof, p)
    
    # Noise stats, 1D
    mean = x.mean(axis=1)
    std = noise_std(x)

    # Normalise to zero mean
    # Pad to power of two length along phase axis
    xp = cpadpow2(x - mean.reshape(-1, 1))

    fx = np.fft.rfft(xp).reshape(nprof, 1, -1)
    fy = np.fft.rfft(temp.prepared_data(p).reshape(1, ntemp, -1))

    conv = np.fft.irfft(fx * fy)[..., :p]
    snr = conv / std.reshape(nprof, 1, 1)

    # Reshape snr and stats to expected shape
    snr = snr.reshape(*data.shape[:-1], ntemp, p)

    mean = mean.reshape(*data.shape[:-1])
    std = std.reshape(*data.shape[:-1])
    return snr, mean, std
