import logging
import numpy as np
from psnr import cpadpow2, noise_std


log = logging.getLogger('psnr')


def get_snr(x, t, sigma_method='iqr'):
    """
    x: ndarray
        Array of single pulses, last dimension must be phase
    t: Template
        Noise-free pulse template
    sigma: str or float
        Either the method to evaluate the background noise standard deviation,
        or specify its value directly.
    """
    if not t.size == x.shape[-1]:
        raise ValueError("Template has wrong number of bins")

    # Normalise to zero mean along the last dimension
    n = x.shape[-1]
    stats_shape = (*x.shape[:-1], 1)
    mean = x.mean(axis=-1).reshape(stats_shape)
    std = noise_std(x).reshape(stats_shape)

    # Pad to power of two length along phase axis
    xp = cpadpow2(x - mean)

    fx = np.fft.rfft(xp)
    fy = t._conv_input()
    conv = np.fft.irfft(fx * fy)[..., :n]
    return conv / std


# def get_snr(x, t, sigma_method='iqr'):
#     """
#     x: ndarray
#         Single pulse
#     t: Template
#         Noise-free pulse template
#     """
#     if not t.size == x.shape[-1]:
#         raise ValueError("Template has wrong number of bins")

#     # Normalise to zero mean along the last dimension
#     n = x.shape[-1]
#     xp = cpadpow2(x - x.mean(axis=-1))

#     sigma = noise_std(x)

#     fx = np.fft.rfft(xp)
#     fy = t._conv_input()
#     conv = np.fft.irfft(fx * fy)[..., :n]
#     return conv / sigma


def get_snr_final(x, tbank, sigma='iqr'):
    """
    x: ndarray
    tbank: TemplateBank
    """
    pass