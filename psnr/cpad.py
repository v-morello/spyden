import logging
import numpy as np
from numpy import ceil, log2


def ceilpow2(n):
    return 2 ** int(ceil(log2(n)))


def cpadpow2(x):
    """
    Circularly pad the last dimension of ndarray 'x' to a length that is a power of 2

    Parameters
    ----------
    x: ndarray

    Returns
    -------
    y: ndarray
    """
    n = x.shape[-1]
    N = ceilpow2(n)
    pad_width = (x.ndim - 1) * [(0, 0)] + [(0, N - n)]
    return np.pad(x, pad_width, mode='wrap')