import numpy as np


def normalise_template(data):
    """
    Normalise data to zero-mean and unit square sum 
    """
    m = data.mean()
    # This gives the square sum of (data - m)
    sqsum = data.var(ddof=0) * data.size
    return (data - m) * sqsum**-0.5


class Template(object):
    """ """
    def __init__(self, data, phi0):
        """
        data: ndarray
        phi0: float
            The phase centre of the template pulse, expressed in periods
            phi0 must be in the interval [0, 1]
        """
        self._data = normalise_template(data)
        self._phi0 = phi0
        self._ftdata = np.fft.rfft(self._data)

    @property
    def data(self):
        return self._data

    @property
    def ftdata(self):
        return self._ftdata

    @property
    def phi0(self):
        return self._phi0

    @classmethod
    def boxcar(cls, nbins, width):
        """
        nbins: int
            Number of phase bins in the template
        width: int
            Width of the boxcar in number of phase bins
        """
        x = np.zeros(nbins)
        x[:width] = 1.0
        phi0 = (width / 2.0) / nbins
        return cls(x, phi0)



def template_boxcar(nbins, width):
    x = np.zeros(nbins)
    x[:width] = 1.0
    return normalise_template(x)


def template_vonmises(nbins, fwhm):
    pass


def get_stdnoise(data):
    p25, p75 = np.percentile(data, (25, 75))
    return (p75 - p25) / 1.349


def get_snr(data, templates):
    """

    Parameters
    ----------
    data: ndarray or array-like
        Input data (1D)
    templates: list or iterable of Template obejcts
        Noise-free pulse templates used to measure S/N, with the same number of
        phase bins as data.

    Returns
    -------
    out: ?
        Results, for each template
    """
    nbins = data.shape[-1]
    stdnoise = get_stdnoise(data)
    ftdata = np.fft.rfft(data)

    results = []

    for tmp in templates:
        conv = np.fft.irfft(ftdata * tmp.ftdata) / stdnoise
        amax = conv.argmax()
        snr = conv[amax]

        # Phase at which the convolution product peaks
        phi0 = amax / float(nbins)

        # The template is not centered on phase 0, but on tmp.phi0
        phi0 = phi0 - tmp.phi0

        # And make sure to place negative phi0 back in the interval [0, 1]
        if phi0 < 0.0:
            phi0 = phi0 + 1.0

        results.append((tmp, snr, phi0))
    return results
