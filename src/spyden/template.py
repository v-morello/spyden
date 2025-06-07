import numpy as np
import matplotlib.pyplot as plt
from numpy import log, pi, sin, sqrt, ceil, exp


def normalise(data):
    """
    Normalise data to unit square sum

    Parameters
    ----------
    data: ndarray, 1D

    Returns
    -------
    normalised: ndarray, 1D
    """
    sqsum = (data ** 2).sum()
    return data * sqsum**-0.5


class Template(object):
    """ 
    Create a noise-free pulse template of arbitrary shape from a numpy array.
    NOTE: Use classmethods such as Template.boxcar() or Template.gaussian() to
    generate Templates of pre-defined shape.

    Parameters
    ----------
    data: ndarray
        One dimensional array representing the pulse values
    refbin: int
        The reference bin index of the template. If 'S' is the S/N array
        obtained by circularly convolving this template with some input data X,
        then:
        S[k] = signal-to-noise ratio when X[k] is aligned with the reference
            bin of the template
    reference: str
        A string explaining what the reference bin corresponds to, e.g.
        'start', 'peak', etc. This is only for clarity, and can be anything.
    kind: str
        A string describing the type of function, e.g. 'boxcar', 'gaussian'.
        This is only for clarity, and can be anything.
    shape_params: dict
        A dictionary with any additional shape parameters required to fully
        describe the template. For both boxcars and gaussians, this only 
        contains one key 'w' which is the FWHM of the template expressed in
        number of bins. This is only for clarity, and can be anything.

    Returns
    -------
    temp: Template
        Noise-free pulse template where the data have been normalised to unit
        square sum
    """
    def __init__(self, data, refbin=0, reference='start', 
        kind='undefined', shape_params={}):
        if not isinstance(data, np.ndarray):
            raise ValueError("data must be a np.ndarray instance")
        if data.ndim != 1:
            raise ValueError("data must have exactly one dimension")
        if not data.size:
            raise ValueError("data is empty")

        if not isinstance(refbin, int):
            raise ValueError("refbin must be an int")
        if not 0 <= refbin < data.size:
            raise ValueError("Must have 0 <= refbin < data.size")

        self._data = normalise(data)
        self._refbin = int(refbin)
        self._reference = str(reference)
        self._kind = str(kind)
        self._shape_params = dict(shape_params)

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self.data.size

    @property
    def refbin(self):
        return self._refbin

    @property
    def reference(self):
        return self._reference

    @property
    def kind(self):
        return self._kind

    @property
    def shape_params(self):
        return self._shape_params

    def prepared_data(self, n):
        """
        Returns the pulse template padded to n bins, ready to be FFT convolved
        with some input data with n bins.

        The template is:
        - padded to a total of n bins by appending its specified 
          pad value on the right, 
        - circularly shifted to place the reference bin at index 0
        - properly time-reversed so that the circular convolution theorem 
          correctly applies
        - normalised to unit square sum
        - cast to np.float32 to save computation time later on

        Parameters
        ----------
        n: int
            The total length to which the template must be padded. This will
            generally be a power of two to compute FFTs faster.

        Raises
        ------
        ValueError: if n is smaller than the template size
        """
        # NOTE: normalisation to unit square sum must be done BEFORE padding !

        if not n >= self.size:
            msg = ("Cannot pad template data to length n = {}; this "
                "is shorter than the template size ({}). You are probably "
                "trying to use this template on data that is too short.")
            msg = msg.format(n, self.size)
            raise ValueError(msg)

        # Zero-pad to a total of n bins
        # NOTE: padding has to be done on the right side so that 'refbin'
        # stays the same !
        x = np.pad(
            self.data, 
            (0, n - self.size), # pad on right size up to length n
            mode='constant', 
            constant_values=(0.0, 0.0))

        # Place reference bin at index 0
        x = np.roll(x, -self.refbin)

        # We want to get the array y such that: y[k] = x[-k]
        # NOTE: this is NOT like reversing the array
        # since y[0] = x[0]
        x = np.roll(x[::-1], 1)
        return x.astype(np.float32)

    @classmethod
    def boxcar(cls, w):
        """ 
        Generate a boxcar pulse template with given width, expressed in number
        of bins.

        Parameters
        ----------
        w: int
            Width in number of bins

        Returns
        -------
        t: Template
            A boxcar pulse template. The reference bin is located at the start
            of the boxcar, that is bin index 0.
        """
        if not isinstance(w, int):
            raise ValueError("w must be of type int")
        if not w > 0:
            raise ValueError("w must be strictly positive")
        shape = {'w': w}
        return cls(np.ones(w), refbin=0, reference='start', kind='boxcar', shape_params=shape)

    @classmethod
    def gaussian(cls, w):
        """
        Generate a Gaussian pulse template with given pulse FWHM, expressed
        in number of bins. The total number of bins in the template is chosen
        so that the range [-3.5 sigma, +3.5 sigma] is covered, where

        sigma = w / (2 * sqrt(2 * log(2)))

        giving a total of approximately 7 x sigma bins, with a minimum of 3.

        Parameters
        ----------
        w: float or int
            FWHM in number of bins. Non-integer values are accepted.

        Returns
        -------
        t: Template
            A Gaussian pulse template. The reference bin is located at the peak
            of the Gaussian.
        """
        if not isinstance(w, (float, int)):
            raise ValueError("w must be of type float or int")
        w = float(w)
        if not w > 0:
            raise ValueError("w must be strictly positive")

        sigma = w / (2 * sqrt(2 * log(2)))
        xmax = int(ceil(3.5 * sigma))
        x = np.arange(-xmax, xmax + 1)
        data = exp(-x**2 / (2 * sigma**2))
        shape = {'w': w}
        return cls(data, refbin=len(x)//2, reference='peak', kind='gaussian', shape_params=shape)

    def plot(self, dpi=100):
        """
        Make a nice plot of the Template

        Returns
        -------
        fig: matplotlib.Figure
        """
        fig = plt.figure(figsize=(6, 4.5), dpi=dpi)
        plt.bar(range(self.size), self.data, width=0.9, color='#b3b3b3')

        ymin, ymax = plt.ylim()
        plt.plot(
            [self.refbin, self.refbin], 
            [ymin, ymax], 
            linestyle='--', color='k', lw=2.0, label='Reference Bin')
        plt.ylim(ymin, ymax)

        plt.xlim(-0.5, self.size - 0.5)
        plt.xlabel("Bin Index")
        plt.ylabel("Amplitude")
        plt.grid(linestyle=':')
        plt.legend(loc='upper right')

        title = str(self)
        plt.title(title)
        plt.tight_layout()
        return fig

    def __str__(self):
        shape_strings = [
            "{:s}={:.3f}".format(key, value)
            for key, value in self.shape_params.items()
            ]
        shape_descr = ','.join(shape_strings)
        return "Template(size={s.size}, kind={s.kind}, {0:s})".format(shape_descr, s=self)

    def __repr__(self):
        return str(self)


class TemplateBank(list):
    """ 
    A convenience class for generating and wrapping a list of Template objects.
    NOTE: Convenience classmethods are provided to easily generate templates

    Parameters
    ----------
    template: iterable
        List or iterable of Template objects
    """
    def __init__(self, templates):
        if not templates:
            raise ValueError("templates is empty")

        if not all(isinstance(t, Template) for t in templates):
            raise ValueError("All input elements must be of Template instances")

        super(TemplateBank, self).__init__(templates)

    @property
    def maxsize(self):
        """ Size of the largest Template """
        return max(t.size for t in self)

    @classmethod
    def boxcars(cls, widths):
        """
        Generate a TemplateBank with boxcars of specified widths. The templates
        will be stored stored in *increasing* width order.

        Parameters
        ----------
        widths: iterable
            List or iterable of ints representing the boxcar widths in number
            of bins

        Returns
        -------
        tbank: TemplateBank
        """
        templates = [
            Template.boxcar(w) 
            for w in sorted(widths)
            ]
        return cls(templates)

    @classmethod
    def gaussians(cls, widths):
        """
        Generate a TemplateBank with Gaussians of specified widths. The 
        templates will be stored stored in *increasing* width order.

        Parameters
        ----------
        widths: iterable
            List or iterable of numbers (both int and float accepted)
            representing the boxcar widths in number of bins

        Returns
        -------
        tbank: TemplateBank
        """
        templates = [
            Template.gaussian(w) 
            for w in sorted(widths)
            ]
        return cls(templates)        

    def prepared_data(self, n):
        """
        Returns the pulse templates padded to n bins, ready to be FFT convolved
        with some input data with n bins. The output is a 2D array. See
        Template.prepared_data() for details.
        """
        return np.asarray([t.prepared_data(n) for t in self])
