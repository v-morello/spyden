import numpy as np
import matplotlib.pyplot as plt


def normalise(data):
    """
    Normalise data to zero-mean and unit square sum
    data: ndarray, 1D
    """
    m = data.mean()
    # This gives the square sum of (data - m)
    sqsum = data.var(ddof=0) * data.size
    return (data - m) * sqsum**-0.5


# Template description:
# - type/kind (boxcar, von mises, etc.)
# - number of bins
# - shape parameters
# - reference ('start', 'peak', etc.)
# - reference bin index

# Example:
# {'kind': 'boxcar', 'nbins': 1024, 'shape': {'w': 5}, 'reference': 'start', 'refbin': 0}

class Template(object):
    """ """
    def __init__(self, data, header):
        """
        data: ndarray
        header: dict
        """
        # TODO: check input, validity of header, etc.
        self._data = normalise(data)
        self._header = header

    @property
    def data(self):
        return self._data

    @property
    def nbins(self):
        return self._data.size

    @property
    def header(self):
        """ Dictionary describing the template type and parameters """
        return self._header

    @property
    def refbin(self):
        """ Reference bin index """
        return self._header['refbin']

    @classmethod
    def boxcar(cls, n, w):
        """ Boxcar with a total of n bins, and a width of w bins """
        data = np.zeros(n)
        data[:w] = 1.0
        header = {
            'kind': 'boxcar',
            'nbins': data.size,
            'shape': {'w': w},
            'reference': 'start',
            'refbin': 0
        }
        return cls(data, header)

    def _conv_input(self):
        """
        Returns the FFT of the properly time-reversed template. This gets
        multiplied with the FFT of the input data to get a convolution
        product as we expect it.

        Time-reversal means that out[k] = in[-k]
        Here we need out[0] = in[0], which means that it is NOT like reversing
        the input array elements via np.flip() or in[::-1]
        """
        # TODO: merge first two operations into one
        # Place reference bin at index 0
        t = np.roll(self.data, -self.refbin)

        # We want to get the array y such that: y[k] = t[-k]
        # NOTE: this is NOT like reversing the array
        # since y[0] = t[0]
        y = np.roll(t[::-1], 1)

        # And now we just have to multiply the FFT of the input data with
        # this to get the convolution product as we expect it
        return np.fft.rfft(y)

    def display(self):
        fig = plot_template(self)
        fig.show()

    def __str__(self):
        return "Template({})".format(self.header)

    def __repr__(self):
        return str(self)


def plot_template(t, figsize=(10, 4), dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(t.data)
    plt.xlim(0, t.size - 1)
    plt.xlabel("Phase bin index")
    plt.ylabel("Normalised amplitude")
    title = "Template {!r}".format(t.header['kind'])
    plt.title(title)
    plt.grid(linestyle=':')
    plt.tight_layout()
    return fig


class TemplateBank(object):
    """ """
    def __init__(self, templates):
        # TODO: check all widths are identical, etc.
        self._templates = templates

    @property
    def templates(self):
        return self._templates

    @property
    def nbins(self):
        raise NotImplementedError

    @classmethod
    def boxcars(cls, n, widths):
        templates = [
            Template.boxcar(n, w) 
            for w in sorted(widths)
            ]
        return cls(templates)

    def _conv_input(self):
        return np.asarray([
            t._conv_input() for t in self.templates
            ])


if __name__ == '__main__':
    t = Template.boxcar(1024, 5)
    tbank = TemplateBank.boxcars(256, range(1, 32))
    for t in tbank.templates:
        print(t)
