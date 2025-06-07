import unittest
import numpy as np

from spyden import snratio, Template, TemplateBank


class TestSnratio(unittest.TestCase):
    """ """
    def test_single(self):
        """ Test one profile vs one template """
        size = 100 # need an even number

        # True start bin and width of pulse
        # make sure that i0 + w0 < size
        i0 = 42
        w0 = 5 # need an odd number to line up well with the peak of a Gaussian

        noise = np.zeros(size)
        noise[::2] = 1.0
        nstd = 0.5 # if size is even

        data = noise.copy()
        data[i0:i0+w0] = 10.0

        tbox = Template.boxcar(w0)
        snr, mu, sigma, models = snratio(data, tbox, sigma=nstd)
        self.assertEqual(snr.ndim, 3)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(sigma.ndim, 1)
        self.assertEqual(models.ndim, 2)

        iprof, itemp, ibin = np.unravel_index(snr.argmax(), snr.shape)
        self.assertEqual(snr.dtype, np.float32)
        self.assertEqual(ibin, i0)

        tgauss = Template.gaussian(w0)
        snr, mu, sigma, models = snratio(data, tgauss, sigma=nstd)
        self.assertEqual(snr.ndim, 3)
        self.assertEqual(models[0].argmax(), i0 + w0 // 2)

        iprof, itemp, ibin = np.unravel_index(snr.argmax(), snr.shape)
        self.assertEqual(ibin, i0 + w0 // 2)

    def test_multi(self):
        """ Test multiple profiles vs a template bank """
        nprof = 5
        ntemp = 7
        size = 100 # need an even number

        # Profile with the signal
        i0 = nprof // 2

        # True start bin and width of pulse
        # make sure that i0 + w0 < size
        j0 = 42
        w0 = 5 # need an odd number to line up well with the peak of a Gaussian

        noise = np.zeros(shape=(nprof, size))
        noise[:, ::2] = 1.0
        nstd = 0.5 # if size is even

        # Add signal at line i0 only
        data = noise.copy()
        data[i0, j0:j0+w0] = 10.0

        bank = TemplateBank.boxcars(range(1, ntemp + 1))
        snr, mu, sigma, models = snratio(data, bank, sigma=nstd)
        self.assertEqual(snr.dtype, np.float32)
        self.assertEqual(snr.ndim, 3)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(sigma.ndim, 1)
        self.assertEqual(models.ndim, 2)

        iprof, itemp, ibin = np.unravel_index(snr.argmax(), snr.shape)
        self.assertEqual(iprof, i0)
        self.assertEqual(itemp, w0 - 1)
        self.assertEqual(ibin, j0)

        # again, with single gaussian template
        tgauss = Template.gaussian(w0)
        snr, mu, sigma, models = snratio(data, tgauss, sigma=nstd)
        self.assertEqual(snr.ndim, 3)
        self.assertEqual(mu.ndim, 1)
        self.assertEqual(sigma.ndim, 1)
        self.assertEqual(models.ndim, 2)

        iprof, itemp, ibin = np.unravel_index(snr.argmax(), snr.shape)
        self.assertEqual(iprof, i0)
        self.assertEqual(itemp, 0)
        self.assertEqual(ibin, j0 + w0 // 2)