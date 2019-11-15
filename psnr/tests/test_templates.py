import unittest
import numpy as np
import matplotlib.pyplot as plt

from psnr import Template, TemplateBank
from psnr.template import normalise


class TestTemplate(unittest.TestCase):
    """ """
    def test_normalise(self):
        x = np.arange(10)
        z = normalise(x)
        m = z.mean()
        sqsum = (z**2).sum()
        self.assertAlmostEqual(m, 0.0)
        self.assertAlmostEqual(sqsum, 1.0)

    def test_init(self):
        size = 5
        data = np.arange(size)
        pad_value = -0.5
        refbin = 1
        reference = 'secondbin'
        kind = 'custom'
        shape_params={'hello': 1}

        t = Template(
            data, 
            pad_value=pad_value, 
            refbin=refbin, 
            reference=reference, 
            kind=kind, 
            shape_params=shape_params
            )

        self.assertEqual(size, t.size)
        self.assertTrue(np.allclose(data, t.data))
        self.assertEqual(pad_value, t.pad_value)
        self.assertEqual(refbin, t.refbin)
        self.assertEqual(reference, t.reference)
        self.assertEqual(kind, t.kind)
        self.assertEqual(shape_params, t.shape_params)

    def test_prepared_data(self):
        size = 10
        size_padded = 16
        data = np.zeros(size)
        refbin = 3
        data[refbin] = 2.0
        data[refbin+1] = 1.0

        t = Template(data, refbin=refbin)
        prep = t.prepared_data(size_padded)

        ### size
        self.assertEqual(prep.size, size_padded)

        ## dtype
        self.assertEqual(prep.dtype, np.float32)

        ### normalisation
        self.assertAlmostEqual(prep.mean(), 0.0)
        self.assertAlmostEqual((prep ** 2).sum(), 1.0)

        ### alignment and time-reversal
        # indices of largest and second-largest value
        # Before preparation, this should be (refbin, refbin+1)
        # After rolling: (0, 1)
        # After reversal: (0, size_padded - 1)
        m0, m1 = prep.argsort()[::-1][:2] 
        self.assertEqual(m0, 0)
        self.assertEqual(m1, size_padded - 1)

    def test_boxcar(self):
        w = 5
        t = Template.boxcar(w)
        self.assertEqual(t.size, w)
        self.assertEqual(t.refbin, 0)
        self.assertEqual(t.pad_value, 0.0)
        self.assertTrue(np.allclose(t.data, 1.0))

    def test_gaussian(self):
        w = 5.0
        t = Template.gaussian(w)
        self.assertEqual(t.refbin, t.size//2)
        self.assertEqual(t.pad_value, 0.0)

    def test_plot(self):
        t = Template.boxcar(5)
        fig = t.plot()
        plt.close(fig)
    

class TestTemplateBank(unittest.TestCase):
    """ """
    def test_boxcars(self):
        bank = TemplateBank.boxcars(range(1, 5))

    def test_gaussians(self):
        bank = TemplateBank.gaussians(range(1, 5))