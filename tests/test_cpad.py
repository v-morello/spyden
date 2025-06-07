import unittest
import numpy as np

from spyden.cpad import cpadpow2


class TestCpadpow2(unittest.TestCase):
    """ """
    def test_1d(self):
        cols = 17
        colspow2 = 32

        # Input
        x = np.arange(cols, dtype=float)

        # Expected result
        y = np.arange(colspow2, dtype=float)
        y[cols:] = np.arange(colspow2 - cols, dtype=float)

        xp = cpadpow2(x)
        self.assertEqual(xp.shape, y.shape)
        self.assertTrue(np.allclose(xp, y))

    def test_2d(self):
        lines, cols = (4, 13)
        colspow2 = 16

        # Input
        x = np.zeros((lines, cols), dtype=float)
        x[:, :cols] = np.arange(cols)

        # Expected result
        y = np.zeros((lines, colspow2))
        y[:, :cols] = x
        y[:, cols:] = np.arange(colspow2 - cols)

        xp = cpadpow2(x)
        self.assertEqual(xp.shape, y.shape)
        self.assertTrue(np.allclose(xp, y))
