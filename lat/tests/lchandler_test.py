"""Test for lchandler.py.
"""
import numpy as np
import unittest

import lat.lchandler as lchandler


class TestSplit(unittest.TestCase):
    """
    """

    def test_case1(self):
        dt = 0.01

        time = np.arange(0, 100, dt)
        flux = np.random.normal(0, 1, len(time))
        error = np.random.normal(0, 0.01, len(time))
        lcdata = np.array([time, flux, error]).T

        del_ids = [100, 200, 300]
        lcdata = np.delete(lcdata, del_ids, axis=0)

        lcs = list(lchandler.lcsplit(lcdata, dt, min_points=1))
        expected = 4
        returned = len(lcs)
        self.assertEqual(expected, returned)

    def test_case2(self):
        dt = 0.01

        time = np.arange(0, 100, dt)
        flux = np.random.normal(0, 1, len(time))
        error = np.random.normal(0, 0.01, len(time))
        lcdata = np.array([time, flux, error]).T

        del_ids = [100, 200, 300, 304]
        lcdata = np.delete(lcdata, del_ids, axis=0)

        lcs = list(lchandler.lcsplit(lcdata, dt, min_points=3))
        expected = 5
        returned = len(lcs)
        self.assertEqual(expected, returned)

    def test_case3(self):
        dt = 0.01

        time = np.arange(0, 100, dt)
        flux = np.random.normal(0, 1, len(time))
        error = np.random.normal(0, 0.01, len(time))
        lcdata = np.array([time, flux, error]).T

        del_ids = [100, 200, 300, 304]
        lcdata = np.delete(lcdata, del_ids, axis=0)

        lcs = list(lchandler.lcsplit(lcdata, dt, min_points=4))
        expected = 4
        returned = len(lcs)
        self.assertEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
