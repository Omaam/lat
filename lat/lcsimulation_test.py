"""Test for lcsimulation.
"""
import unittest

from absl import logging
import numpy as np

from lcsimulation import LCSimulation
import lcsimulation


def _get_lightcurves(num_sample, lag, dt):
    lcsim = LCSimulation(lag, dt)
    lc1, lc2 = lcsim.sample_lightcurves(num_sample)
    return lc1, lc2


class LCSimulationTest(unittest.TestCase):

    logging.set_verbosity(logging.DEBUG)

    def testSampleDecimalDT(self):

        lc1, lc2 = _get_lightcurves(num_sample=100, lag=5, dt=0.1)

        returned = lc1[:, 0]
        expected = np.arange(0, 9.9+0.1, 0.1)
        self.assertTrue(np.allclose(returned, expected))

    def testNegativeLag(self):
        pass

    def testShapeOfBlendTwoCurves(self):

        design_matrix = [[0.9, 0.1], [0.8, 0.2]]

        num_sample = 100
        time_start = 0
        lag = 5
        dt = 1

        lcsim = LCSimulation(lag, dt)
        lcsim.sample_lightcurves(num_sample, time_start,
                                 time_scale=5)
        time, cts_matrix = lcsim.blend_lightcurves(design_matrix)

        expected = (2, num_sample)
        returned = cts_matrix.shape
        self.assertTrue(expected, returned)

    def testOUProcessCreation(self):
        cts = lcsimulation._sample_ouprocess(
            100, 5, 1, random_state=1)

        import matplotlib.pyplot as plt
        plt.plot(cts)
        plt.show()


if __name__ == "__main__":
    unittest.main()
