"""Test for lcsimulation.
"""
import unittest

from absl import logging

from lcsimulation import LCSimulation


class LCSimulationTest(unittest.TestCase):

    def test_lcmatrix_creation(self):

        simulator = LCSimulation()

        simulator.add_oucurve(ou_id=0, lengthscale=5)
        simulator.add_oucurve(ou_id=1, lengthscale=10)
        simulator.add_oucurve(ou_id=2, lengthscale=1)

        simulator.extract_curve(ou_id=0, lag=0)
        simulator.extract_curve(ou_id=0, lag=3)
        simulator.extract_curve(ou_id=1, lag=0)
        simulator.extract_curve(ou_id=1, lag=-2)
        simulator.extract_curve(ou_id=1, lag=4)
        simulator.extract_curve(ou_id=2, lag=4)
        simulator.extract_curve(ou_id=2, lag=5)
        lcmatrix = simulator.sample(100)

        returned = lcmatrix.shape
        expected = (100, 7)

        self.assertEqual(returned, expected)

    def test_blend_curves(self):
        simulator = LCSimulation()

        simulator.add_oucurve(ou_id=0, lengthscale=5)

        simulator.extract_curve(ou_id=0, lag=0)
        simulator.extract_curve(ou_id=0, lag=5)
        simulator.extract_curve(ou_id=0, lag=8)

        simulator.sample(100)
        design_vector = [0.5, 0.3, 0.2]
        lcmatrix_blended = simulator.blend_curves([0, 1, 2], design_vector)

        expected = (100, 1)
        self.assertEqual(lcmatrix_blended.shape, expected)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    unittest.main()
