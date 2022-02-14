"""Test for lcsimulation.
"""
import unittest

from absl import logging

from lcsimulation import LCSimulation


class LCSimulationTest(unittest.TestCase):

    logging.set_verbosity(logging.INFO)

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
        expected = (7, 100)

        self.assertEqual(returned, expected)


if __name__ == "__main__":
    unittest.main()
