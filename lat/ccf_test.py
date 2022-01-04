import unittest
import numpy as np

import ccf


class TestCCF(unittest.TestCase):

    def test_divide_into_segment(self):
        a = np.arange(50)
        nperseg = 10
        noverlap = 5
        returned = ccf.divide_into_segment(a, nperseg, noverlap)
        print(returned.shape)
        self.assertEqual(returned.shape, (9, 10))


if __name__ == "__main__":
    unittest.main()
