import unittest
import numpy as np

import correlation as corr


def _detect_lag_at_maxcorrelation(lags, r):
    idx_rmax = np.argmax(r)
    lag_rmax = lags[idx_rmax]
    return lag_rmax


def _create_curve_with_normaldist(n_sample: int):
    return np.random.normal(0, 1, n_sample)


def _create_lagging_paircurve(n_sample: int, n_lag: int):

    a = _create_curve_with_normaldist(100)

    n_lag_abs = abs(n_lag)
    if n_lag >= 0:
        x = a[n_lag_abs:n_lag_abs+n_sample]
        y = a[:n_sample]
    else:
        x = a[:n_sample]
        y = a[n_lag_abs:n_lag_abs+n_sample]

    return x, y


def _assert_lag_detection(n_lag):

    x, y = _create_lagging_paircurve(
        n_sample=100, n_lag=n_lag)
    lags, r = corr.ccf(
        x, y, fs=1, maxlags=None)

    is_correct = \
        n_lag == _detect_lag_at_maxcorrelation(lags, r)

    return is_correct


class TestCCF(unittest.TestCase):

    def test_correlation_function(self):
        # Test for lag detection. You must specify
        # a lag within absolute value of 10.
        self.assertTrue(_assert_lag_detection(+5))
        self.assertTrue(_assert_lag_detection(-3))
        self.assertTrue(_assert_lag_detection(0))

    def test_acf(self):
        x = _create_curve_with_normaldist(100)
        lags, r = corr.acf(x)
        self.assertEqual(
            _detect_lag_at_maxcorrelation(lags, r), 0)

    def test_ccf(self):
        x = _create_curve_with_normaldist(100)
        x, y = _create_lagging_paircurve(100, 5)
        lags, r = corr.ccf(x, y)
        self.assertEqual(
            _detect_lag_at_maxcorrelation(lags, r), 5)


if __name__ == "__main__":
    unittest.main()
