"""Lag simulation class.
"""
from absl import logging
import GPy as gpy
import numpy as np
import pandas as pd


def _calculate_convfunc():
    pass


def _convolute():
    pass


def _sample_ouprocess(num_sample: int,
                      variance: float = 1,
                      lengthscale: float = None,
                      random_state=None):

    x = np.arange(0, num_sample, 1)
    x = x[:, np.newaxis]
    logging.debug("x: %s", x.shape)

    kernel = gpy.kern.Exponential(input_dim=1,
                                  variance=variance,
                                  lengthscale=lengthscale)
    mean = np.zeros(num_sample)
    cov = kernel.K(x, x)
    logging.debug("mean: %s", mean.shape)
    logging.debug("cov: %s", cov.shape)

    np.random.seed(random_state)
    y = np.random.multivariate_normal(mean, cov)

    return y


def _create_lagged_curves(num_sample_lc: int,
                          lag: float,
                          time_start: float,
                          dt: float,
                          amplitude_variance: float,
                          time_scale: float,
                          random_state=None):

    abs_lag = abs(lag)

    # Calcurate times and coutns.
    time_stop = dt * (num_sample_lc)
    times = np.arange(time_start, time_stop, dt)
    logging.debug("times: %s", times.shape)

    num_sample_lag = int(np.ceil(abs_lag / dt))
    num_sample_total = num_sample_lc + num_sample_lag

    lenghscale = time_scale / dt if time_scale else None
    cts = _sample_ouprocess(num_sample_total,
                            variance=amplitude_variance,
                            lengthscale=lenghscale)

    # Create lagged curves.
    if num_sample_lag >= 0:
        cts1 = cts[num_sample_lag:num_sample_lc+num_sample_lag]
        cts2 = cts[0:num_sample_lc]
    elif num_sample_lag < 0:
        cts1 = cts[0:num_sample_lc]
        cts2 = cts[num_sample_lag:num_sample_lc+num_sample_lag]

    lc1 = np.vstack([times, cts1]).T
    lc2 = np.vstack([times, cts2]).T
    logging.debug("lc1: %s", lc1.shape)
    logging.debug("lc2: %s", lc2.shape)

    return lc1, lc2


class LCSimulation():
    """LC simulation light curve.

    Note:
        * This class does NOT give time information. Thus, you must
          give parameters in point unit.
          For example, time lag of 5 means 5 points. if you specify
          dt as 0.1 s, then lag would be 0.5 s.
          If you want to get time information, you can use
          "acquire_time" function.

    TODO:
        * (Ommam) Make it possible to specify a number of independent
          light curve used in blending light curves.
    """

    def __init__(self, lag: float, dt: float):

        self.dt = dt
        self.lag = lag

        self.time = None
        self.lc1 = None
        self.lc2 = None

        self.lc_counter = 0
        self.lag_abs_max = None
        self.lc_blueprint = pd.DataFrame()
        self.ou_blueprint = pd.DataFrame()
        self.error_list = []

    def acquire_time(self, time_start: float = 0.0, dt: float = 0.0):
        pass

    def add_curve_lagging(self, ou_id: int, lag: int):

        if self.lag_abs_max < abs(lag):
            self.lag_abs_max = abs(lag)

        self.lc_counter += 1
        lc_info = {"lc_id": self.lc_counter,
                   "ou_id": ou_id,
                   "lag":   0}
        self.lc_blueprint.append(lc_info, index_ignore=True)

    def add_curve_ouprocess(self, variance: float, lengthscale: int):
        """Add OU process infomation.
        """

        ou_id = len(self.ou_info)

        ou_info = {
            "ou_id": ou_id,
            "variance": variance,
            "lengthscale": lengthscale
        }
        self.ou_blueprint.append(ou_info, index_ignore=True)

        self.lc_counter += 1
        lc_info = {"lc_id": self.lc_counter,
                   "ou_id": ou_id,
                   "lag":   0}
        self.lc_blueprint.append(lc_info, ignore_index=True)

    def add_error_gaussian(self, target_lc_indices: list,
                           mean: float = 0.0, variance: float = 0.1):
        """ Add error information.
        """
        self.error_list.append([mean, variance])

    def blend_lightcurves(self, design_matrix: np.array):
        """Blend two light cruves.

        Args:
            * design_matrix:
                [(Number of obsevation) x 2]
                Matrix of ratios used when blending light curves.
        Return:
            * time: Time for blended light cruves.
            * lcs_blended: Blended light curves.
        """
        # TODO: (Omama) Adapt this to new API.
        if (self.lc1 is None) & (self.lc2 is None):
            raise AttributeError(
                "You must sample curves before blending.")

        design_matrix = np.array(design_matrix)

        cts_matrix_original = np.array([self.lc1[:, 1], self.lc2[:, 1]])
        cts_matrix_blended = np.dot(design_matrix, cts_matrix_original)

        return self.time, cts_matrix_blended

    def sample(self, num_point: int, random_state: int = None):
        """Sample curves using added information.
        """

        if self.lc_blueprint == []:
            raise ValueError(
                "You must add curve(s) before sampling.")

        # Sample OU process.
        num_point_total = num_point + self.lag_abs_max
        ys = []
        for i, ou_info in enumerate(self.ou_blueprint):

            # To avoid sampling the same curve.
            if random_state is not None:
                random_state = random_state + i

            y = _sample_ouprocess(num_point_total,
                                  ou_info["variance"],
                                  ou_info["lengthscale"],
                                  random_state=random_state)
            ys.append(y)
        self.ou_blueprint["curve"] = ys

        # Convolute to get lagging curve.
        # TODO: (Omama) Complete below.
        lags_list = self.lc_blueprint["lag"]
        conv_funcs = _calculate_convfunc(num_point_total, lags_list)

        lc_list = []
        for ou in self.ou_blueprintp["curve"]:
            for conv_func in conv_funcs:
                lc = _convolute(ou, conv_func)
                lc_list.append(lc)

        return lc_list

    # TODO: (Omama) Following function is old one. Check dependencies
    #       and delete.
    def sample_curves(self, num_sample: int,
                      time_start: float = 0.0,
                      amplitude_variance: float = 1.0,
                      time_scale: float = None,
                      random_state: int = None):
        """Sample lagged light curves.
        """

        lc1, lc2 = _create_lagged_curves(
            num_sample_lc=num_sample,
            lag=self.lag,
            time_start=time_start,
            dt=self.dt,
            amplitude_variance=amplitude_variance,
            time_scale=time_scale,
            random_state=random_state)

        self.time = lc1[:, 0]
        self.lc1 = lc1
        self.lc2 = lc2

        return lc1, lc2
