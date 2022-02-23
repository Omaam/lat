"""Lag simulation class.
"""
from absl import logging
import GPy as gpy
import numpy as np
import pandas as pd


def _add_ouprocess_to_df(num_sample_lc: int,
                         df_ou: pd.DataFrame,
                         df_lc: pd.DataFrame):
    """Add OU process samples to pandas.DataFrame.
    """

    df_ou = _parse_df_lc(df_ou, df_lc)

    curves = []
    for ou_info in df_ou.itertuples():

        interval_min_max = ou_info.max_lag - ou_info.min_lag
        num_sample_ou = num_sample_lc + interval_min_max
        curve = _sample_ouprocess(num_sample_ou,
                                  ou_info.variance,
                                  ou_info.lengthscale,
                                  ou_info.random_state)
        curves.append(curve)

    df_ou["curve"] = curves
    logging.debug("df_ou:\n%s", df_ou)

    return df_ou


def _create_lcmatrix(num_sample_lc: int, df_ou: pd.DataFrame,
                     df_lc: pd.DataFrame):
    """Create curve matrix from blueprint for light curves.
    """

    df_lc = df_lc.sort_values("lc_id", ignore_index=True)
    logging.debug("df_lc:\n%s", df_lc)

    lc_matrix = []
    for lcinfo in df_lc.values:

        lc_id, ou_id, lag = lcinfo
        ou_info = df_ou.iloc[ou_id]

        shift = -lag + ou_info.max_lag
        lc = ou_info.curve[shift:shift+num_sample_lc]
        lc_matrix.append(lc)

    lc_matrix = np.array(lc_matrix).T
    return lc_matrix


def _sample_ouprocess(num_sample: int,
                      variance: float = 1,
                      lengthscale: float = None,
                      random_state=None):

    x = np.arange(0, num_sample, 1)
    x = x[:, np.newaxis]

    kernel = gpy.kern.Exponential(input_dim=1,
                                  variance=variance,
                                  lengthscale=lengthscale)
    mean = np.zeros(num_sample)
    cov = kernel.K(x, x)

    # Make random_state be int or None.
    if np.isnan(random_state):
        random_state = None

    np.random.seed(random_state)
    y = np.random.multivariate_normal(mean, cov)

    return y


def _parse_df_lc(df_ou, df_lc):

    df_ou = df_ou.copy()
    df_ou["min_lag"] = df_lc.groupby("ou_id").min().lag
    df_ou["max_lag"] = df_lc.groupby("ou_id").max().lag

    # Make min_lag be lesser than zero.
    df_ou["min_lag"] = np.where(df_ou["min_lag"] > 0,
                                0, df_ou["min_lag"])
    df_ou["max_lag"] = np.where(df_ou["max_lag"] < 0,
                                0, df_ou["max_lag"])

    logging.debug("df_ou:\n%s", df_ou)

    return df_ou


def blend_curves(lc_matrix: list,
                 design_matrix: list):
    """Blend two light cruves.

    Args:
        * target_lc_ids:
        * design_matrix:
            [(Number of obsevation) x 2]
            Matrix of ratios used when blending light curves.
    Return:
        * time: Time for blended light cruves.
        * lcs_blended: Blended light curves.
    """
    lc_matrix = np.array(lc_matrix)
    design_matrix = np.array(design_matrix)

    blended_lc = np.dot(lc_matrix, design_matrix)

    return blended_lc


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

    def __init__(self):
        self.lc_counter = 0
        self.df_lc = pd.DataFrame(columns=["lc_id", "ou_id", "lag"])
        self.df_ou = pd.DataFrame(
            columns=["ou_id", "variance", "lengthscale",
                     "random_state"])

        self.lcmatrix = None

    def acquire_time(self, num_sample_lc: int,
                     time_start: float = 0.0, dt: float = 0.0):
        pass

    def add_oucurve(self, ou_id,
                    variance: float = 1.0,
                    lengthscale: float = None,
                    random_state: int = None):
        """Add OU process information to dataframe.
        """
        ou_info = {
            "ou_id": ou_id,
            "variance": variance,
            "lengthscale": lengthscale,
            "random_state": random_state
        }
        self.df_ou = self.df_ou.append(ou_info, ignore_index=True)

    def add_error_gaussian(self, target_lc_indices: list,
                           mean: float = 0.0, variance: float = 0.1):
        """ Add error information.
        """
        self.error_list.append([mean, variance])

    def blend_curves(self, ids_target_curve: list, design_vector: list):
        """Blend curves using pre-sampled curves.
        """
        if self.lcmatrix is None:
            raise AttributeError(
                "self.lcmatrix doesn't exist. You must run "
                "'sample' method beforehand."
            )

        design_vector = np.array(design_vector)
        design_vector = design_vector[:, np.newaxis]

        lcmatrix_target = self.lcmatrix[:, ids_target_curve]
        lcmatrix_blended = np.dot(lcmatrix_target, design_vector)

        return lcmatrix_blended

    def extract_curve(self, ou_id: int, lag: int):

        lc_info = {"lc_id": self.lc_counter,
                   "ou_id": ou_id,
                   "lag":   lag}
        self.df_lc = self.df_lc.append(lc_info, ignore_index=True)
        self.lc_counter += 1

    def sample(self, num_sample_lc: int):
        """Sample curves using added information.
        """
        if len(self.df_lc) == 0:
            raise ValueError(
                "You must add curve(s) before sampling.")

        self.df_ou = _add_ouprocess_to_df(num_sample_lc,
                                          self.df_ou,
                                          self.df_lc)
        lcmatrix = _create_lcmatrix(num_sample_lc, self.df_ou,
                                    self.df_lc)
        self.lcmatrix = lcmatrix

        return lcmatrix
