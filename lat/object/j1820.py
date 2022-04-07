import os

import numpy as np
import pandas as pd
from astropy.time import Time

__all__ = ["obsid2date"]


def obsid2date(obsids, fmt="mjd"):

    # load database
    path = os.path.dirname(__file__)
    df_obs = pd.read_csv(path+"/maxi_j1820p070.csv")

    # search indicies
    idx = np.searchsorted(df_obs["obsid"], obsids)
    times = np.array(df_obs.loc[idx, "time"])

    # convert
    t = Time(times.astype(str), format="iso")
    tt = t.to_value(fmt)

    return tt
