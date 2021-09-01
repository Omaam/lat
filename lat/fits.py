import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


def make_gti(time, dt=1, min_points=1):
    # search for irregular interval
    diff = np.round(np.diff(time), 6)
    idxs_split = np.array(np.where(diff != dt)) + 1

    # add 0 and -1 to idxs_split
    idxs_split = np.insert(idxs_split, 0, 0)
    idxs_split = np.append(idxs_split, len(time))

    # generate split lightcurves
    gtis = []
    for k in range(len(idxs_split)-1):
        time_chunk = time[idxs_split[k]:idxs_split[k+1]]
        if len(time_chunk) >= min_points:
            gtis.append([time_chunk[0], time_chunk[-1]])

    return np.array(gtis)
