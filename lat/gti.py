"""GTI handling functions.
"""
from astropy.io import fits
from astropy.table import Table
import numpy as np
from numpy.typing import ArrayLike
# import pandas as pd
# import matplotlib.pyplot as plt


def apply_gtis(X: np.ndarray, time: ArrayLike, gtis):
    mask = np.zeros(time.shape)
    for gti in gtis:
        m = np.ones(time.shape)
        m = np.where(((gti[0] <= time) & (time <= gti[1])), m, 0)
        mask += m
    mask = mask.astype(bool)
    return X[mask]


def calc_total_time_in_gti(gtis):
    gtis = np.array(gtis)
    total_time = np.sum(gtis[:, 1] - gtis[:, 0])
    return total_time


def calc_exposure(start, stop, gtis):
    gtis = tidy_gti(gtis)
    dura = 0
    for g in gtis:
        dura += min(stop, g[1]) - max(start, g[0])
    return dura


def load_gti_from_event(event_file, idx_gti_hdu=2):
    hdul = fits.open(event_file)
    df = Table.read(hdul[idx_gti_hdu]).to_pandas()
    return df[["START", "STOP"]].values


def make_gti_from_timeseries(time, dt):
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
        if len(time_chunk) > 1:
            gtis.append([time_chunk[0], time_chunk[-1]])

    return np.array(gtis)


def make_hdu_of_gti(hdul, gtis, idx_gti_hdu):
    gtis = np.array(gtis)
    tstart = fits.Column(name="TSTART", array=gtis[:, 0], format='E')
    tstop = fits.Column(name="TSTOP", array=gtis[:, 1], format='E')
    tbl = fits.BinTableHDU.from_columns([tstart, tstop], name="GTI")
    hdul.append(tbl)
    return hdul


def tidy_gti(gtis):
    gtis_out = []
    for g in gtis:
        if len(gtis_out) == 0:
            gtis_out.append(g)
            continue
        for o in gtis_out:
            if o[0] < g[0] and g[1] < o[1]:
                gtis_out.remove(o)
                gtis_out.append(g)
            else:
                if g[1] < o[0]:
                    gtis_out.append(g)
                elif o[0] < g[1]:
                    o[1] = g[1]
                elif g[0] < o[1]:
                    o[0] = g[0]
                elif o[1] < g[1]:
                    gtis_out.append(g)
    return np.array(gtis_out)


def save_gti(gtis, savename, overwrite=False):
    gtis = np.array(gtis)
    tstart = fits.Column(name="TSTART", array=gtis[:, 0], format='E')
    tstop = fits.Column(name="TSTOP", array=gtis[:, 1], format='E')
    tbl = fits.BinTableHDU.from_columns([tstart, tstop], name="GTI")
    tbl.writeto(savename, overwrite=overwrite)


def update_gti(hdul, gtis, idx_gti_hdu=2):
    if len(hdul) >= idx_gti_hdu:
        make_hdu_of_gti(hdul, gtis, idx_gti_hdu)
    else:
        tbl = Table.read(hdul[idx_gti_hdu])
        for gti in gtis:
            tbl.add_row(gti)
        hdul[idx_gti_hdu] = fits.BinTableHDU(tbl)
    return hdul
