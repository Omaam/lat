"""There are some kind of CCF.

TODO:
    * Make function that is able to calcurate CCF, even if
      light curves are unevened.(2021-12-20 17:16:50)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from lat.basic import lcsplit
import lat.lchandler as lchandler


__all__ = ["ccf", "dcf", "sccf", "sym_sccf",
           "subtract_oneside", "ccf_error"]


# Cross-correlation function
def ccf(x, y, fs=1, maxlags=None):

    # calcurate correlation and lags
    n_x, n_y = len(x), len(y)
    T = max(n_x, n_y)
    x_cent = x - np.mean(x)
    y_cent = y - np.mean(y)
    r = signal.correlate(y_cent, x_cent, mode='full')
    r = r / np.std(x) / np.std(y) / T
    lags = np.arange(-n_x + 1, n_y) / fs

    # query
    maxlags = 2 * T - 1 if maxlags is None else maxlags
    lag_out = lags[((-maxlags <= lags) & (lags <= maxlags))]
    r_out = r[((-maxlags <= lags) & (lags <= maxlags))]

    return lag_out, r_out


def dcf(X1, X2, dt=1, maxlags=20):

    # calcurate basic values
    f1 = X1[:, 1] - np.mean(X1[:, 1])
    f2 = X2[:, 1] - np.mean(X2[:, 1])
    sg1 = np.std(X1[:, 1])
    sg2 = np.std(X2[:, 1])
    er1 = np.mean(X1[:, 2])
    er2 = np.mean(X2[:, 2])
    div = np.sqrt((sg1**2-er1**2)*(sg2**2-er2**2))

    # time difference matrix
    tt2, tt1 = np.meshgrid(X2[:, 0], X1[:, 0])
    tdelta_mat = tt2 - tt1

    # calcurate udcf
    f1 = np.reshape(f1, (len(f1), 1))
    f2 = np.reshape(f2, (1, len(f2)))
    udcf = np.dot(f1, f2)/div

    # calcurate correlations
    lags = np.arange(-maxlags-dt, maxlags+dt+dt, dt)
    corrs = np.zeros(len(lags)-1)
    errs = np.zeros(len(lags)-1)
    # ns = np.zeros(len(lags)-1)
    place = np.array([np.searchsorted(lags, tdelta)
                      for tdelta in tdelta_mat])
    for i in range(len(lags)-1):
        bools = np.where(place == i)
        m = len(udcf[bools])
        corrs[i] = np.sum(udcf[bools])/m
        errs[i] = np.sqrt(np.sum((udcf[bools] - corrs[i])**2))/(m-1)

    # delete head and tail because these are
    # out of lag range
    lags, corrs, errs = lags[1:-1], corrs[1:-1], errs[1:-1]

    return lags, corrs, errs


def sccf(lcdata1, lcdata2, dt, maxlags,
         min_gap=0, min_points=1, output='ave'):

    # split lightcurves
    lcs1 = lcsplit(lcdata1, dt, min_gap, min_points)
    lcs2 = lcsplit(lcdata2, dt, min_gap, min_points)

    # ccf calulation
    r_tile = []
    for i, (l1, l2) in enumerate(zip(lcs1, lcs2)):
        if len(l1) > maxlags/dt:
            lag, r = ccf(l1[:, 1], l2[:, 1], 1/dt, maxlags)
            r_tile.append(r)
    r_tile = np.array(r_tile)

    # output arrangemt
    if output == 'ave':
        r_ave = np.average(r_tile, 0)
        return lag, r_ave
    elif output == 'tile':
        return lag, r_tile


def sym_sccf(lcdata1, lcdata2, dt, maxlags,
             min_gap=0, min_points=1,
             base_side='left', output='ave'):

    # split lightcurves
    lcs1 = lcsplit(lcdata1, dt, min_gap, min_points)
    lcs2 = lcsplit(lcdata2, dt, min_gap, min_points)

    # ccf calulation
    r_tile = []
    for i, (l1, l2) in enumerate(zip(lcs1, lcs2)):
        if len(l1) > maxlags/dt:
            lag, r = ccf(l1[:, 1], l2[:, 1], 1/dt, maxlags)
            r = symccf(r, base_side)
            r_tile.append(r)
    r_tile = np.array(r_tile)

    # output arrangemt
    if output == 'ave':
        r_ave = np.average(r_tile, 0)
        return lag, r_ave
    elif output == 'tile':
        return lag, r_tile


def symccf(a, base_side='left'):

    # make array
    corr_sym = np.copy(a) if base_side == 'left' else np.copy(a[::-1])

    # get center index + 1
    idx_med_m2 = int(np.floor(len(corr_sym)/2)) - 1
    idx_med_p1 = int(np.ceil(len(corr_sym)/2))

    # substitute
    corr_sym[idx_med_p1:] = corr_sym[idx_med_m2::-1]

    # substraction
    corr_rest = a - corr_sym

    return corr_rest


def subtract_oneside(a, base_side):

    # make array
    corr_sym = np.copy(a) if base_side == 'left' else np.copy(a[::-1])

    # get center index + 1
    idx_med_m2 = int(np.floor(len(corr_sym)/2)) - 1
    idx_med_p1 = int(np.ceil(len(corr_sym)/2))

    # substitute
    corr_sym[idx_med_p1:] = corr_sym[idx_med_m2::-1]

    # substraction
    corr_rest = a - corr_sym

    return corr_rest


def ccf_error(x, y, fs=1.0, nperseg=256, noverlap=None,
              maxlags=None):
    """Calcurate CCF with error.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if maxlags is None:
        maxlags = x.size/fs/2  # time unit

    if noverlap is None:
        noverlap = nperseg//2

    x_stack = lchandler.divide_into_segment(x, nperseg, noverlap)
    y_stack = lchandler.divide_into_segment(y, nperseg, noverlap)

    cc_mat = []
    for (x_seg, y_seg) in zip(x_stack, y_stack):
        lags, cc = ccf(x_seg, y_seg, fs, maxlags)
        cc_mat.append(cc)
    cc_mat = np.asarray(cc_mat)

    return lags, cc_mat


def main():

    np.random.seed(20210612)

    npoints = 10000
    lag = 5

    # lightcurve creation
    time = np.arange(npoints)
    flux = np.random.normal(0, 1, npoints+lag)
    lcdata1 = np.array([time, flux[lag:npoints+lag]]).T
    lcdata2 = np.array([time, flux[0:npoints]]).T

    # test
    lags, cc_mat = ccf_error(lcdata1[:, 1], lcdata2[:, 1],
                             nperseg=256, noverlap=128,
                             maxlags=20)
    cc_05, cc_50, cc_95 = np.quantile(cc_mat, [0.05, 0.50, 0.95], axis=0)

    plt.plot(lags, cc_50)
    plt.fill_between(lags, cc_05, cc_95, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
