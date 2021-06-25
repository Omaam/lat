import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def lcsplit(lcdata, dt, min_gap=0, min_points=1):
    '''
    '''
    # search for irregular interval
    diff = np.round(np.diff(lcdata[:, 0]), 6)
    idxs_split = np.array(np.where(diff != dt)) + 1
    # print(f'Number of segment: {idxs_split.shape[1]}')

    # add 0 and -1 to idxs_split
    idxs_split = np.insert(idxs_split, 0, 0)
    idxs_split = np.append(idxs_split, len(lcdata)-1)

    # generate split lightcurves
    for k in range(len(idxs_split)-1):
        yield lcdata[idxs_split[k]:idxs_split[k+1]]


# Cross-correlation function
def ccf(x, y, fs=1, maxlags=None):

    # calcurate correlation and lags
    n_x, n_y = len(x), len(y)
    r = signal.correlate(y, x, mode='full')
    r = r / (np.sqrt(n_x)*np.std(x)) / (np.sqrt(n_y)*np.std(y))
    lags = np.arange(-n_x + 1, n_y) / fs

    # query
    T = max(n_x, n_y)
    maxlags = 2 * T - 1 if maxlags is None else maxlags
    lag_out = lags[((-maxlags <= lags) & (lags <= maxlags))]
    r_out = r[((-maxlags <= lags) & (lags <= maxlags))]

    return lag_out, r_out


def stackccf(lcdata1, lcdata2, dt, maxlags, output='ave'):

    # ccf calulation
    r_tile = []
    for i, (l1, l2) in enumerate(lcsplit(lcdata1, lcdata2, dt, maxlags)):
        if len(l1) > maxlags/dt:
            lag, r = ccf(l1[:, 1], l2[:, 1], 1/dt, maxlags)
            r_tile.append(r)
    r_tile = np.array(r_tile)
    if (r_tile.shape[0]-1) / i < 0.5:
        print(f'!!! Accept rate is below 50 %: {r_tile.shape[0]-1} / {i}')
        # print(f'number of accepted segment: {r_tile.shape[0]-1} / {i}')

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

    # arrangement
    out = corr_rest if base_side == 'left' else corr_rest[::-1]

    return out


def main():

    # lcdata generatta
    np.random.seed(20210609)
    time = np.arange(0, 30)
    time = np.append(time, np.arange(60, 110))
    time = np.append(time, np.arange(120, 140))
    fluxbase = np.random.normal(0, 1, 200)
    flux1 = fluxbase[5:105]
    flux2 = fluxbase[0:100]

    lcdata1 = np.array([time, flux1]).T
    lcdata2 = np.array([time, flux2]).T

    # lcsplit
    lag, rs = stackccf(lcdata1, lcdata2, 1, 10, 'tile')
    for r in rs:
        plt.plot(lag, r)
    plt.show()


if __name__ == '__main__':
    main()
