import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy import signal


def lcsplit(lcdata1, lcdata2, dt, maxlags=None):
    '''
    '''
    # search for irregular interval
    diff = np.diff(lcdata1[:, 0])
    idxs_split = np.array(np.where(diff != dt)) + 1
    print(f'Number of segment: {len(idxs_split)}')

    # add 0 and -1 to idxs_split
    idxs_split = np.insert(idxs_split, 0, 0)
    idxs_split = np.append(idxs_split, len(lcdata1)-1)

    # generate split lightcurves
    for k in range(len(idxs_split)-1):
        yield lcdata1[idxs_split[k]:idxs_split[k+1]], \
              lcdata2[idxs_split[k]:idxs_split[k+1]]


# Cross-correlation function
def ccf(x, y, fs=1, maxlags=None):
    # standardized
    x, y = zscore(x), zscore(y)

    # calcurate correlation and lags
    n_x, n_y = len(x), len(y)
    T = max(n_x, n_y)
    r = signal.correlate(y, x, mode='full') / np.std(x) / np.std(y) / T
    # r = signal.correlate(y, x, mode='full') / T
    lags = np.arange(-n_x + 1, n_y) / fs

    # query
    maxlags = 2 * T - 1 if maxlags is None else maxlags
    lag_out = lags[((-maxlags <= lags) & (lags <= maxlags))]
    r_out = r[((-maxlags <= lags) & (lags <= maxlags))]
    return lag_out, r_out


def stackccf(lcdata1, lcdata2, dt, maxlags, output='ave'):

    # ccf calulation
    r_tile = []
    for l1, l2 in lcsplit(lcdata1, lcdata2, dt, maxlags):
        lag, r = ccf(l1[:, 1], l2[:, 1], 1, 10)
        r_tile.append(r)
    r_tile = np.array(r_tile)

    if output == 'ave':
        r_ave = np.average(r_tile, 0)
        return lag, r_ave
    elif output == 'tile':
        return lag, r_tile


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
