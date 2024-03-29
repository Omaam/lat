import numpy as np


def set_module(module):
    """Decorator for overriding __module__ on a function or class.
    Example usage::
        @set_module('numpy')
        def example():
            pass
        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


def lcsplit(lcdata, dt, min_points=1, min_gap=None):
    """Generated splitted light curves.

    Yields:
        np.ndarray: splitted light curves, which shape is not
        the same each other.
    """

    if min_gap is None:
        min_gap = dt

    # search for irregular interval
    diff = np.round(np.diff(lcdata[:, 0]), 6)
    # idxs_split = np.array(np.where(diff != dt))
    idxs_split = np.array(np.where(diff > min_gap)) + 1
    # print(f'Number of segment: {idxs_split.shape[1]}')

    # add 0 and -1 to idxs_split
    idxs_split = np.insert(idxs_split, 0, 0)
    idxs_split = np.append(idxs_split, len(lcdata)-1)

    # generate split lightcurves
    for k in range(len(idxs_split)-1):
        lcdata_splited = lcdata[idxs_split[k]:idxs_split[k+1]]
        if min_points <= len(lcdata_splited):
            yield lcdata[idxs_split[k]:idxs_split[k+1]]


def moving_average(a, width):
    return np.convolve(a, np.ones(width), 'valid') / width
