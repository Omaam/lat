import numpy as np

__all__ = ["divide_into_segment", "lcsplit"]


def divide_into_segment(x, nperseg, noverlap):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    divided_ary = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                  strides=strides)
    return divided_ary


def lcsplit(lcdata, dt, min_points=1, min_gap=None):
    """
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
