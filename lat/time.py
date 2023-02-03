"""Module for handling time.
"""
import numpy as np
import pandas as pd


def convert_time_grego2mjd(y, m, d):
    if m <= 2:
        m += 12
        y -= 1
    y_term = int(365.25*y) + (y//400) - (y//100)
    md_term = int(30.59*(m-2)) + d - 678912
    return y_term+md_term


def convert_time_mjd2grego(mjd):
    """Modified Julian Date -> Gregorian Calendar Date
    """
    n = mjd + 678881
    a = 4*n + 3 + 4*(3*(4*(n+1)//146097+1)//4)
    b = 5*((a % 1461)//4) + 2
    y, m, d = a//1461, b//153 + 3, (b % 153)//5 + 1
    if m >= 13:
        y += 1
        m -= 12
    return y, m, d


def convert_time_mjd2jd(mjd):
    jd = mjd + 2400000.5
    return jd


def convert_time_nicertime2mjd(time):
    """Convert nicertime to MJD.

    Alias for 'convert_reftime2mjd'
    """
    return convert_time_reftime2mjd(time)


def convert_time_reftime2mjd(time):
    # reference:
    # https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time/
    return 56658 + time/3600/24


def convert_time_utc2xrismtime(utc):
    xrism_time_start = pd.Timestamp("2014-01-01 00:00:00")
    time_in_xrismtime = utc - xrism_time_start
    return time_in_xrismtime / np.timedelta64(1, 's')


def convert_time_xrismtime2utc(xrism_time_sec):
    xrism_time_start = pd.Timestamp("2014-01-01 00:00:00")
    xrism_time_sec = pd.to_timedelta(xrism_time_sec, unit="s")
    return xrism_time_sec + xrism_time_start
