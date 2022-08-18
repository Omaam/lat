"""Convert time.
"""


def convert_reftime2mjd(time):
    # reference:
    # https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time/
    return 56658 + time/3600/24


def convert_nicertime2mjd(time):
    """Convert nicertime to MJD.

    Alias for 'convert_reftime2mjd'
    """
    return convert_reftime2mjd(time)
