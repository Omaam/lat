"""Convert time.
"""


def convert_reftime2mjd(time):
    # reference:
    # https://heasarc.gsfc.nasa.gov/docs/nicer/analysis_threads/time/
    return 56658 + time/3600/24
