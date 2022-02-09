"""FitsLightCurve class.
"""

from absl import logging
from astropy.table import Table


class FitsLightCurve:
    """
    This class could calculate some fundamental calcuration
    """

    def __init__(self, lcname_fits, energy_range=None, dt=None):

        self.table = Table.read(lcname_fits, format="fits")
        logging.debug("table: %s", self.table.shape)

    def calc_psd(self, mode="lomb-scagle"):
        return freqs, powers

    def calc_rms(self):
        return rms

    @property
    def pandas(self):
        return self.table.to_pandas()
