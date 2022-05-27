"""Input and output handler for astronomical data.

This module handles fits and qdp files.
"""
from astropy.table import Table
import numpy as np
import pandas as pd


def load_eventfile(event_name):
    """Input event file and output dataframe.
    """
    df_event = None
    return df_event


def load_lcfile(lcname):
    """Input light curve and output dataframe.
    """
    tbl = Table.read(lcname, format="fits", hdu=1)
    df_lc = tbl.to_pandas()
    return df_lc


class QDP:

    def __init__(self, qdp_file: str, colnames: list = None):
        self.df_qdp = self._load_qdpfile(qdp_file, colnames)

    def request_flux(self, energy_range: list,
                     target_colname: str):

        df_qdp = self.df_qdp

        ids = np.searchsorted(df_qdp["ENERGY"], energy_range)
        fluxes = df_qdp[target_colname].iloc[ids]

        return fluxes

    def calc_count_within_range(self, energy_range: list,
                                target_colname: str,
                                ) -> int or float:
        """Calculate counts or countrate within energy range.

        Args:
            energy_range: Energy range to calculate counts or countrate.
            target_colname: Columns name from which you obtain count.
        Returns:
            countrate: Count rate.
        """
        cond_for_range = energy_range[0] <= self.df_qdp["ENERGY"]
        cond_for_range &= self.df_qdp["ENERGY"] < energy_range[1]
        spec_within_range = self.df_qdp[cond_for_range][target_colname]
        countrate = spec_within_range.sum()
        return countrate

    @property
    def qdp(self):
        return self.df_qdp

    def _load_qdpfile(self, qdp_file, columns=None):

        # Reshape df for qdp in proper one.
        df = pd.read_table(qdp_file, skiprows=3, header=None, sep=" ")
        idx_change_point = np.where(df.iloc[:, 0] == "NO")[0][0]
        df1 = df.iloc[:idx_change_point].reset_index(drop=True)
        df2 = df.iloc[idx_change_point+1:].reset_index(drop=True)
        df_tot = pd.merge(df1, df2, on=[0, 1])
        colnum_all_no = np.where(df_tot.iloc[0] == "NO")[0]
        df_tot = df_tot.drop(columns=df_tot.columns[colnum_all_no])
        df_tot = df_tot.astype(float)

        if columns is not None:
            df_tot.columns = columns

        return df_tot
