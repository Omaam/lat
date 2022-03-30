"""Input and output handler for astronomical data.

This module handles fits and qdp files.
"""
from astropy.table import Table
import numpy as np
import pandas as pd


def load_qdpfile(qdp_file, columns=None):

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
        self.df_qdp = load_qdpfile(qdp_file, colnames)

    def request_flux(self, energies: list,
                     target_colname: str):

        df_qdp = self.df_qdp

        ids = np.searchsorted(df_qdp["ENERGY"], energies)
        fluxes = df_qdp[target_colname].iloc[ids]

        return fluxes

    @property
    def qdp(self):
        return self.df_qdp
