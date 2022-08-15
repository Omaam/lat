"""Input and output handler for astronomical data.

This module handles fits and qdp files.
"""
import warnings

from astropy.table import Table
import matplotlib.pyplot as plt
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

    def __init__(self, qdp_file: str, component_names: list = None):
        self.df_qdp = self._load_qdpfile(qdp_file, component_names)

        # Number of columns except for the components is 7.
        self.num_comp = len(self.df_qdp.columns) - 7

    def request_flux(self, energy_range: list,
                     target_colname: str):

        df_qdp = self.df_qdp

        ids = np.searchsorted(df_qdp["energy"], energy_range)
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
        cond_for_range = energy_range[0] <= self.df_qdp["energy"]
        cond_for_range &= self.df_qdp["energy"] < energy_range[1]
        spec_within_range = self.df_qdp[cond_for_range][target_colname]
        countrate = spec_within_range.sum()
        return countrate

    def plot_spec(self, xlim=None, ylim_spec=None, figsize=None):

        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=figsize,
            sharex=True, constrained_layout=True,
            gridspec_kw={
                "height_ratios": [2, 1]}
        )

        energy = self.df_qdp["energy"].values

        ax0.errorbar(energy, self.df_qdp["obs"],
                     yerr=self.df_qdp["obs_err"],
                     drawstyle="steps-mid",
                     linestyle="-", color="k")
        ax0.errorbar(energy, self.df_qdp["model_total"],
                     linestyle="--", color="k")

        models = self.df_qdp.values[:, 5:5+self.num_comp].T
        for model in models:
            ax0.errorbar(energy, model, linestyle="--", color="k")

        ax0.set_xscale("log")
        ax0.set_ylabel("counts/s/keV")
        # ax0.set_ylim(ylim_spec)
        ax0.set_yscale("log")

        ax1.errorbar(energy, self.df_qdp["resid"],
                     yerr=self.df_qdp["resid_err"],
                     fmt=".", color="k")
        ax1.axhline(color="r", zorder=99)
        ax1.set_xlabel("Energy (keV)")
        # ax1.set_xlim(xlim)
        ax1.set_xscale("log")
        ax1.set_ylabel("(data-model)/model")

        return fig

    @property
    def qdp(self):
        warnings.warn(
            "Don't use 'qdp' when getting dataframe."
            "Insted, use values."
        )
        return self.df_qdp

    @property
    def values(self):
        return self.df_qdp

    def _create_columns_for_qdp(self, component_names: list):
        start = ["energy", "energy_err",
                 "obs", "obs_err",
                 "model_total"]
        comps = list(component_names)
        end = ["resid", "resid_err"]
        return start + comps + end

    def _load_qdpfile(self, qdp_file, component_names=None):

        # Reshape df for qdp in proper one.
        df = pd.read_table(qdp_file, skiprows=3, header=None, sep=" ")
        idx_change_point = np.where(df.iloc[:, 0] == "NO")[0][0]
        df1 = df.iloc[:idx_change_point].reset_index(drop=True)
        df2 = df.iloc[idx_change_point+1:].reset_index(drop=True)
        df_tot = pd.merge(df1, df2, on=[0, 1])
        colnum_all_no = np.where(df_tot.iloc[0] == "NO")[0]
        df_tot = df_tot.drop(columns=df_tot.columns[colnum_all_no])
        df_tot = df_tot.astype(float)

        if component_names is None:
            num_comp = len(df_tot.columns) - 7
            component_names = [f"comp_{i}" for i in range(num_comp)]

        df_tot.columns = self._create_columns_for_qdp(component_names)

        return df_tot
