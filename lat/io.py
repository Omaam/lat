"""Input and output handler for astronomical data.
"""
import numpy as np
import pandas as pd


def load_qdpfile(qdp_name, columns=None):

    # Reshape df for qdp in proper one.
    df = pd.read_table(qdp_name, skiprows=3, header=None, sep=" ")
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
