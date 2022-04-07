"""Utility for analyzing astronomical data.
"""
import numpy as np


def calc_energy_centers_and_errors(energy_edges):
    """Calculate energy centers and errors."""
    energy_edges = np.array(energy_edges)

    energy_centers = (energy_edges[1:] + energy_edges[:-1]) / 2

    energy_errors_upper = energy_edges[1:] - energy_centers
    energy_errors_lower = energy_centers - energy_edges[:-1]
    energy_errors = np.vstack(
        [energy_errors_lower, energy_errors_upper])

    return energy_centers, energy_errors


def calc_hardnessratio(hard, hard_error, soft, soft_error):
    """Calculate errors for hardness ratio."""
    hard = np.array(hard)
    hard_error = np.array(hard_error)
    soft = np.array(soft)
    soft_error = np.array(soft_error)

    ratio = hard / soft

    hard_term = 1/soft**2 * hard_error**2
    soft_term = hard**2/soft**4 * hard_error**2
    error = np.sqrt(hard_term + soft_term)

    return ratio, error
