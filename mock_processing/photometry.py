import os

import numpy as np


def load_photometry(table, photometry_path, filter_selection=None):
    photometry_columns = {}
    for column in table.colnames:
        # match the root name of each column against the photometry path
        root, key = os.path.split(column)
        if root == photometry_path:
            # check if the matching filter is excluded
            if filter_selection is not None:
                if key not in filter_selection:
                    continue
            photometry_columns[key] = column
    if len(photometry_columns) == 0:
        raise KeyError("photometry not found: {:}".format(photometry_path))
    return photometry_columns


def MICE2_evolution_correction(mag, redshift):
    """
    Evolution correction as described in the official MICE2 manual
    https://www.dropbox.com/s/0ffa8e7463n8h1q/README_MICECAT_v2.0_for_new_CosmoHub.pdf?dl=0

    Parameters
    ----------
    mag : array_like
        Uncorrected model magnitudes.
    redshift : array_like
        True galaxy redshift (z_cgal).

    Returns
    -------
    mag_evo : array_like
        Evolution corrected model magnitudes.
    """
    return mag - 0.8 * (np.arctan(1.5 * redshift) - 0.1489)
