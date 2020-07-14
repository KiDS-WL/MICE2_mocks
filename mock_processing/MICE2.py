import numpy as np


def evolution_correction(mag, redshift):
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
