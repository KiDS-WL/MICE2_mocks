import numpy as np

from .core.parallel import Schedule


def evolution_correction(redshift, mag):
    """
    Evolution correction as described in the official MICE2 manual
    https://www.dropbox.com/s/0ffa8e7463n8h1q/README_MICECAT_v2.0_for_new_CosmoHub.pdf?dl=0

    Parameters
    ----------
    redshift : array_like
        True galaxy redshift (z_cgal).
    mag : array_like
        Uncorrected model magnitudes.

    Returns
    -------
    mag_evo : array_like
        Evolution corrected model magnitudes.
    """
    return mag - 0.8 * (np.arctan(1.5 * redshift) - 0.1489)


@Schedule.workload(0.10)
def evolution_correction_wrapped(redshift, *mags):
    """
    Wrapper for evolution_correction() to compute the evolution correction for
    a set of magnitudes simultaneously.

    Parameters:
    -----------
    redshift : array_like
        True galaxy redshift (z_cgal).
    *mags : array_like
        Series of model magnitudes.

    Returns
    -------
    mags_evo : tuple of array_like
        Series of evolution corrected model magnitudes (matching input order).
    """
    d_mag = -0.8 * (np.arctan(1.5 * redshift) - 0.1489)
    mags_evo = tuple(mag + d_mag for mag in mags)
    return mags_evo
