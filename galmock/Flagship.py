#
# This module implements methods to process raw data from the Flagship galaxy
# mock cataloge.
#

import numpy as np

from galmock.core.parallel import Schedule


def flux_to_magnitudes(flux):
    """
    Convert flux columns in the Flagship simulation to AB magnitudes with a
    zero-point of 48.6.

    Parameters:
    -----------
    flux : array_like
        Input flux data.
    
    Returns:
    --------
    mag : array_like
        AB magnitude to given flux.
    """
    ZP = 48.6
    mag = -2.5 * np.log10(flux) - ZP
    return mag


@Schedule.description("converting fluxes to magnitudes")
@Schedule.workload(0.10)
def flux_to_magnitudes_wrapped(*fluxes):
    """
    Wrapper for flux_to_magnitudes() to convert a set of fluxes to magnitudes
    simultaneously.

    Parameters
    ----------
    *fluxes : array_like
        Series of input flux data.

    Returns
    -------
    mags : tuple of array_like
        Series of AB magnitudes to the input fluxes (same order).
    """
    mags = tuple(flux_to_magnitudes(flux) for flux in fluxes)
    return mags


@Schedule.description("find central halo galaxies")
@Schedule.IObound
def find_central_galaxies(galaxy_idx):
    """
    Identify the host-halo's central galaxy based on the Flagship galaxy index.

    Parameters
    ----------
    galaxy_idx : array_like
        Index of the galaxy within the halo.

    Returns
    -------
    is_central : bool
        Whether a galaxy is the halo's central galaxy.
    """
    return galaxy_idx == 0
