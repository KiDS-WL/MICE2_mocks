import numpy as np

from .core.parallel import workload


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


@workload(0.10)
def flux_to_magnitudes_wrapped(*fluxes):
    """
    Wrapper for flux_to_magnitudes() to convert a set of fluxes to magnitudes
    simultaneously.

    Parameters
    ----------
    *fluxes : array_like
        Seroes of input flux data.

    Returns
    -------
    mags : tuple of array_like
        Series of AB magnitudes to the input fluxes (same order).
    """
    mags = tuple(flux_to_magnitudes(flux) for flux in fluxes)
    return mags
