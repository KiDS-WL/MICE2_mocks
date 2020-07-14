import os

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gamma, gammainc  # Gamma and incomplete Gamma


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


def magnification_correction(mag, kappa):
    """
    Magnification calculated from the convergence, following Fosalba+15 eq. 21.

    Parameters
    ----------
    mag : array_like
        (Evolution corrected) model magnitudes.
    kappa : array_like
        Convergence field at the galaxy positions.

    Returns
    -------
    mag_magnified : array_like
        Magnitudes corrected for magnification.
    """
    d_mu = 2.0 * kappa
    mag_magnified = mag - 2.5 * np.log10(1 + d_mu)
    return mag_magnified


def f_R_e(R, R_e_Disk, R_e_Bulge, f_B, percentile=0.5):
    """
    Function used to find the effective radius of a galaxy with combined
    bulge and disk component. Computes the fraction of the total flux emitted
    within a radius R minus a percentile. The percentile sets the zero-point
    of this function.

    By running a root-finding algorithm, the radius corresponding to a given
    percentile of emitted flux within can be computed. Setting the percentile
    to 0.5 yields the effective radius, the radius from within which 50% of the
    total flux are emitted.

    Parameters
    ----------
    R : float
        Radius (angular) at which to evaluate the function.
    R_e_Disk : float
        Effective angular size of the disk component.
    R_e_Bulge : float
        Effective angular size of the bulge component.
    f_B : float
        Bulge fraction, (flux bulge / total flux).
    percentile : float
        The percentile subtracted from the calculated flux fraction within R.

    Returns
    -------
    flux_fraction_offset : float
        Fraction of flux emitted within R minus percentile.
    """
    if R_e_Disk == 0.0 or f_B == 1.0:  # no disk component
        disk_term = 0.0
    else:  # evaluate the integrated Sersic n=1 profile
        x_D = 1.6721 * R / R_e_Disk
        disk_term = (1.0 - f_B) * gammainc(2, x_D)
    if R_e_Bulge == 0.0 or f_B == 0.0:  # no bulge component
        bulge_term = 0.0
    else:  # evaluate the integrated Sersic n=4 profile
        x_B = 7.6697 * (R / R_e_Bulge) ** 0.25
        bulge_term = f_B * gammainc(8, x_B)
    # disk_term and bulge_term are already normalized by the total flux
    flux_fraction_offset = disk_term + bulge_term - percentile
    return flux_fraction_offset


def f_R_e_derivative(R, R_e_Disk, R_e_Bulge, f_B, *args):
    """
    Derivative of f_R_e wrt. the radius used by the root-finding algorithm.

    Parameters
    ----------
    R : float
        Radius (angular) at which to evaluate the derivative.
    R_e_Disk : float
        Effective angular size of the disk component.
    R_e_Bulge : float
        Effective angular size of the bulge component.
    f_B : float
        Bulge fraction, (flux bulge / total flux).

    Returns
    -------
    flux_fraction_der : float
        Derivative of f_R_e.
    """
    if R_e_Disk == 0.0 or f_B == 1.0:  # no disk component
        disk_term = 0.0
    else:  # evaluate the derivative of the integrated Sersic n=1 profile
        b_1 = 1.6721
        x_D = b_1 * R / R_e_Disk
        disk_term = (1.0 - f_B) * np.exp(-x_D) * x_D / R_e_Disk * b_1
    if R_e_Bulge == 0.0 or f_B == 0.0:  # no bulge component
        bulge_term = 0.0
    else:  # evaluate the derivative of the integrated Sersic n=4 profile
        b_4 = 7.6697
        x_B = b_4 * (R / R_e_Bulge) ** 0.25
        bulge_term = (
            f_B * np.exp(-x_B) / 5040.0 * x_B**4 / R_e_Bulge * b_4**4 / 4.0)
    # combined derivative is sum of disk and bulge derivatives
    flux_fraction_der = disk_term + bulge_term
    return flux_fraction_der


def find_percentile(
        percentile, R_e_Disk, R_e_Bulge, f_B, method="newton"):
    """
    Compute the radius within which a certain percentile of flux is emitted
    using the scipy.optimize.root_scalar root-finding algorithm. By default
    the Newton's method is used.

    Parameters
    ----------
    percentile : float
        The percentile of emitted flux from within the radius of interest.
    R_e_Disk : float
        Effective angular size of the disk component.
    R_e_Bulge : float
        Effective angular size of the bulge component.
    f_B : float
        Bulge fraction, (flux bulge / total flux).
    method : sting
        Root-finding method to use (from scipy.optimize.root_scalar).

    Returns
    -------
    solution.root : float
        The radius within which the percentile of flux is emitted.
    """
    assert(0.0 <= f_B <= 1.0)
    assert(R_e_Disk >= 0.0 and R_e_Bulge >= 0.0)
    x0 = (1.0 - f_B) * R_e_Disk + f_B * R_e_Bulge
    solution = root_scalar(
        f_R_e, fprime=f_R_e_derivative, x0=x0, method=method, maxiter=100,
        args=(R_e_Disk, R_e_Bulge, f_B, percentile))
    return solution.root
