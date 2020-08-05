import os

import toml

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gamma, gammainc  # Gamma and incomplete Gamma


def magnification_correction(kappa, mag):
    """
    Magnification calculated from the convergence, following Fosalba+15 eq. 21.

    Parameters
    ----------
    kappa : array_like
        Convergence field at the galaxy positions.
    mag : array_like
        (Evolution corrected) model magnitudes.

    Returns
    -------
    mag_magnified : array_like
        Magnitudes corrected for magnification.
    """
    d_mu = 2.0 * kappa
    mag_magnified = mag - 2.5 * np.log10(1 + d_mu)
    return mag_magnified


def magnification_correction_wrapped(kappa, *mags):
    """
    Wrapper for magnification_correction() to compute the magnification
    correction for a set of magnitudes simultaneously.

    Parameters
    ----------
    kappa : array_like
        Convergence field at the galaxy positions.
    *mags : array_like
        Series of (evolution corrected) model magnitudes.

    Returns
    -------
    mags_magnified : tuple of array_like
        Series of magnification corrected model magnitudes (matching input
        order).
    """
    d_mu = 2.0 * kappa
    d_mag = -2.5 * np.log10(1 + d_mu)
    mags_magnified = tuple(mag + d_mag for mag in mags)
    return mags_magnified


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
    r_effective : float
        The radius within which the percentile of flux is emitted.
    """
    assert(0.0 <= f_B <= 1.0)
    assert(R_e_Disk >= 0.0 and R_e_Bulge >= 0.0)
    x0 = (1.0 - f_B) * R_e_Disk + f_B * R_e_Bulge
    solution = root_scalar(
        f_R_e, fprime=f_R_e_derivative, x0=x0, method=method, maxiter=100,
        args=(R_e_Disk, R_e_Bulge, f_B, percentile))
    r_effective = solution.root
    return r_effective


def find_percentile_wrapped(
        percentile, R_e_Disk, R_e_Bulge, f_B, method="newton"):
    """
    Wrapper for find_percentile() to compute the effective radius for a set of
    galaxy.

    Parameters
    ----------
    percentile : float
        The percentile of emitted flux from within the radius of interest.
    R_e_Disk : array_like
        Effective angular size of the disk component.
    R_e_Bulge : array_like
        Effective angular size of the bulge component.
    f_B : array_like
        Bulge fraction, (flux bulge / total flux).
    method : sting
        Root-finding method to use (from scipy.optimize.root_scalar).

    Returns
    -------
    r_effective : array_like
        The radius within which the percentile of flux is emitted.
    """
    find_percentile_vectorized = np.vectorize(
        find_percentile, R_e_Disk.dtype.kind)
    r_effective = find_percentile_vectorized(
        percentile, R_e_Disk, R_e_Bulge, f_B, method)
    return r_effective


def FWHM_to_sigma(FWHM):
    # sigma = FWHM / (2.0 * sqrt(2.0 * log(2.0)))
    return FWHM / 2.3548200450309493


def apertures_SExtractor(config, filter_key, r_effective, ba_ratio):
    # compute the intrinsic galaxy major and minor axes and area
    psf_sigma = config.PSF[filter_key]
    mag_auto_scale = config.SExtractor["phot_autoparams"]
    if config.legacy:
        galaxy_major = r_effective * mag_auto_scale
    else:
        galaxy_major = r_effective
    galaxy_minor = galaxy_major * ba_ratio
    # "convolution" with the PSF
    if config.legacy:
        aperture_major = np.sqrt(galaxy_major**2 + psf_sigma**2)
        aperture_minor = np.sqrt(galaxy_minor**2 + psf_sigma**2)
    else:  # mag auto: scaling the Petrosian radius by 2.5 (default)
        aperture_major = mag_auto_scale * np.sqrt(
            galaxy_major**2 + psf_sigma**2)
        aperture_minor = mag_auto_scale * np.sqrt(
            galaxy_minor**2 + psf_sigma**2)
    aperture_ba = aperture_minor / aperture_major
    # compute the aperture area
    aperture_area = np.pi * aperture_major * aperture_minor
    psf_area = np.pi * psf_sigma**2
    # compute the S/N correction by comparing the aperture area to the PSF
    snr_correction = np.sqrt(psf_area / aperture_area)
    return aperture_major, aperture_minor, snr_correction


def apertures_GAaP(config, filter_key, r_effective, ba_ratio):
    psf_sigma = FWHM_to_sigma(config.PSF[filter_key])
    aper_min = config.GAaP["aper_min"]
    aper_max = config.GAaP["aper_max"]
    # compute the intrinsic galaxy major and minor axes and area
    galaxy_major = r_effective
    galaxy_minor = galaxy_major * ba_ratio
    # "convolution" with the PSF
    aperture_major = np.sqrt(galaxy_major**2 + psf_sigma**2)
    aperture_minor = np.sqrt(galaxy_minor**2 + psf_sigma**2)
    # GAaP aperture must be betweem aper_min and aper_max
    gaap_major = np.minimum(
        np.sqrt(aperture_major**2 + aper_min**2), aper_max)
    gaap_minor = np.minimum(
        np.sqrt(aperture_minor**2 + aper_min**2), aper_max)
    gaap_ba = gaap_minor / gaap_major
    # compute the aperture area
    gaap_area = np.pi * gaap_major * gaap_minor
    psf_area = np.pi * psf_sigma**2
    # compute the S/N correction by comparing the aperture area to the PSF
    snr_correction = np.sqrt(psf_area / gaap_area)
    return gaap_major, gaap_minor, snr_correction


def apertures_wrapped(method, config, r_effective, ba_ratio):
    # select the photometry method
    if method == "GAaP":
        aperture_func = apertures_GAaP
    elif method == "SExtractor":
        aperture_func = apertures_SExtractor
    else:
        raise ValueError("invalid photometry method: {:}".format(method))
    results = []
    # iterate through the psf sizes of all filters
    for filter_key in config.filter_names:
        result = aperture_func(
            config, filter_key, r_effective, ba_ratio)
        # collect the results
        results.extend(result)
    return results


def photometry_realisation(config, filter_key, mag, snr_correction):
    mag_lim = config.limits[filter_key]
    if config.legacy:  # computation in magnitudes
        # compute the S/N of the model magnitudes
        snr = 10 ** (-0.4 * (mag - mag_lim)) * config.limit_sigma
    else:  # computation in fluxes
        # compute model fluxes and the S/N
        flux = 10 ** (-0.4 * mag)
        flux_err = 10 ** (-0.4 * mag_lim)
        snr = flux / flux_err * config.limit_sigma
    snr *= snr_correction  # aperture correction
    snr = np.maximum(snr, config.SN_floor)  # clip S/N
    if config.legacy:  # magnitudes draw incorrectly with Gaussian errors
        mag_err = 2.5 / np.log(10.0) / snr
        # compute the magnitde realisation and S/N
        real = np.random.normal(mag, mag_err, size=len(mag))
        snr = 10 ** (-0.4 * (real - mag_lim)) * config.limit_sigma
    else:  # magnitudes constructed from fluxes with Gaussian errors
        # compute the flux realisation and S/N
        flux = np.random.normal(  # approximation for Poisson error
            flux, flux_err, size=len(flux))
        flux = np.maximum(flux, 1e-3 * flux_err)  # prevent flux <= 0.0
        snr = flux / flux_err * config.limit_sigma
        # convert fluxes to magnitudes
        real = -2.5 * np.log10(flux)
    snr *= snr_correction  # aperture correction
    snr = np.maximum(snr, config.SN_floor)  # clip S/N
    # compute the magnitude error of the realisation
    real_err = 2.5 / np.log(10.0) / snr
    # set magnitudes of undetected objects and mag < 5.0 to 99.0
    not_detected = (snr < config.SN_detect) | (real < 5.0)
    real[not_detected] = config.no_detect_value
    real_err[not_detected] = mag_lim - 2.5 * np.log10(config.limit_sigma)
    return real, real_err


def photometry_realisation_wrapped(config, *mag_mag_lim_snr_correction):
    # iterate through the listing of magnitude columns, magnitude limits and
    # S/N correction factors for all filters
    results = []
    for idx, filter_key in enumerate(config.filter_names):
        # unpack the arguments
        mag, mag_lim, snr_correction = \
            mag_mag_lim_snr_correction[3*idx:3*(idx+1)]
        result = photometry_realisation(
            config, filter_key, mag, snr_correction)
        # collect the results
        results.extend(result)
    return results
