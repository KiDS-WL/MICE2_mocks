#
# This module implements methods to add a realistic photometry realisation to
# the mock data.
#

import os

import toml

import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gamma, gammainc  # Gamma and incomplete Gamma

from galmock.core.config import (Parameter, ParameterCollection,
                                 ParameterGroup, ParameterListing, Parser)
from galmock.core.parallel import Schedule


class PhotometryParser(Parser):
    """
    Parser for the TOML photometry configuration file.
    
    From python use
        print(PhotometryParser.default)
    to get an empty, default configuration file
    """

    default = ParameterCollection(
        Parameter(
            "method", str, "SExtractor",
            "photometry algorthim to apply (choices: SExtractor, GAaP)"),
        Parameter(
            "legacy", bool, False,
            "use legacy mode (van den Busch et al. 2020)"),
        Parameter(
            "aperture_name", str, "SExtractor",
            "name under with the aperture realisation is stored in the "
            "aperture/ directory of the data store"),
        ParameterGroup(
            "intrinsic",
            Parameter(
                "r_effective", str, "shape/R_effective",
                "path of the effective radius column in the data store"),
            Parameter(
                "flux_frac", float, 0.5,
                "this defines the effective radius by setting the fraction of "
                "the total flux/luminosity for which the radius of the source "
                "is computed"),
            header=None),
        ParameterListing(
            "limits", float,
            header=(
                "numerical values for the magnitude limits, the keys must be "
                "the same as in column map file used for "
                "mocks_init_pipeline")),
        ParameterListing(
            "PSF", float,
            header=(
                "numerical values for the PSF FWHM in arcsec, the keys must "
                "be the same as in column map file used for "
                "mocks_init_pipeline")),
        ParameterGroup(
            "GAaP",
            Parameter(
                "aper_min", float, 0.7,
                "GAaP lower aperture size limit in arcsec"),
            Parameter(
                "aper_max", float, 2.0,
                "GAaP upper aperture size limit in arcsec"),
            header=None),
        ParameterGroup(
            "SExtractor",
            Parameter(
                "phot_autoparams", float, 2.5,
                "MAG_AUTO-like scaling factor for Petrosian radius, here "
                "applied to intrinsic galaxy size derived from effective "
                "radius"),
            header=None),
        ParameterGroup(
            "photometry",
            Parameter(
                "apply_apertures", bool, True,
                "whether to include the aperture size in the SNR computation"),
            Parameter(
                "limit_sigma", float, 1.0,
                "sigma value of the detection (magnitude) limit with respect "
                "to the sky background"),
            Parameter(
                "no_detect_value", float, 99.0,
                "magnitude value assigned to undetected galaxies"),
            Parameter(
                "SN_detect", float, 1.0,
                "signal-to-noise ratio detection limit"),
            Parameter(
                "SN_floor", float, 0.2,
                "numerical lower limit for signal-to-noise ratio"),
            header=None),
        header=(
            "This configuration file is required for mocks_apertures and "
            "mocks_photometry. It defines the free parameters of the aperture "
            "and photometry methods as well as observational PSF sizes and "
            "magnitude limits."))

    def _run_checks(self):
        if set(self.limits.keys()) != set(self.PSF.keys()):
            message = "some filter(s) do not provide pairs of magnitude limit "
            message += "and PSF size"
            raise KeyError(message)

    @property
    def filter_names(self):
        return tuple(sorted(self.limits.keys()))


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


@Schedule.description("applying flux magnification")
@Schedule.workload(0.10)
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


@Schedule.description("calculating size from light profile")
@Schedule.CPUbound
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
    """
    Convert the full width at half maximum of a Gaussian to it corresponding
    sigma value.

    Parameters:
    -----------
    FWHM : float
        Full width at half maximum.
    
    Returns:
    sigma : float
        Value for sigma of the Gaussian.
    """
    # sigma = FWHM / (2.0 * sqrt(2.0 * log(2.0)))
    sigma =  FWHM / 2.3548200450309493
    return sigma


def apertures_SExtractor(config, filter_key, r_effective, ba_ratio):
    """
    Construct an Source Extractor-like aperture for a given intrinic size and
    galaxy axis ratio for a given PSF size. This aproximates scaling the
    Petrosion ratius by a constant factor (typically 2.5) to obtain an aperture
    for MAG_AUTO magnitudes.

    Parameters:
    -----------
    config : PhotometryParser
        Configures all parameters required by the photometry functions.
    filter_key : str
        Identifier key for the photometric filter to process.
    r_effective : float, array-like
        Effective radius of the galaxies, proxy for the intrinsic size.
    ba_ratio : float, array-like
        Ration of minor to major galaxies axes.

    Returns:
    --------
    aperture_major : float, array-like
        Constructed aperture major axis.
    aperture_minor : float, array-like
        Constructed aperture minor axis.
    snr_correction : float, array-like
        Signal-to-noise ratio correction factor based on the aperture size.
        Calculated from the square root of the ratio of the PSF (point-source)
        to galaxy aperture area ratio.
    """
    # compute the intrinsic galaxy major and minor axes and area
    mag_auto_scale = config.SExtractor["phot_autoparams"]
    if config.legacy:  # legacy mode (van den Busch et al. 2020)
        galaxy_major = r_effective * mag_auto_scale
        galaxy_minor = galaxy_major * ba_ratio
        # "convolution" with the PSF
        psf_sigma = config.PSF[filter_key]
        aperture_major = np.sqrt(galaxy_major**2 + psf_sigma**2)
        aperture_minor = np.sqrt(galaxy_minor**2 + psf_sigma**2)
        # compute the aperture area
        psf_area = np.pi * psf_sigma**2
    else:
        galaxy_major = r_effective
        galaxy_minor = galaxy_major * ba_ratio
        # "convolution" with the PSF
        psf_sigma = FWHM_to_sigma(config.PSF[filter_key])
        aperture_major = mag_auto_scale * np.sqrt(
            galaxy_major**2 + psf_sigma**2)
        aperture_minor = mag_auto_scale * np.sqrt(
            galaxy_minor**2 + psf_sigma**2)
        # compute the aperture area
        psf_area = np.pi * (mag_auto_scale * psf_sigma)**2
    aperture_area = np.pi * aperture_major * aperture_minor
    # compute the S/N correction by comparing the aperture area to the PSF
    snr_correction = np.sqrt(psf_area / aperture_area)
    return aperture_major, aperture_minor, snr_correction


def apertures_GAaP(config, filter_key, r_effective, ba_ratio):
    """
    Construct an GAaP-like aperture (Gaussian Aperture and Photometry) for a
    given intrinic size and galaxy axis ratio for a given PSF size.

    Parameters:
    -----------
    config : PhotometryParser
        Configures all parameters required by the photometry functions.
    filter_key : str
        Identifier key for the photometric filter to process.
    r_effective : float, array-like
        Effective radius of the galaxies, proxy for the intrinsic size.
    ba_ratio : float, array-like
        Ration of minor to major galaxies axes.

    Returns:
    --------
    aperture_major : float, array-like
        Constructed aperture major axis.
    aperture_minor : float, array-like
        Constructed aperture minor axis.
    snr_correction : float, array-like
        Signal-to-noise ratio correction factor based on the aperture size.
        Calculated from the square root of the ratio of the PSF (point-source)
        to galaxy aperture area ratio.
    """
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
    gaap_pointsource = np.minimum(
        np.sqrt(psf_sigma**2 + aper_min**2), aper_max)
    # compute the aperture area
    gaap_area = np.pi * gaap_major * gaap_minor
    psf_area = np.pi * gaap_pointsource**2
    # compute the S/N correction by comparing the aperture area to the PSF
    snr_correction = np.sqrt(psf_area / gaap_area)
    return gaap_major, gaap_minor, snr_correction


@Schedule.description("constructing apertures")
@Schedule.workload(0.15)
def apertures_wrapped(config, r_effective, ba_ratio):
    """
    Wrapper for apertures_SExtractor() and apertures_GAaP() to compute
    apertures for a set of galaxies for all configured filters.

    Parameters:
    -----------
    config : PhotometryParser
        Configures all parameters required by the photometry functions.
    r_effective : float, array-like
        Effective radius of the galaxies, proxy for the intrinsic size.
    ba_ratio : float, array-like
        Ration of minor to major galaxies axes.

    Returns:
    --------
    results : list of float or array-like
        Returns a concatenated listing of aperture major axes, aperture minor
        axes and signal-to-noise correction factors for the galaxies in each
        filter specified in the configuration.
    """
    # select the photometry method
    if config.method == "GAaP":
        aperture_func = apertures_GAaP
    elif config.method == "SExtractor":
        aperture_func = apertures_SExtractor
    else:
        message = "invalid photometry method: {:}"
        raise ValueError(message.format(config.method))
    results = []
    # iterate through the psf sizes of all filters
    for filter_key in config.filter_names:
        result = aperture_func(
            config, filter_key, r_effective, ba_ratio)
        # collect the results
        results.extend(result)
    return results


def photometry_realisation(config, filter_key, mag, snr_correction):
    """
    Compute a photometry realisation in a given filter for a given limiting
    magnitude.

    Parameters:
    -----------
    config : PhotometryParser
        Configures all parameters required by the photometry functions.
    filter_key : str
        Identifier key for the photometric filter to process.
    mag : float or array-like
        Magnitude of the input galaxies in the given filter.
    snr_correction : float, array-like
        Signal-to-noise ratio correction factor based on the aperture size.
        Calculated from the square root of the ratio of the PSF (point-source)
        to galaxy aperture area ratio.

    Returns:
    --------
    real : float or array-like
        Magnitude realisation considering the limiting magnitude.
    real_err : float or array-like
        Gaussian error of the magnitude realisation considering the limiting
        magnitude.
    """
    mag_lim = config.limits[filter_key]
    limit_sigma = config.photometry["limit_sigma"]
    if config.legacy:  # legacy mode (van den Busch et al. 2020)
        # compute the S/N of the model magnitudes
        snr = 10 ** (-0.4 * (mag - mag_lim)) * limit_sigma
    else:  # computation in fluxes
        # compute model fluxes and the S/N
        flux = 10 ** (-0.4 * mag)
        flux_err = 10 ** (-0.4 * mag_lim)
        snr = flux / flux_err * limit_sigma
    snr *= snr_correction  # aperture correction
    snr = np.maximum(snr, config.photometry["SN_floor"])  # clip S/N
    if config.legacy:  # legacy mode (van den Busch et al. 2020)
        # assumes Gaussian magnitude error
        mag_err = 2.5 / np.log(10.0) / snr
        # compute the magnitde realisation and S/N
        real = np.random.normal(mag, mag_err, size=len(mag))
        snr = 10 ** (-0.4 * (real - mag_lim)) * limit_sigma
    else:
        # assumes Gaussian flux error
        # compute the flux realisation and S/N
        flux = np.random.normal(  # approximation for Poisson error
            flux, flux_err, size=len(flux))
        flux = np.maximum(flux, 1e-3 * flux_err)  # prevent flux <= 0.0
        snr = flux / flux_err * limit_sigma
        # convert fluxes to magnitudes
        real = -2.5 * np.log10(flux)
    snr *= snr_correction  # aperture correction
    snr = np.maximum(snr, config.photometry["SN_floor"])  # clip S/N
    # compute the magnitude error of the realisation
    real_err = 2.5 / np.log(10.0) / snr
    # set magnitudes of undetected objects and mag < 5.0 to 99.0
    not_detected = (snr < config.photometry["SN_detect"]) | (real < 5.0)
    real[not_detected] = config.photometry["no_detect_value"]
    real_err[not_detected] = mag_lim - 2.5 * np.log10(limit_sigma)
    return real, real_err


@Schedule.description("generating photometry realisation")
@Schedule.workload(0.33)
def photometry_realisation_wrapped(config, *mag_mag_lim_snr_correction):
    """
    Wrapper for photometry_realisation() to compute the magnitude realistions
    for all configured filters simultaneously.

    Parameters:
    -----------
    config : PhotometryParser
        Configures all parameters required by the photometry functions.
    mag_mag_lim_snr_correction : listing of float or array-like
        Listing of input magnitudes, magnitude limit and signal-to-noise
        correction factors for each filter in the configuration.

    Returns:
    --------
    results : list of float or array-like
        Returns a concatenated listing of magnitude realisations and their
        Gaussian errors the galaxies in each filter specified in the
        configuration.
    """
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
