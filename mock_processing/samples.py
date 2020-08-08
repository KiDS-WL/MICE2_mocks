import json
import os
import pickle
from collections import OrderedDict

import numpy as np

from scipy.interpolate import interp1d, interp2d


# folder containing data for the spec. success rate for the deep spec-z samples
SUCCESS_RATE_DIR = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "sample_data")
# files that provide (redshift dependent) sample density
DENSITY_FILE_TEMPLATE = os.path.join(SUCCESS_RATE_DIR, "{:}.json")


class AverageShiftedHistogram(object):

    def __init__(
            self, data, xmin=None, xmax=None, width=None, density=False,
            smooth=10, kind="linear", fill_value=None):
        self._xmin = np.min(data) if xmin is None else xmin
        self._xmax = np.max(data) if xmax is None else xmax
        if width is None:
            n_bins = int(np.sqrt(len(data)))
            self._width = (self._xmax - self._xmin) / n_bins 
        else:
            self._width = width
        # bin the data
        binning = np.arange(self._xmin, self._xmax + self._width, self._width)
        self._centers = (binning[1:] + binning[:-1]) / 2.0
        counts = np.histogram(data, binning)[0]
        # compute the average shifted histogram
        self._smooth = int(smooth)
        self._counts = self._ash_counts(counts, density)
        # compute interpolation
        self._kind = kind
        self._fill_value = fill_value
        self._interp = self._interpolate()
 
    @classmethod
    def from_file(cls, path, kind="linear", fill_value=None):
        instance = cls.__new__(cls)
        instance._kind = kind
        instance._fill_value = fill_value
        # load the data from the file
        with open(path) as f:
            attr_dict = json.load(f)
        # set the attributes and rerun the interpolation
        for attr, value in attr_dict.items():
            if type(value) is list:
                setattr(instance, attr, np.array(value))
            else:
                setattr(instance, attr, value)
        instance._interpolate()
        return instance

    def to_file(self, path):
        # save all required attributes as JSON file
        attrs = ("_xmin", "_xmax", "_width", "_smooth", "_centers", "_counts")
        attr_dict = {}
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, np.ndarray):
                attr_dict[attr] = list(value)
            else:
                attr_dict[attr] = value
        with open(path, "w") as f:
            json.dump(attr_dict, f)

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def width(self):
        return self._width

    @property
    def smooth(self):
        return self._smooth

    @property
    def kind(self):
        return self._kind

    @property
    def fill_value(self):
        return self._fill_value

    def _ash_counts(self, counts, density):
        smoothed = np.empty_like(counts)
        idx_offset = max(1, self._smooth // 2)
        for i in range(len(smoothed)):
            start = max(0, i - idx_offset)
            end = min(len(smoothed), i + idx_offset)
            smoothed[i] = counts[start:end].sum()
        if density:
            smoothed = smoothed / np.trapz(smoothed, x=self._centers)
        return smoothed

    def _interpolate(self):
        kwargs = {"kind": self._kind}
        if self._fill_value is not None:
            kwargs.update({
                "bounds_error": False, "fill_value": self._fill_value})
        return interp1d(self._centers, self._counts, **kwargs)

    def scale_amplitude(self, scale):
        self._counts *= scale
        self._interp = self._interpolate()

    def __call__(self, x):
        return self._interp(x)


class Sampler(object):

    def __init__(self, path, mock_redshifts, mock_area):
        try:  # sample as function or redshift
            # get the sample density in arcmin^-2
            self._sample_dens = AverageShiftedHistogram.from_file(
                path, fill_value=0.0)
            self._sample_dens.scale_amplitude(3600.0)  # scale deg^-2
            # mock density in deg^-2, we can use the same binning
            self._mock_dens = AverageShiftedHistogram(
                mock_redshifts, self._sample_dens.xmin, self._sample_dens.xmax,
                self._sample_dens.width, True, self._sample_dens.smooth,
                fill_value=0.0)
            # Find the number of mock galaxies that fall in the sample redshift
            # range. If all mock galaxies are used, the number density is
            # underestimated
            n_mock = np.count_nonzero(
                (mock_redshifts >= self._sample_dens.xmin) &
                (mock_redshifts < self._sample_dens.xmax))
            # rescale such that it the n(z) integrates to mock density
            self._mock_dens.scale_amplitude(n_mock / mock_area)

        except AttributeError:  # sample surface denisty
            # get the sample density in arcmin^-2
            with open(path) as f:
                self._sample_dens = json.load(f)["density"]
                self._sample_dens *= 3600.0  # scale deg^-2
            # mock density in deg^-2
            self._mock_dens = len(mock_redshifts) / mock_area

        if self.sample_density >= self.mock_density:
            message = "sample density must be smaller than mock density"
            raise ValueError(message)

    def _integrate_redshift_density(self, histogram):
        density = np.trapz(histogram._counts, x=histogram._centers)
        return density

    @property
    def sample_density(self):
        try:
            return self._integrate_redshift_density(self._sample_dens)
        except AttributeError:
            return self._sample_dens
    
    @property
    def mock_density(self):
        try:
            return self._integrate_redshift_density(self._mock_dens)
        except AttributeError:
            return self._mock_dens

    def odds(self, redshift):
        try:
            mock_density = self._mock_dens(redshift)
            with np.errstate(divide="ignore", invalid="ignore"):
                odds = np.where(
                    mock_density == 0.0, 0.0,  # NaNs are substituted by zeros
                    self._sample_dens(redshift) / mock_density)
        except TypeError:
            odds = self._sample_dens / self._mock_dens
        return odds

    def draw(self, redshift):
        random_draw = np.random.rand(len(redshift))
        mask = random_draw < self.odds(redshift)
        return mask

    def update_bitmask(self, redshift, bitmask, bit_value):
        mask = self.draw(redshift)
        bitmask = np.bitwise_or(np.where(mask, bit_value, 0), bitmask)
        return bitmask


class BaseSelection(object):

    name = "Unnamed"
    _dtype = np.uint8
    _description = "describes the fields of the selection bit mask"

    def __init__(self, mock_redshifts, mock_area):
        # initialize a sample if a corresponding data file exists
        if self.name in IMPLEMENT_DOWNSAMPLING:
            self._sampler = Sampler(density_file, mock_redshifts, mock_area)

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def dtype(self):
        return self._dtype

    @property
    def description(self):
        return self._description.format(self.name)


class SelectKiDS(BaseSelection):

    name = "KiDS"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "the lensfit weight selection by (mask & 2), objects "
    _description += "passing the prior magnitude selection by (mask & 4) and "
    _description += "the full sample by (mask & 1)."

    def lensing_selection(self, recal_weight, prior_magnitude):
        # select objects with non-zero lensfit weight
        bitmask = np.where(recal_weight > 0.0, 2, 0).astype(self.dtype)
        # select objects that are detected in the BPZ prior band
        bitmask = np.bitwise_or(
            np.where(prior_magnitude < 90.0, 4, 0).astype(self.dtype),
            bitmask)
        return bitmask

    def __call__(self, recal_weight, prior_magnitude):
        bitmask = self.lensing_selection(recal_weight, prior_magnitude)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


###############################################################################
#                            WIDE SURVEYS                                     #
###############################################################################


class Select2dFLenS(BaseSelection):

    name = "2dFLenS"
    _description = "{:} sample selection bit mask. Select the LOWZ sample "
    _description += "by (mask & 2), the MIDZ sample by (mask & 4), the HIGHZ "
    _description += "sample by (mask & 8) and the full sample by (mask & 1)."

    def colour_selection(self, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        ###############################################################
        #   based on Blake+16                                         #
        ###############################################################
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_g) < 90.0) & (np.abs(mag_r) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_i) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_Ks) < 90.0)
        colour_mask &= (np.abs(mag_i) < 90.0) & (np.abs(mag_Z) < 90.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (mag_g - mag_r) + 1.2 * (mag_r - mag_i - 0.18)
        c_r = (mag_r - mag_i) - (mag_g - mag_r) / 4.0 - 0.18
        d_r = (mag_r - mag_i) - (mag_g - mag_r) / 8.0
        # defining the LOWZ sample
        low_z1 = colour_mask & (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.2) &
            (mag_r < (13.1 + c_p / 0.32)) &  # 13.1, 0.3
            (np.abs(c_r) < 0.2))
        low_z2 = colour_mask & (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.5) &
            (mag_g - mag_r > (1.3 + 0.25 * (mag_r - mag_i))) &
            (c_r > 0.45 - (mag_g-mag_r) / 6.0))
        low_z3 = colour_mask & (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.6) &
            (mag_r < (13.5 + c_p / 0.32)) &  # 13.5, 0.3
            (np.abs(c_r) < 0.2))
        bitmask = np.where(low_z1 | low_z2 | low_z3, 2, 0).astype(self.dtype)
        # defining the MIDZ sample
        mid_z = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            ((mag_r - mag_i) < 2.0) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.9)))  # 19.86, 1.6, 0.8
        bitmask = np.bitwise_or(
            np.where(mid_z, 4, 0).astype(self.dtype),
            bitmask)
        # defining the HIGHZ sample
        high_z = colour_mask & (
            (mag_Z < 19.9) &  # 19.95
            (mag_i > 19.9) &
            (mag_i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((mag_r - mag_Ks) > 1.9 * (mag_r - mag_i)) &
            ((mag_r - mag_i) > 0.98) &
            ((mag_i - mag_Z) > 0.6))
        bitmask = np.bitwise_or(
            np.where(high_z, 8, 0).astype(self.dtype),
            bitmask)
        return bitmask

    def __call__(self, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        bitmask = self.colour_selection(mag_g, mag_r, mag_i, mag_Z, mag_Ks)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with OR
            np.where(np.bitwise_and(bitmask, 14), 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


class SelectGAMA(BaseSelection):

    name = "GAMA"
    _description = "{:} sample selection bit mask. Select sample objects by "
    _description += "(mask & 1)"

    def __call__(self, mag_r):
        ###############################################################
        #   based on Driver+11                                        #
        ###############################################################
        bitmask = np.where(mag_r < 19.87, 1, 0).astype(self.dtype)
        return bitmask


class SelectSDSS(BaseSelection):

    name = "SDSS"
    _description = "{:} sample selection bit mask. Select the main galaxy "
    _description += "sample by (mask & 2), the BOSS LOWZ sample by "
    _description += "(mask & 4), the CMASS sample by (mask & 8), the QSO "
    _description += "sample by (mask & 16) and the full sample by (mask & 1)."

    def MAIN_selection(self, mag_r):
        ###############################################################
        #   based on Strauss+02                                       #
        ###############################################################
        bitmask = np.where(mag_r < 17.7, 2, 0).astype(self.dtype)  # r_pet~17.5
        return bitmask

    def BOSS_selection(self, mag_g, mag_r, mag_i):
        ###############################################################
        #   based on                                                  #
        #   http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php    #
        #   The selection has changed slightly compared to Dawson+13  #
        ###############################################################
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_g) < 90.0) & (np.abs(mag_r) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_i) < 90.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (mag_g - mag_r) + 1.2 * (mag_r - mag_i - 0.18)
        c_r = (mag_r - mag_i) - (mag_g - mag_r) / 4.0 - 0.18
        d_r = (mag_r - mag_i) - (mag_g - mag_r) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = colour_mask & (
            (mag_r > 16.0) &
            (mag_r < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (mag_r < 13.35 + c_p / 0.3))  # 13.5, 0.3
        bitmask = np.where(low_z, 4, 0).astype(self.dtype)
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 20.1) &  # 19.9
            (d_r > 0.55) &
            (mag_i < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((mag_r - mag_i) < 2.0))
        bitmask = np.bitwise_or(
            np.where(cmass, 8, 0).astype(self.dtype),
            bitmask)
        return bitmask

    def QSO_selection(self, is_central, lmhalo, lmstellar):
        """
        Method create a fake MICE2 quasar sample. Quasars do not exists in
        MICE2, therefore the assumption is made that quasars sit in the central
        galaxies of the most massive halos. This approach is only justifed by
        compairing the tails of the mock and data n(z) for the combined
        SDSS main, SDSS BOSS and SDSS QSO samples.
        """
        qso = is_central & (lmstellar > 11.2) & (lmhalo > 13.3)
        bitmask = np.where(qso, 16, 0).astype(self.dtype)
        return bitmask

    def __call__(
            self, mag_g, mag_r, mag_i, is_central=None, lmhalo=None,
            lmstellar=None):
        bitmask = self.MAIN_selection(mag_r)
        bitmask = np.bitwise_or(
            self.BOSS_selection(mag_g, mag_r, mag_i),
            bitmask)
        if all(val is not None for val in [is_central, lmhalo, lmstellar]):
            bitmask = np.bitwise_or(
                self.QSO_selection(is_central, lmhalo, lmstellar),
                bitmask)
            # for convenience flag all objects that pass the selection
            bitmask = np.bitwise_or(
                # join conditions with OR
                np.where(np.bitwise_and(bitmask, 30), 1, 0).astype(self.dtype),
                bitmask)
        else:
            # for convenience flag all objects that pass the selection
            bitmask = np.bitwise_or(
                # join conditions with OR
                np.where(np.bitwise_and(bitmask, 14), 1, 0).astype(self.dtype),
                bitmask)
        return bitmask


class SelectWiggleZ(BaseSelection):

    name = "WiggleZ"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "the photometric selection by (mask & 2), objects passing "
    _description += "the redshift weighted density sampling by (mask & 4) "
    _description += "and the full sample by (mask & 1)."

    def colour_selection(self, mag_g, mag_r, mag_i, mag_Z):
        ###############################################################
        #   based on Drinkwater+10                                    #
        ###############################################################
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_g) < 90.0) & (np.abs(mag_r) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_i) < 90.0)
        colour_mask &= (np.abs(mag_g) < 90.0) & (np.abs(mag_i) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_Z) < 90.0)
        # photometric cuts
        # we cannot reproduce the FUV, NUV, S/N and position matching cuts
        include = (
            (mag_r > 20.0) &
            (mag_r < 22.5))
        exclude = (
            (mag_g < 22.5) &
            (mag_i < 21.5) &
            (mag_r-mag_i < (mag_g-mag_r - 0.1)) &
            (mag_r-mag_i < 0.4) &
            (mag_g-mag_r > 0.6) &
            (mag_r-mag_Z < 0.7 * (mag_g-mag_r)))
        bitmask = np.where(include & ~exclude, 2, 0).astype(self.dtype)
        return bitmask

    def __call__(self, redshift, mag_g, mag_r, mag_i, mag_Z):
        bitmask = self.colour_selection(mag_g, mag_r, mag_i, mag_Z)
        bitmask = self._sampler.update_bitmask(redshift, bitmask, 4)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


class SelectDEEP2(BaseSelection):

    name = "DEEP2"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) objects the density "
    _description += "sampling by (mask & 8) and the full sample by (mask & 1)."

    def __init__(self, path, mock_redshifts, mock_area):
        super().__init__(path, mock_redshifts, mock_area)
        # Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        # Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        # with the first and last bin centered on 19 and 24.
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        # paper has given 1 - [sucess rate] in the histogram
        success_R_rate = np.loadtxt(os.path.join(
            SUCCESS_RATE_DIR, "DEEP2_success"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        self._p_success_R = interp1d(
            success_R_centers, success_R_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_R_rate[0], 0.0))

    def colour_selection(self, mag_B, mag_Rc, mag_Ic):
        ###############################################################
        #   based on Newman+13                                        #
        ###############################################################
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_B) < 90.0) & (np.abs(mag_Rc) < 90.0)
        colour_mask &= (np.abs(mag_Rc) < 90.0) & (np.abs(mag_Ic) < 90.0)
        # this modified selection gives the best match to the data n(z) with
        # its cut at z~0.75 and the B-R/R-I distribution (Newman+13, Fig. 12)
        # NOTE: We cannot apply the surface brightness cut and do not apply the
        #       Gaussian weighted sampling near the original colour cuts.
        mask = colour_mask & (
            (mag_Rc > 18.5) &
            (mag_Rc < 24.0) & (  # 24.1
                #                 2.45,                     0.2976
                (mag_B - mag_Rc < 2.0 * (mag_Rc - mag_Ic) - 0.4) |
                (mag_Rc - mag_Ic > 1.1) |
                (mag_B - mag_Rc < 0.2)))  # 0.5
        bitmask = np.where(mask, 2, 0).astype(self.dtype)
        return bitmask

    def specz_success(self, mag_Rc):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Rc))
        mask = random_draw < self._p_success_R(mag_Rc)
        bitmask = np.where(mask, 4, 0).astype(self.dtype)
        return bitmask

    def __call__(self, redshift, mag_B, mag_Rc, mag_Ic):
        bitmask = self.colour_selection(mag_B, mag_Rc, mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_Rc),
            bitmask)
        bitmask = self._sampler.update_bitmask(redshift, bitmask, 8)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 14, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


class SelectVVDSf02(BaseSelection):

    name = "VVDS-02h"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) objects the density "
    _description += "sampling by (mask & 8) and the full sample by (mask & 1)."

    def __init__(self, mock_redshifts, mock_area):
        super().__init__(mock_redshifts, mock_area)
        # NOTE: We use a redshift-based and I-band based success rate
        #       independently here since we do not know their correlation,
        #       which makes the success rate worse than in reality.
        # Spec-z success rate as function of i_AB read of Figure 16 in
        # LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        # 0.5 mag with the first starting at 17 and the last bin ending at 24.
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_I_rate = np.loadtxt(os.path.join(
                SUCCESS_RATE_DIR, "VVDSf02_I_success"))
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        self._p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+13 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        # NOTE: at z > 1.75 there are only lower limits (due to a lack of
        # spec-z?), thus the success rate is extrapolated as 1.0 at z > 1.75
        success_z_bright_centers, success_z_bright_rate = np.loadtxt(
            os.path.join(SUCCESS_RATE_DIR, "VVDSf02_z_bright_success")).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(SUCCESS_RATE_DIR, "VVDSf02_z_deep_success")).T
        # interpolate the success rates as probability of being selected with
        # the probability in the bright bin at z > 1.75 being 1.0 and the deep
        # bin at z > 4.0 being 0.0
        self._p_success_z_bright = interp1d(
            success_z_bright_centers, success_z_bright_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_bright_rate[0], 1.0))
        self._p_success_z_deep = interp1d(
            success_z_deep_centers, success_z_deep_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_deep_rate[0], 0.0))

    def colour_selection(self, mag_Ic):
        ###############################################################
        #   based on LeFèvre+05                                       #
        ###############################################################
        mask = (mag_Ic > 18.5) & (mag_Ic < 24.0)  # 17.5, 24.0
        # NOTE: The oversight of 1.0 magnitudes on the bright end misses 0.2 %
        #       of galaxies.
        # update the internal state
        bitmask = np.where(mask, 2, 0).astype(self.dtype)
        return bitmask

    def specz_success(self, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        mask = random_draw < self._p_success_I(mag_Ic)
        iterator = zip(
            [mag_Ic <= 22.5, mag_Ic > 22.5],
            [self._p_success_z_bright, self._p_success_z_deep])
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(redshift[m])
        bitmask = np.where(mask, 4, 0).astype(self.dtype)
        return bitmask

    def __call__(self, redshift, mag_Ic):
        bitmask = self.colour_selection(mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_Ic, redshift),
            bitmask)
        bitmask = self._sampler.update_bitmask(redshift, bitmask, 8)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 14, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


class SelectzCOSMOS(BaseSelection):

    name = "zCOSMOS"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) objects the density "
    _description += "sampling by (mask & 8) and the full sample by (mask & 1)."

    def __init__(self, mock_redshifts, mock_area):
        super().__init__(mock_redshifts, mock_area)
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        pickle_file = os.path.join(SUCCESS_RATE_DIR, "zCOSMOS.cache")
        if not os.path.exists(pickle_file):
            x = np.loadtxt(os.path.join(
                SUCCESS_RATE_DIR, "zCOSMOS_z_sampling"))
            y = np.loadtxt(os.path.join(
                SUCCESS_RATE_DIR, "zCOSMOS_I_sampling"))
            rates = np.loadtxt(os.path.join(
                SUCCESS_RATE_DIR, "zCOSMOS_success"))
            self._p_success_zI = interp2d(
                x, y, rates, copy=True, kind="linear")
            with open(pickle_file, "wb") as f:
                pickle.dump(self._p_success_zI, f)
        else:
            with open(pickle_file, "rb") as f:
                self._p_success_zI = pickle.load(f)

    def colour_selection(self, mag_Ic):
        ###############################################################
        #   based on Lilly+09                                         #
        ###############################################################
        mask = (mag_Ic > 15.0) & (mag_Ic < 22.5)  # 15.0, 22.5
        # NOTE: This only includes zCOSMOS bright.
        bitmask = np.where(mask, 2, 0).astype(self.dtype)
        return bitmask

    def specz_success(self, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        object_rates = np.empty_like(random_draw)
        for i, (z, I_AB) in enumerate(zip(redshift, mag_Ic)):
            # this must be in a loop since interp2d will create a grid from the
            # input redshifts and magnitudes instead of evaluating pairs of
            # values
            object_rates[i] = self._p_success_zI(z, I_AB)
        mask = random_draw < object_rates
        bitmask = np.where(mask, 4, 0).astype(self.dtype)
        return bitmask

    def __call__(self, redshift, mag_Ic):
        bitmask = self.colour_selection(mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_Ic, redshift),
            bitmask)
        bitmask = self._sampler.update_bitmask(redshift, bitmask, 8)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 14, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


###############################################################################
#                            TEST SAMPLES                                     #
###############################################################################


class SelectSparse24mag(BaseSelection):

    name = "Sparse24mag"
    _description = "{:} sample selection bit mask. Select objects passing "
    _description += "the photometric selection by (mask & 2), objects passing "
    _description += "the density sampling by (mask & 4) and the full sample "
    _description += "by (mask & 1)."

    def __call__(self, redshift, mag_r):
        bitmask = np.where(mag_r < 24.0, 2, 0).astype(self.dtype)
        bitmask = self._sampler.update_bitmask(redshift, bitmask, 4)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


###############################################################################


REGISTERED_SAMPLES = OrderedDict()
for name, selector in OrderedDict(locals()).items():
    if name.startswith("Select"):
        REGISTERED_SAMPLES[name[6:]] = selector
# find all samples that provide a density file
IMPLEMENT_DOWNSAMPLING = OrderedDict()
for name in REGISTERED_SAMPLES:
    density_file = DENSITY_FILE_TEMPLATE.format(name)
    if os.path.exists(density_file):
        IMPLEMENT_DOWNSAMPLING[name] = density_file
