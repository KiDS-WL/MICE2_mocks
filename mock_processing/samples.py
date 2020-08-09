import json
import os
import pickle
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d, interp2d

from .core.utils import ProgressBar
from .matching import DistributionEstimator


# folder containing data for the spec. success rate for the deep spec-z samples
SUCCESS_RATE_DIR = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "sample_data")
# files that provide (redshift dependent) sample density
DENSITY_FILE_TEMPLATE = os.path.join(SUCCESS_RATE_DIR, "{:}.json")


class BitMaskHelper(object):

    _bit_type_map = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _base_description = "{:} sample selection bit mask, select (sub-)samples "
    _base_description += "by (mask & bit), where the bits represent: "

    def __init__(self, sample, nbits=8):
        if nbits not in self._bit_type_map.keys():
            message = "number of bits must be either of: {:}".format(
                sorted(self._bit_type_map.keys))
            raise ValueError(message)
        self._sample = sample
        self._bit_desc = OrderedDict()
        # indicate which bits are occupied
        self._bit_reserv = OrderedDict(
            (2**i, False) for i in reversed(range(nbits)))
        # reserve the lowest bit for easy selection of the full sample
        self._bit_reserv[1] = True
        self._bit_desc[1] = "full sample"

    def __str__(self):
        bitstring = "".join(
            "1" if reserved else "0" for reserved in self._bit_reserv.values())
        return "BitMask({:})".format(bitstring)

    @property
    def dtype(self):
        return self._bit_type_map[len(self._bit_reserv)]

    @property
    def available(self):
        return tuple(
            bit for bit, reserved in self._bit_reserv.items() if not reserved)

    @property
    def reserved(self):
        return tuple(
            bit for bit, reserved in self._bit_reserv.items() if reserved)

    @property
    def description(self):
        bit_descs = []
        for bit in self.reserved:
            bit_descs.append("({:d}) {:}".format(bit, self._bit_desc[bit]))
        # build the documentation string
        string = "{:} sample selection bit mask, select ".format(self._sample)
        string += "(sub-)samples by (mask & bit), where the bits represent: "
        string += ", ".join(reversed(bit_descs))
        return string

    def reserve(self, description):
        if all(self._bit_reserv.values()):
            raise ValueError("all bits are reserved")
        # get the lowest, unreserved bit
        bit = self.available[-1]
        self.reserve_bit(bit, description)
        return bit

    def reserve_bit(self, bit, description):
        if all(self._bit_reserv.values()):
            raise ValueError("all bits are reserved")
        try:
            if self._bit_reserv[bit]:
                raise ValueError("bit is already reserved: {:d}".format(bit))
            else:
                self._bit_reserv[bit] = True
                self._bit_desc[bit] = description
        except KeyError:
            raise KeyError("invalid bit value: {:}".format(bit))


def update_bitmask(bitmask, bit_value, condition, copy=False):
    is_true = bitmask.dtype.type(bit_value)
    is_false = bitmask.dtype.type(0)
    bits = np.where(condition, is_true, is_false)
    if copy:
        return bitmask | bits
    else:
        bitmask |= bits
        return bitmask


def set_master(bitmask, values, copy=False):
    # action:   clear bit 1              set new value
    new_mask = (bitmask & ~np.uint8(1)) | values
    if copy:
        return new_mask
    else:  # change in place
        bitmask[:] = new_mask
        return bitmask


def join_AND_update_master(bitmask, bit_sum, copy=False):
    is_true = bitmask.dtype.type(1)  # write in first bit
    is_false = bitmask.dtype.type(0)
    # if we join with AND we want all bits to match, ignoring the selection bit
    condition = (bitmask & bit_sum) == bit_sum
    bits_new = np.where(condition, is_true, is_false)  # new selection bits
    # join the new selection bit with the existing one using AND logic
    bits_old = (bitmask & 1)
    return set_master(bitmask, bits_new & bits_old, copy=copy)


def join_OR_update_master(bitmask, bit_sum, copy=False):
    is_true = bitmask.dtype.type(1)  # write in first bit
    is_false = bitmask.dtype.type(0)
    # if we join with OR we want any bit to match, ignoring the selection bit
    condition = (bitmask & bit_sum) > 0  # only true if any bit matches
    bits_new = np.where(condition, is_true, is_false)  # new selection bits
    # join the new selection bit with the existing one using AND logic
    bits_old = (bitmask & 1)
    return set_master(bitmask, bits_new & bits_old, copy=copy)


class DensitySampler(object):

    name = "Generic Sample"

    def __init__(self, bitvalues, path, area, bitmask):
        self._bit_values = bitvalues
        # load sample density in deg^-2
        with open(path) as f:
            self._sample_dens = json.load(f)["density"]
        # mock density in deg^-2
        self._mock_dens = self._count_selected(bitmask) / area
        # check the density values
        if self.sample_density >= self.mock_density:
            message = "sample density must be smaller than mock density"
            raise ValueError(message)
        print("densities:", self.sample_density, self.mock_density, len(bitmask) / area)

    def _count_selected(self, bitmask):
        # count the objects that pass the current selection, marked by the
        # selection bit (bit 1)
        n_mocks = 0
        chunksize = 16384
        pbar = ProgressBar(len(bitmask), "estimate mock density")
        for start in range(0, len(bitmask), chunksize):
            end = min(start + chunksize, len(bitmask))
            n_mocks += np.sum(bitmask[start:end] & 1)  # 1 if match, else 0
            pbar.update(end - start)
        pbar.close()
        return n_mocks

    @property
    def sample_density(self):
        return self._sample_dens

    @property
    def mock_density(self):
        return self._mock_dens

    def odds(self):
        return self._sample_dens / self._mock_dens

    def mask(self, n_points):
        random_draw = np.random.rand(n_points)
        mask = random_draw < self.odds()
        return mask

    def __call__(self, bitmask):
        mask = self.mask(len(bitmask))
        update_bitmask(bitmask, self._bit_values[0], mask)
        join_AND_update_master(bitmask, sum(self._bit_values))


class RedshiftSampler(object):

    name = "Generic Sample"

    def __init__(self, bitvalues, path, area, bitmask, redshifts):
        self._bit_values = bitvalues
        # load sample density in deg^-2 per redshift
        self._sample_dens = DistributionEstimator.from_file(path)
        # mock density per redshift
        self._mock_dens = self._get_redshift_distribution(bitmask, redshifts)
        self._mock_dens.normalisation = area  # yields deg^-2 per redshift
        # check the density values
        if self.sample_density >= self.mock_density:
            message = "sample density must be smaller than mock density"
            raise ValueError(message)
        print("densities:", self.sample_density, self.mock_density, len(bitmask) / area)

    def _get_redshift_distribution(self, bitmask, redshifts):
        density = DistributionEstimator(
            self._sample_dens._xmin, self._sample_dens._xmax,
            self._sample_dens._width, self._sample_dens._smooth,
            "linear", fill_value=0.0)
        # build an interpolated, average shifted histogram of the mock redshift
        # distribution considering only objects that pass the selection
        chunksize = 16384
        pbar = ProgressBar(len(bitmask), "estimate mock n(z)")
        for start in range(0, len(bitmask), chunksize):
            end = min(start + chunksize, len(bitmask))
            mask = (bitmask[start:end] & 1) == 1  # 1 if match, else 0
            density.add_data(redshifts[start:end][mask])
            pbar.update(end - start)
        density.interpolate()
        pbar.close()
        return density

    @property
    def sample_density(self):
        return self._sample_dens.normalisation

    @property
    def mock_density(self):
        return self._mock_dens.normalisation

    def odds(self, redshift):
        mock_density = self._mock_dens(redshift)
        # divisions by zero are likely and must be substituted
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.where(
                mock_density == 0.0, 0.0,
                self._sample_dens(redshift) / mock_density)
        return odds

    def draw(self, redshift):
        random_draw = np.random.rand(len(redshift))
        mask = random_draw < self.odds(redshift)
        return mask

    def __call__(self, bitmask, redshift):
        mask = self.draw(redshift)
        update_bitmask(bitmask, self._bit_values[0], mask)
        join_AND_update_master(bitmask, sum(self._bit_values))


class BaseSelection(object):

    name = "Generic Sample"

    def __init__(self, bitvalues):
        self._bit_values = bitvalues

    @property
    def __name__(self):
        return self.__class__.__name__


class SelectKiDS(BaseSelection):

    name = "KiDS"
    bit_descriptions = ("non-zero lensfit weight", "BPZ prior band detection")

    def lensing_selection(self, bitmask, recal_weight, prior_magnitude):
        # select objects with non-zero lensfit weight
        update_bitmask(bitmask, self._bit_values[0], recal_weight > 0.0)
        # select objects that are detected in the BPZ prior band
        update_bitmask(bitmask, self._bit_values[1], prior_magnitude < 90.0)

    def __call__(self, bitmask, recal_weight, prior_magnitude):
        self.lensing_selection(bitmask, recal_weight, prior_magnitude)
        join_AND_update_master(bitmask, sum(self._bit_values))


###############################################################################
#                            WIDE SURVEYS                                     #
###############################################################################


class Select2dFLenS(BaseSelection):

    name = "2dFLenS"
    bit_descriptions = ("LOWZ", "MIDZ", "HIGHZ")

    def colour_selection(self, bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
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
        update_bitmask(bitmask, self._bit_values[0], low_z1 | low_z2 | low_z3)
        # defining the MIDZ sample
        mid_z = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            ((mag_r - mag_i) < 2.0) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.9)))  # 19.86, 1.6, 0.8
        update_bitmask(bitmask, self._bit_values[1], mid_z)
        # defining the HIGHZ sample
        high_z = colour_mask & (
            (mag_Z < 19.9) &  # 19.95
            (mag_i > 19.9) &
            (mag_i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((mag_r - mag_Ks) > 1.9 * (mag_r - mag_i)) &
            ((mag_r - mag_i) > 0.98) &
            ((mag_i - mag_Z) > 0.6))
        update_bitmask(bitmask, self._bit_values[2], high_z)

    def __call__(self, bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        self.colour_selection(bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks)
        join_OR_update_master(bitmask, sum(self._bit_values))


class SelectGAMA(BaseSelection):

    name = "GAMA"
    bit_descriptions = ("r-band cut",)

    def __call__(self, bitmask, mag_r):
        ###############################################################
        #   based on Driver+11                                        #
        ###############################################################
        update_bitmask(bitmask, self._bit_values[0], mag_r < 19.87)
        join_AND_update_master(bitmask, sum(self._bit_values))


class SelectSDSS(BaseSelection):

    name = "SDSS"
    bit_descriptions = ("main sample", "BOSS LOWZ", "BOSS CMASS", "QSO")

    def MAIN_selection(self, bitmask, mag_r):
        ###############################################################
        #   based on Strauss+02                                       #
        ###############################################################
        update_bitmask(
            bitmask, self._bit_values[0], mag_r < 17.7)  # r_pet ~ 17.5

    def BOSS_selection(self, bitmask, mag_g, mag_r, mag_i):
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
        update_bitmask(bitmask, self._bit_values[1], low_z)
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 20.1) &  # 19.9
            (d_r > 0.55) &
            (mag_i < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((mag_r - mag_i) < 2.0))
        update_bitmask(bitmask, self._bit_values[2], cmass)

    def QSO_selection(self, bitmask, is_central, lmhalo, lmstellar):
        """
        Method create a fake MICE2 quasar sample. Quasars do not exists in
        MICE2, therefore the assumption is made that quasars sit in the central
        galaxies of the most massive halos. This approach is only justifed by
        compairing the tails of the mock and data n(z) for the combined
        SDSS main, SDSS BOSS and SDSS QSO samples.
        """
        qso = is_central & (lmstellar > 11.2) & (lmhalo > 13.3)
        update_bitmask(bitmask, self._bit_values[3], qso)

    def __call__(
            self, bitmask, mag_g, mag_r, mag_i, is_central=None, lmhalo=None,
            lmstellar=None):
        self.MAIN_selection(bitmask, mag_r)
        self.BOSS_selection(bitmask, mag_g, mag_r, mag_i)
        if all(val is not None for val in [is_central, lmhalo, lmstellar]):
                self.QSO_selection(bitmask, is_central, lmhalo, lmstellar)
        join_OR_update_master(bitmask, sum(self._bit_values))


class SelectWiggleZ(BaseSelection):

    name = "WiggleZ"
    bit_descriptions = ("inclusion rules", "exclusion rules")

    def colour_selection(self, bitmask, mag_g, mag_r, mag_i, mag_Z):
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
        include = colour_mask & (
            (mag_r > 20.0) &
            (mag_r < 22.5))
        update_bitmask(bitmask, self._bit_values[0], include)
        exclude = colour_mask & (
            (mag_g < 22.5) &
            (mag_i < 21.5) &
            (mag_r-mag_i < (mag_g-mag_r - 0.1)) &
            (mag_r-mag_i < 0.4) &
            (mag_g-mag_r > 0.6) &
            (mag_r-mag_Z < 0.7 * (mag_g-mag_r)))
        update_bitmask(bitmask, self._bit_values[1], ~exclude)

    def __call__(self, bitmask, redshift, mag_g, mag_r, mag_i, mag_Z):
        self.colour_selection(bitmask, mag_g, mag_r, mag_i, mag_Z)
        join_AND_update_master(bitmask, sum(self._bit_values))


class SampleWiggleZ(RedshiftSampler):

    name = "WiggleZ"

    def __init__(self, bitvalues, mock_area, bitmask, redshifts):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(
            bitvalues, density_file, mock_area, bitmask, redshifts)


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


class SelectDEEP2(BaseSelection):

    name = "DEEP2"
    bit_descriptions = ("colour/magnitude selection", "spectroscopic success")

    def __init__(self, bitvalues):
        super().__init__(bitvalues)
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

    def colour_selection(self, bitmask, mag_B, mag_Rc, mag_Ic):
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
        selection = colour_mask & (
            (mag_Rc > 18.5) &
            (mag_Rc < 24.0) & (  # 24.1
                #                 2.45,                     0.2976
                (mag_B - mag_Rc < 2.0 * (mag_Rc - mag_Ic) - 0.4) |
                (mag_Rc - mag_Ic > 1.1) |
                (mag_B - mag_Rc < 0.2)))  # 0.5
        update_bitmask(bitmask, self._bit_values[0], selection)

    def specz_success(self, bitmask, mag_Rc):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Rc))
        success = random_draw < self._p_success_R(mag_Rc)
        update_bitmask(bitmask, self._bit_values[1], success)

    def __call__(self, bitmask, mag_B, mag_Rc, mag_Ic):
        self.colour_selection(bitmask, mag_B, mag_Rc, mag_Ic)
        self.specz_success(bitmask, mag_Rc)
        join_AND_update_master(bitmask, sum(self._bit_values))


class SampleDEEP2(DensitySampler):

    name = "DEEP2"

    def __init__(self, bitvalues, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bitvalues, density_file, mock_area, bitmask)


class SelectVVDSf02(BaseSelection):

    name = "VVDSf02"
    bit_descriptions = ("magnitude selection", "spectroscopic success")

    def __init__(self, bitvalues):
        super().__init__(bitvalues)
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

    def colour_selection(self, bitmask, mag_Ic):
        ###############################################################
        #   based on LeFèvre+05                                       #
        ###############################################################
        selection = (mag_Ic > 17.5) & (mag_Ic < 24.0)  # 17.5, 24.0
        update_bitmask(bitmask, self._bit_values[0], selection)

    def specz_success(self, bitmask, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        success = random_draw < self._p_success_I(mag_Ic)
        iterator = zip(
            [mag_Ic <= 22.5, mag_Ic > 22.5],
            [self._p_success_z_bright, self._p_success_z_deep])
        for m, p_success_z in iterator:
            success[m] &= random_draw[m] < p_success_z(redshift[m])
        update_bitmask(bitmask, self._bit_values[1], success)

    def __call__(self, bitmask, redshift, mag_Ic):
        self.colour_selection(bitmask, mag_Ic)
        self.specz_success(bitmask, mag_Ic, redshift)
        join_AND_update_master(bitmask, sum(self._bit_values))


class SampleVVDSf02(DensitySampler):

    name = "VVDSf02"

    def __init__(self, bitvalues, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bitvalues, density_file, mock_area, bitmask)


class SelectzCOSMOS(BaseSelection):

    name = "zCOSMOS"
    bit_descriptions = ("magnitude selection", "spectroscopic success")

    def __init__(self, bitvalues):
        super().__init__(bitvalues)
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

    def colour_selection(self, bitmask, mag_Ic):
        ###############################################################
        #   based on Lilly+09                                         #
        ###############################################################
        # NOTE: This only includes zCOSMOS bright.
        selection = (mag_Ic > 15.0) & (mag_Ic < 22.5)  # 15.0, 22.5
        update_bitmask(bitmask, self._bit_values[0], selection)

    def specz_success(self, bitmask, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        object_rates = np.empty_like(random_draw)
        for i, (z, I_AB) in enumerate(zip(redshift, mag_Ic)):
            # this must be in a loop since interp2d will create a grid from the
            # input redshifts and magnitudes instead of evaluating pairs of
            # values
            object_rates[i] = self._p_success_zI(z, I_AB)
        success = random_draw < object_rates
        update_bitmask(bitmask, self._bit_values[1], success)

    def __call__(self, bitmask, redshift, mag_Ic):
        self.colour_selection(bitmask, mag_Ic)
        self.specz_success(bitmask, mag_Ic, redshift),
        join_AND_update_master(bitmask, sum(self._bit_values))


class SamplezCOSMOS(DensitySampler):

    name = "zCOSMOS"

    def __init__(self, bitvalues, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bitvalues, density_file, mock_area, bitmask)


###############################################################################
#                            TEST SAMPLES                                     #
###############################################################################


class SelectSparse24mag(BaseSelection):

    name = "Sparse24mag"
    bit_descriptions = ("magnitude selection")

    def __call__(self, bitmask, redshift, mag_r):
        update_bitmask(bitmask, self._bit_values[0], mag_r < 24.0)


class SampleSparse24mag(DensitySampler):

    name = "Sparse24mag"

    def __init__(self, bitvalues, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bitvalues, density_file, mock_area, bitmask)


###############################################################################
#   determine the implemented samples and if/what kind of density sampling    #
#   is implemented                                                            #
###############################################################################

PHOTOMETRIC_SELECTIONS = OrderedDict()
SAMPLING_SELECTIONS = OrderedDict()
for name, selector in OrderedDict(locals()).items():
    if name.startswith("Select"):
        PHOTOMETRIC_SELECTIONS[name[6:]] = selector
    elif name.startswith("Sample"):
        SAMPLING_SELECTIONS[name[6:]] = selector
# all known samples
REGISTERED_SAMPLES = tuple(sorted(
    tuple(PHOTOMETRIC_SELECTIONS.keys()) +
    tuple(SAMPLING_SELECTIONS.keys())))
