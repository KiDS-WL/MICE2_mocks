import json
import os
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d, interp2d

from .core.bitmask import BitMaskManager as BMM
from .core.config import (LineComment, Parameter, ParameterGroup,
                          ParameterCollection, Parser)
from .core.parallel import Schedule
from .core.utils import ProgressBar
from .matching import DistributionEstimator


# folder containing data for the spec. success rate for the deep spec-z samples
SUCCESS_RATE_DIR = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "sample_data")
# files that provide (redshift dependent) sample density
DENSITY_FILE_TEMPLATE = os.path.join(SUCCESS_RATE_DIR, "{:}.json")

# register all known selections here
PHOTOMETRIC_SELECTIONS = OrderedDict()
SAMPLING_SELECTIONS = OrderedDict()
SELECTION_PARSERS = OrderedDict()
REGISTERED_SAMPLES = set()


def register(selector):
    """
    Register selection function objects.
    """
    if issubclass(selector, BaseSelection):
        PHOTOMETRIC_SELECTIONS[selector.name] = selector
        REGISTERED_SAMPLES.update(PHOTOMETRIC_SELECTIONS.keys())
    elif issubclass(selector, Sampler):
        SAMPLING_SELECTIONS[selector.name] = selector
        REGISTERED_SAMPLES.update(PHOTOMETRIC_SELECTIONS.keys())
    elif issubclass(selector, Parser):
        SELECTION_PARSERS[selector.name] = selector
        REGISTERED_SAMPLES.update(PHOTOMETRIC_SELECTIONS.keys())
    else:
        message = "Object must be subclass of 'BaseSelection', 'Sampler' or "
        message += "'Parser'"
        raise TypeError(message)
    return selector


class DumpConfig(argparse.Action):

    def __init__(self, *args, nargs=1,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        sample = values[0]
        try:
            sample_parser = REGISTERED_SAMPLES[sample]
        except KeyError:
            message = "sample '{:}' does not implement a configuration"
            raise parser.error(message.format(sample))
        print(sample_parser.default)
        parser.exit()


def make_bitmask(sample):
    param = Parameter(
        "bitmask", str, "samples/" + sample,
        "path at which the sample selection bit mask is saved in the data "
        "store")
    return param


header = (
    "This configuration file is required for mocks_select_sample. It "
    "defines the output column name of the selection bit mask and the "
    "data columns for the selection functions (such as magnitudes or "
    "redshift).")
selection_header = (
    "Mapping form keyword argument names in selection function to "
    "column path in the data store. Optional arguments can be "
    "left blank.")
param_redshift = Parameter("redshift", str, "...", "(observed) redshifts")
param_sdss = {}
for key in ("g", "r", "i", "Z", "Ks"):
    param_sdss[key] = Parameter(
        "mag_" + key, str, "...", "SDSS {:}-band magnitude".format(key))
param_johnson = {}
for key in ("B", "Rc", "Ic"):
    param_johnson[key] = Parameter(
        "mag_" + key, str, "...", "Johnson {:}-band magnitude".format(key))


class Sampler(object):

    name = "Generic Sample"


class DensitySampler(Sampler):

    bit_descriptions = ("surface density downsampling",)

    def __init__(self, bit_manager, path, area, bitmask):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]
        # load sample density in deg^-2
        with open(path) as f:
            self._sample_dens = json.load(f)["density"]
        # mock density in deg^-2
        self._mock_dens = self.count_selected(bitmask) / area
        # check the density values
        if self.sample_density >= self.mock_density:
            message = "sample density must be smaller than mock density"
            raise ValueError(message)

    @staticmethod
    def count_selected(bitmask):
        # count the objects that pass the current selection, marked by the
        # selection bit (bit 1)
        n_mocks = 0
        chunksize = 16384
        pbar = ProgressBar(len(bitmask))
        for start in range(0, len(bitmask), chunksize):
            end = min(start + chunksize, len(bitmask))
            is_selected = BMM.check_master(bitmask[start:end])
            n_mocks += np.count_nonzero(is_selected)
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

    @Schedule.description("sampling surface density")
    @Schedule.workload(0.10)
    def apply(self, bitmask):
        is_selected = self.mask(len(bitmask))
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


class RedshiftSampler(Sampler):

    bit_descriptions = ("redshift weighted density downsampling",)

    def __init__(self, bit_manager, path, area, bitmask, redshifts):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]
        # load sample density in deg^-2 per redshift
        self._sample_dens = DistributionEstimator.from_file(path)
        # mock density per redshift in deg^-2 per redshift
        self._mock_dens = self.get_redshift_density(bitmask, redshifts, area)
        # check the density values
        if self.sample_density >= self.mock_density:
            message = "sample density must be smaller than mock density"
            raise ValueError(message)

    def get_redshift_density(self, bitmask, redshifts, area):
        density = DistributionEstimator(
            self._sample_dens._xmin, self._sample_dens._xmax,
            self._sample_dens._width, self._sample_dens._smooth,
            "linear", fill_value=0.0)
        # build an interpolated, average shifted histogram of the mock redshift
        # distribution considering only objects that pass the selection
        n_mocks = 0
        chunksize = 16384
        pbar = ProgressBar(len(bitmask))
        for start in range(0, len(bitmask), chunksize):
            end = min(start + chunksize, len(bitmask))
            is_selected = BMM.check_master(bitmask[start:end])
            n_mocks += np.count_nonzero(is_selected)
            density.add_data(redshifts[start:end][is_selected])
            pbar.update(end - start)
        pbar.close()
        density.interpolate()
        density.normalisation = 1.0 / area
        return density

    @property
    def sample_density(self):
        n_objects = self._sample_dens._counts.sum()
        return n_objects * self._sample_dens.normalisation

    @property
    def mock_density(self):
        n_objects = self._mock_dens._counts.sum()
        return n_objects * self._mock_dens.normalisation

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

    @Schedule.description("sampling redshift density")
    @Schedule.workload(0.20)
    def apply(self, bitmask, redshift):
        is_selected = self.draw(redshift)
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


class BaseSelection(object):

    name = "Generic Sample"

    def __init__(self, bit_manager):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]

    @property
    def __name__(self):
        return self.__class__.__name__


@register
class ParserKiDS(Parser):

    name = "KiDS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            Parameter(
                "recal_weight", str, "lensing/recal_weight",
                "path of lensing weight in the data store"),
            LineComment(
                "setting the following paramter will remove objects with "
                "non-detections"),
            Parameter(
                "prior_magnitude", str, None,
                "path of BPZ prior magnitude in the data store"),
            header=selection_header),
        header=header)


@register
class SelectKiDS(BaseSelection):

    name = "KiDS"
    bit_descriptions = ("non-zero lensfit weight", "BPZ prior band detection")

    def lensing_selection(self, bitmask, recal_weight, prior_magnitude):
        # select objects with non-zero lensfit weight
        is_selected = recal_weight > 0.0
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # select objects that are detected in the BPZ prior band
        is_selected = prior_magnitude < 90.0
        BMM.set_bit(bitmask, self._bits[1], condition=is_selected)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, recal_weight, prior_magnitude):
        self.lensing_selection(bitmask, recal_weight, prior_magnitude)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


###############################################################################
#                            WIDE SURVEYS                                     #
###############################################################################


@register
class Parser2dFLenS(Parser):

    name = "2dFLenS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            param_sdss["Z"],
            param_sdss["Ks"],
            header=selection_header),
        header=header)


@register
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
        lowz = low_z1 | low_z2 | low_z3
        BMM.set_bit(bitmask, self._bits[0], condition=lowz)
        # defining the MIDZ sample
        mid_z = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            ((mag_r - mag_i) < 2.0) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.9)))  # 19.86, 1.6, 0.8
        BMM.set_bit(bitmask, self._bits[1], condition=mid_z)
        # defining the HIGHZ sample
        high_z = colour_mask & (
            (mag_Z < 19.9) &  # 19.95
            (mag_i > 19.9) &
            (mag_i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((mag_r - mag_Ks) > 1.9 * (mag_r - mag_i)) &
            ((mag_r - mag_i) > 0.98) &
            ((mag_i - mag_Z) > 0.6))
        BMM.set_bit(bitmask, self._bits[2], condition=high_z)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        self.colour_selection(bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="OR")


@register
class ParserGAMA(Parser):

    name = "GAMA"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["r"],
            header=selection_header),
        header=header)


@register
class SelectGAMA(BaseSelection):

    name = "GAMA"
    bit_descriptions = ("r-band cut",)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_r):
        ###############################################################
        #   based on Driver+11                                        #
        ###############################################################
        is_selected = mag_r < 19.87
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits))


@register
class ParserSDSS(Parser):

    name = "SDSS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            Parameter(
                "is_central", str, "environ/is_central",
                "flag indicating if it is central host galaxy"),
            Parameter(
                "lmhalo", str, "environ/log_M_halo",
                "logarithmic halo mass"),
            Parameter(
                "lmstellar", str, "environ/log_M_stellar",
                "logarithmic stellar mass"),
            header=selection_header),
        header=header)


@register
class SelectSDSS(BaseSelection):

    name = "SDSS"
    bit_descriptions = ("main sample", "BOSS LOWZ", "BOSS CMASS", "QSO")

    def MAIN_selection(self, bitmask, mag_r):
        ###############################################################
        #   based on Strauss+02                                       #
        ###############################################################
        is_selected = mag_r < 17.7  # r_pet ~ 17.5
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

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
        lowz = colour_mask & (
            (mag_r > 16.0) &
            (mag_r < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (mag_r < 13.35 + c_p / 0.3))  # 13.5, 0.3
        BMM.set_bit(bitmask, self._bits[1], condition=lowz)
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = colour_mask & (
            (mag_i > 17.5) &
            (mag_i < 20.1) &  # 19.9
            (d_r > 0.55) &
            (mag_i < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((mag_r - mag_i) < 2.0))
        BMM.set_bit(bitmask, self._bits[2], condition=cmass)

    def QSO_selection(self, bitmask, is_central, lmhalo, lmstellar):
        """
        Method create a fake MICE2 quasar sample. Quasars do not exists in
        MICE2, therefore the assumption is made that quasars sit in the central
        galaxies of the most massive halos. This approach is only justifed by
        compairing the tails of the mock and data n(z) for the combined
        SDSS main, SDSS BOSS and SDSS QSO samples.
        """
        is_selected = is_central & (lmstellar > 11.2) & (lmhalo > 13.3)
        BMM.set_bit(bitmask, self._bits[3], condition=is_selected)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(
            self, bitmask, mag_g, mag_r, mag_i, is_central, lmhalo, lmstellar):
        self.MAIN_selection(bitmask, mag_r)
        self.BOSS_selection(bitmask, mag_g, mag_r, mag_i)
        self.QSO_selection(bitmask, is_central, lmhalo, lmstellar)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="OR")


@register
class ParserWiggleZ(Parser):

    name = "WiggleZ"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            param_sdss["Z"],
            header=selection_header),
        header=header)


@register
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
        BMM.set_bit(bitmask, self._bits[0], condition=include)
        exclude = colour_mask & (
            (mag_g < 22.5) &
            (mag_i < 21.5) &
            (mag_r-mag_i < (mag_g-mag_r - 0.1)) &
            (mag_r-mag_i < 0.4) &
            (mag_g-mag_r > 0.6) &
            (mag_r-mag_Z < 0.7 * (mag_g-mag_r)))
        BMM.set_bit(bitmask, self._bits[1], condition=~exclude)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, redshift, mag_g, mag_r, mag_i, mag_Z):
        self.colour_selection(bitmask, mag_g, mag_r, mag_i, mag_Z)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


@register
class SampleWiggleZ(RedshiftSampler):

    name = "WiggleZ"

    def __init__(self, bit_manager, mock_area, bitmask, redshifts):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(
            bit_manager, density_file, mock_area, bitmask, redshifts)


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


@register
class ParserDEEP2(Parser):

    name = "DEEP2"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_johnson["B"],
            param_johnson["Rc"],
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
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
        is_selected = colour_mask & (
            (mag_Rc > 18.5) &
            (mag_Rc < 24.0) & (  # 24.1
                #                 2.45,                     0.2976
                (mag_B - mag_Rc < 2.0 * (mag_Rc - mag_Ic) - 0.4) |
                (mag_Rc - mag_Ic > 1.1) |
                (mag_B - mag_Rc < 0.2)))  # 0.5
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

    def specz_success(self, bitmask, mag_Rc):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Rc))
        is_selected = random_draw < self._p_success_R(mag_Rc)
        BMM.set_bit(bitmask, self._bits[1], condition=is_selected)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.25)
    def apply(self, bitmask, mag_B, mag_Rc, mag_Ic):
        self.colour_selection(bitmask, mag_B, mag_Rc, mag_Ic)
        self.specz_success(bitmask, mag_Rc)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


@register
class SampleDEEP2(DensitySampler):

    name = "DEEP2"

    def __init__(self, bit_manager, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bit_manager, density_file, mock_area, bitmask)


@register
class ParserVVDSf02(Parser):

    name = "VVDSf02"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
class SelectVVDSf02(BaseSelection):

    name = "VVDSf02"
    bit_descriptions = ("magnitude selection", "spectroscopic success")

    def __init__(self, bit_manager):
        super().__init__(bit_manager)
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
        is_selected = (mag_Ic > 17.5) & (mag_Ic < 24.0)  # 17.5, 24.0
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

    def specz_success(self, bitmask, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        is_selected = random_draw < self._p_success_I(mag_Ic)
        iterator = zip(
            [mag_Ic <= 22.5, mag_Ic > 22.5],
            [self._p_success_z_bright, self._p_success_z_deep])
        for mask, p_success_z in iterator:
            is_selected[mask] &= \
                random_draw[mask] < p_success_z(redshift[mask])
        BMM.set_bit(bitmask, self._bits[1], condition=is_selected)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.25)
    def apply(self, bitmask, redshift, mag_Ic):
        self.colour_selection(bitmask, mag_Ic)
        self.specz_success(bitmask, mag_Ic, redshift)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


@register
class SampleVVDSf02(DensitySampler):

    name = "VVDSf02"

    def __init__(self, bit_manager, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bit_manager, density_file, mock_area, bitmask)


@register
class ParserzCOSMOS(Parser):

    name = "zCOSMOS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
class SelectzCOSMOS(BaseSelection):

    name = "zCOSMOS"
    bit_descriptions = ("magnitude selection", "spectroscopic success")

    def __init__(self, bit_manager):
        super().__init__(bit_manager)
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        x = np.loadtxt(os.path.join(
            SUCCESS_RATE_DIR, "zCOSMOS_z_sampling"))
        y = np.loadtxt(os.path.join(
            SUCCESS_RATE_DIR, "zCOSMOS_I_sampling"))
        rates = np.loadtxt(os.path.join(
            SUCCESS_RATE_DIR, "zCOSMOS_success"))
        self._p_success_zI = interp2d(
            x, y, rates, copy=True, kind="linear")

    def colour_selection(self, bitmask, mag_Ic):
        ###############################################################
        #   based on Lilly+09                                         #
        ###############################################################
        # NOTE: This only includes zCOSMOS bright.
        is_selected = (mag_Ic > 15.0) & (mag_Ic < 22.5)  # 15.0, 22.5
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

    def specz_success(self, bitmask, mag_Ic, redshift):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_Ic))
        object_rates = np.empty_like(random_draw)
        for i, (z, I_AB) in enumerate(zip(redshift, mag_Ic)):
            # this must be in a loop since interp2d will create a grid from the
            # input redshifts and magnitudes instead of evaluating pairs of
            # values
            object_rates[i] = self._p_success_zI(z, I_AB)
        is_selected = random_draw < object_rates
        BMM.set_bit(bitmask, self._bits[1], condition=is_selected)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.CPUbound
    def apply(self, bitmask, redshift, mag_Ic):
        self.colour_selection(bitmask, mag_Ic)
        self.specz_success(bitmask, mag_Ic, redshift),
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


@register
class SamplezCOSMOS(DensitySampler):

    name = "zCOSMOS"

    def __init__(self, bit_manager, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bit_manager, density_file, mock_area, bitmask)


###############################################################################
#                            TEST SAMPLES                                     #
###############################################################################


@register
class ParserSparse24mag(Parser):

    name = "Sparse24mag"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["r"],
            header=selection_header),
        header=header)


@register
class SelectSparse24mag(BaseSelection):

    name = "Sparse24mag"
    bit_descriptions = ("magnitude selection")

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_r):
        is_selected = mag_r < 24.0
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits))


@register
class SampleSparse24mag(DensitySampler):

    name = "Sparse24mag"

    def __init__(self, bit_manager, mock_area, bitmask):
        density_file = DENSITY_FILE_TEMPLATE.format(self.name)
        super().__init__(bit_manager, density_file, mock_area, bitmask)
