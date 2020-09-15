#
# This module implements the literature (reference) selection functions and
# density samplers for different galaxy samples. The data columns in the data
# store, required to select these samples, must be configured by a
# configuration file. Therefore, each sample should define at least the first
# two of these classes:
#     - Parser_[sample_name]: Implements a ParameterCollection that defines the
#       configuration file.
#     - Select_[sample_name]: Implements the sample selection, such as
#       photometric cuts or spectroscopic success rates.
#     - Sample_[sample_name]: Optional density sampling, may be redshift
#       weighted.
# 
# The naming convention is required to associate all classes belonging to one
# sample. All these sample classes must be registered with
# @SampleManager.register.
#
# Modifications can be implemented for specific input mock catalogues
# (flavours), see MICE2.py for reference.
#
# To obtain an empty, default configuration for any of the samples, import the
# parser in python and type
#     print(Parser_[sample_name].default)
#

import os

import numpy as np
from scipy.interpolate import interp1d, interp2d

from galmock.core.bitmask import BitMaskManager as BMM
from galmock.core.config import (LineComment, Parameter, ParameterCollection,
                                 ParameterGroup, ParameterListing, Parser)
from galmock.core.parallel import Schedule

from galmock.samples import config
from galmock.samples.base import (BaseSelection, DensitySampler,
                                  RedshiftSampler, SampleManager)


######## KiDS ########

@SampleManager.register
class Parser_KiDS(Parser):

    name = "KiDS"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
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
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_KiDS(BaseSelection):

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


######## 2dFLenS (Blake+16) ########

@SampleManager.register
class Parser_2dFLenS(Parser):

    name = "2dFLenS"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_sdss["g"],
            config.param_sdss["r"],
            config.param_sdss["i"],
            config.param_sdss["Z"],
            config.param_sdss["Ks"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_2dFLenS(BaseSelection):

    name = "2dFLenS"
    bit_descriptions = ("LOWZ", "MIDZ", "HIGHZ")

    @staticmethod
    def colour_transformations(mag_g, mag_r, mag_i,):
        c_p = 0.7 * (mag_g - mag_r) + 1.2 * (mag_r - mag_i - 0.18)
        c_r = (mag_r - mag_i) - (mag_g - mag_r) / 4.0 - 0.18
        d_r = (mag_r - mag_i) - (mag_g - mag_r) / 8.0
        return c_p, c_r, d_r

    @staticmethod
    def LOWZ(mag_g, mag_r, mag_i, c_p, c_r):
        lowz_1 = (
            (mag_r > 16.0) &
            (mag_r < 19.2) &
            (mag_r < (13.1 + c_p / 0.3)) &
            (np.abs(c_r) < 0.2))
        lowz_2 = (
            (mag_r > 16.0) &
            (mag_r < 19.5) &
            (mag_g - mag_r > (1.3 + 0.25 * (mag_r - mag_i))) &
            (c_r > 0.45 - (mag_g - mag_r) / 6.0))
        lowz_3 = (
            (mag_r > 16.0) &
            (mag_r < 19.6) &
            (mag_r < (13.5 + c_p / 0.3)) &
            (np.abs(c_r) < 0.2))
        return lowz_1 | lowz_2 | lowz_3

    @staticmethod
    def MIDZ(mag_r, mag_i, d_r):
        midz = (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            ((mag_r - mag_i) < 2.0) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.8)))
        return midz

    @staticmethod
    def HIGHZ(mag_r, mag_i, mag_Z, mag_Ks):
        highz = (
            (mag_Z < 19.95) &
            (mag_i > 19.9) &
            (mag_i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((mag_r - mag_Ks) > 1.9 * (mag_r - mag_i)) &
            ((mag_r - mag_i) > 0.98) &
            ((mag_i - mag_Z) > 0.6))
        return highz

    def colour_selection(self, bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_g) < 90.0) & (np.abs(mag_r) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_i) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_Ks) < 90.0)
        colour_mask &= (np.abs(mag_i) < 90.0) & (np.abs(mag_Z) < 90.0)
        # colour transformation for LRG cuts
        c_p, c_r, d_r = self.colour_transformations(mag_g, mag_r, mag_i,)
        # defining the LOWZ sample
        lowz = colour_mask & self.LOWZ(mag_g, mag_r, mag_i, c_p, c_r)
        BMM.set_bit(bitmask, self._bits[0], condition=lowz)
        # defining the MIDZ sample
        midz = colour_mask & self.MIDZ(mag_r, mag_i, d_r)
        BMM.set_bit(bitmask, self._bits[1], condition=midz)
        # defining the HIGHZ sample
        highz = colour_mask & self.HIGHZ(mag_r, mag_i, mag_Z, mag_Ks)
        BMM.set_bit(bitmask, self._bits[2], condition=highz)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks):
        self.colour_selection(bitmask, mag_g, mag_r, mag_i, mag_Z, mag_Ks)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="OR")


######## GAMA (Driver+11) ########

@SampleManager.register
class Parser_GAMA(Parser):

    name = "GAMA"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_sdss["r"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_GAMA(BaseSelection):

    name = "GAMA"
    bit_descriptions = ("r-band cut",)

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_r):
        is_selected = mag_r < 19.87
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits))


######## SDSS (Strauss+02, Dawson+13 and
# http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php) ########

@SampleManager.register
class Parser_SDSS(Parser):

    name = "SDSS"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_sdss["g"],
            config.param_sdss["r"],
            config.param_sdss["i"],
            Parameter(
                "is_central", str, "environ/is_central",
                "flag indicating if it is central host galaxy"),
            Parameter(
                "lmhalo", str, "environ/log_M_halo",
                "logarithmic halo mass"),
            Parameter(
                "lmstellar", str, "environ/log_M_stellar",
                "logarithmic stellar mass"),
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_SDSS(BaseSelection):

    name = "SDSS"
    bit_descriptions = ("main sample", "BOSS LOWZ", "BOSS CMASS", "QSO")

    def MAIN_selection(self, bitmask, mag_r):
        is_selected = mag_r < 17.5  # r_pet
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

    @staticmethod
    def colour_transformations(mag_g, mag_r, mag_i,):
        c_p = 0.7 * (mag_g - mag_r) + 1.2 * (mag_r - mag_i - 0.18)
        c_r = (mag_r - mag_i) - (mag_g - mag_r) / 4.0 - 0.18
        d_r = (mag_r - mag_i) - (mag_g - mag_r) / 8.0
        return c_p, c_r, d_r

    @staticmethod
    def LOWZ(mag_r, c_p, c_r):
        # we cannot apply the r_psf - r_cmod cut
        lowz = (
            (mag_r > 16.0) &
            (mag_r < 19.6) &
            (np.abs(c_r) < 0.2) &
            (mag_r < 13.5 + c_p / 0.3))
        return lowz

    @staticmethod
    def CMASS(mag_r, mag_i, d_r):
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.8)) &
            ((mag_r - mag_i) < 2.0))
        return cmass

    def BOSS_selection(self, bitmask, mag_g, mag_r, mag_i):
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_g) < 90.0) & (np.abs(mag_r) < 90.0)
        colour_mask &= (np.abs(mag_r) < 90.0) & (np.abs(mag_i) < 90.0)
        # colour transformation for LRG cuts
        c_p, c_r, d_r = self.colour_transformations(mag_g, mag_r, mag_i,)
        # defining the LOWZ sample
        lowz = colour_mask & self.LOWZ(mag_r, c_p, c_r)
        BMM.set_bit(bitmask, self._bits[1], condition=lowz)
        # defining the CMASS sample
        cmass = colour_mask & self.CMASS(mag_r, mag_i, d_r)
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


######## WiggleZ (Drinkwater+10) ########

@SampleManager.register
class Parser_WiggleZ(Parser):

    name = "WiggleZ"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_redshift,
            config.param_sdss["g"],
            config.param_sdss["r"],
            config.param_sdss["i"],
            config.param_sdss["Z"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_WiggleZ(BaseSelection):

    name = "WiggleZ"
    bit_descriptions = ("inclusion rules", "exclusion rules")

    def colour_selection(self, bitmask, mag_g, mag_r, mag_i, mag_Z):
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


@SampleManager.register
class Sample_WiggleZ(RedshiftSampler):
    name = "WiggleZ"


######## DEEP2 (Newman+13) ########

@SampleManager.register
class Parser_DEEP2(Parser):

    name = "DEEP2"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_johnson["B"],
            config.param_johnson["Rc"],
            config.param_johnson["Ic"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_DEEP2(BaseSelection):

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
            self._success_rate_dir, "DEEP2_success"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        self._p_success_R = interp1d(
            success_R_centers, success_R_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_R_rate[0], 0.0))

    def colour_selection(self, bitmask, mag_B, mag_Rc, mag_Ic):
        # mask multi-band non-detections
        colour_mask = (np.abs(mag_B) < 90.0) & (np.abs(mag_Rc) < 90.0)
        colour_mask &= (np.abs(mag_Rc) < 90.0) & (np.abs(mag_Ic) < 90.0)
        # NOTE: We cannot apply the surface brightness cut and do not apply the
        #       Gaussian weighted sampling near the original colour cuts.
        is_selected = colour_mask & (
            (mag_Rc > 18.5) &
            (mag_Rc < 24.1) & (
                (mag_B - mag_Rc < 2.45 * (mag_Rc - mag_Ic) - 0.2976) |
                (mag_Rc - mag_Ic > 1.1) |
                (mag_B - mag_Rc < 0.5)))
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


@SampleManager.register
class Sample_DEEP2(DensitySampler):
    name = "DEEP2"


######## VVDSf02 (LeFèvre+05) ########

@SampleManager.register
class Parser_VVDSf02(Parser):

    name = "VVDSf02"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_redshift,
            config.param_johnson["Ic"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_VVDSf02(BaseSelection):

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
                self._success_rate_dir, "VVDSf02_I_success"))
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
            os.path.join(self._success_rate_dir, "VVDSf02_z_bright_success")).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(self._success_rate_dir, "VVDSf02_z_deep_success")).T
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
        is_selected = (mag_Ic > 17.5) & (mag_Ic < 24.0)
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


@SampleManager.register
class Sample_VVDSf02(DensitySampler):
    name = "VVDSf02"


######## zCOSMOS (Lilly+09) ########

@SampleManager.register
class Parser_zCOSMOS(Parser):

    name = "zCOSMOS"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_redshift,
            config.param_johnson["Ic"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_zCOSMOS(BaseSelection):

    name = "zCOSMOS"
    bit_descriptions = ("magnitude selection", "spectroscopic success")

    def __init__(self, bit_manager):
        super().__init__(bit_manager)
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        x = np.loadtxt(os.path.join(
            self._success_rate_dir, "zCOSMOS_z_sampling"))
        y = np.loadtxt(os.path.join(
            self._success_rate_dir, "zCOSMOS_I_sampling"))
        rates = np.loadtxt(os.path.join(
            self._success_rate_dir, "zCOSMOS_success"))
        self._p_success_zI = interp2d(
            x, y, rates, copy=True, kind="linear")

    def colour_selection(self, bitmask, mag_Ic):
        # NOTE: This only includes zCOSMOS bright.
        is_selected = (mag_Ic > 15.0) & (mag_Ic < 22.5)
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


@SampleManager.register
class Sample_zCOSMOS(DensitySampler):
    name = "zCOSMOS"


######## Sparse24mag ########

@SampleManager.register
class Parser_Sparse24mag(Parser):

    name = "Sparse24mag"
    default = ParameterCollection(
        config.make_bitmask_parameter(name),
        ParameterGroup(
            "selection",
            config.param_sdss["r"],
            header=config.selection_header),
        header=config.header)


@SampleManager.register
class Select_Sparse24mag(BaseSelection):

    name = "Sparse24mag"
    bit_descriptions = ("magnitude selection")

    @Schedule.description("selecting {:}".format(name))
    @Schedule.workload(0.10)
    def apply(self, bitmask, mag_r):
        is_selected = mag_r < 24.0
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits))


@SampleManager.register
class Sample_Sparse24mag(DensitySampler):
    name = "Sparse24mag"
