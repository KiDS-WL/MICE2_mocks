import os
import pickle
import numpy as np

from scipy.interpolate import interp1d, interp2d


# folder containing data for the spec. success rate for the deep spec-z samples
success_rate_dir = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "sample_data")


class SelectSample(object):

    _description = "describes the fields of the selection bit mask"

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def dtype(self):
        return np.uint8

    @property
    def description(self):
        return self._description


class SelectKiDS(SelectSample):

    _description = "KiDS sample selection bit mask. Select objects passing "
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


class Select2dFLenS(SelectSample):

    _description = "2dFLenS sample selection bit mask. Select the LOWZ sample "
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


class SelectGAMA(SelectSample):

    _description = "GAMA sample selection bit mask. Select sample objects by "
    _description += "(mask & 1)"

    def __call__(self, mag_r):
        ###############################################################
        #   based on Driver+11                                        #
        ###############################################################
        bitmask = np.where(mag_r < 19.87, 1, 0).astype(self.dtype)
        return bitmask


class SelectSDSS(SelectSample):

    _description = "SDSS sample selection bit mask. Select the main galaxy "
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
            mag_BOSS_selection(mag_g, mag_r, mag_i),
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


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


class SelectDEEP2(SelectSample):

    _description = "DEEP2 sample selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) and the full sample "
    _description += "by (mask & 1)."

    def __init__(self):
        # Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        # Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        # with the first and last bin centered on 19 and 24.
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        # paper has given 1 - [sucess rate] in the histogram
        success_R_rate = np.loadtxt(os.path.join(
            success_rate_dir, "DEEP2_success"))
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

    def specz_success(self, mag_R):
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(mag_R))
        mask = random_draw < self._p_success_R(mag_R)
        bitmask = np.where(mask, 4, 0).astype(self.dtype)
        return bitmask

    def __call__(self, mag_B, mag_Rc, mag_Ic):
        bitmask = self.colour_selection(mag_B, mag_Rc, mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_R),
            bitmask)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


class SelectVVDSf02(SelectSample):

    _description = "VVDS-02h field selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) and the full sample "
    _description += "by (mask & 1)."

    def __init__(self):
        # NOTE: We use a redshift-based and I-band based success rate
        #       independently here since we do not know their correlation,
        #       which makes the success rate worse than in reality.
        # Spec-z success rate as function of i_AB read of Figure 16 in
        # LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        # 0.5 mag with the first starting at 17 and the last bin ending at 24.
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_I_rate = np.loadtxt(os.path.join(
                success_rate_dir, "VVDSf02_I_success"))
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+13 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        # NOTE: at z > 1.75 there are only lower limits (due to a lack of
        # spec-z?), thus the success rate is extrapolated as 1.0 at z > 1.75
        success_z_bright_centers, success_z_bright_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_bright_success")).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_deep_success")).T
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
        mask = random_draw < p_success_I(mag_Ic)
        iterator = zip(
            [mag_Ic <= 22.5, mag_Ic > 22.5],
            [self._p_success_z_bright, self._p_success_z_deep])
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(redshift[m])
        bitmask = np.where(mask, 4, 0).astype(self.dtype)
        return bitmask

    def __call__(self, mag_Ic, redshift):
        bitmask = self.colour_selection(mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_Ic, redshift),
            bitmask)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


class SelectzCOSMOS(SelectSample):

    _description = "zCOSMOS sample selection bit mask. Select objects passing "
    _description += "photometric cuts by (mask & 2), objects passing the "
    _description += "sampling succes rate by (mask & 4) and the full sample "
    _description += "by (mask & 1)."

    def __init__(self):
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        pickle_file = os.path.join(success_rate_dir, "zCOSMOS.cache")
        if not os.path.exists(pickle_file):
            x = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_z_sampling"))
            y = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_I_sampling"))
            rates = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_success"))
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

    def __call__(self, mag_Ic, redshift):
        bitmask = self.colour_selection(mag_Ic)
        bitmask = np.bitwise_or(
            self.specz_success(mag_Ic, redshift),
            bitmask)
        # for convenience flag all objects that pass the selection
        bitmask = np.bitwise_or(
            # join conditions with AND
            np.where(bitmask == 6, 1, 0).astype(self.dtype),
            bitmask)
        return bitmask


###############################################################################
#                            TEST SAMPLES                                     #
###############################################################################


###############################################################################

REGISTERED_SAMPLES = []
for name in sorted(dir()):
    if name.startswith("Select"):
        REGISTERED_SAMPLES.append(name[6:])
