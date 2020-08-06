import numpy as np


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


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


###############################################################################
#                            MOCK SAMPLES                                     #
###############################################################################


REGISTERED_SAMPLES = []
for name in sorted(dir()):
    if name.startswith("Select"):
        REGISTERED_SAMPLES.append(name[6:])
