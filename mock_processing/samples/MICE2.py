from . import reference

from .base import SampleManager


@SampleManager.register
class Select_2dFLenS(reference.Select_2dFLenS):

    @staticmethod
    def LOWZ(mag_g, mag_r, mag_i, c_p, c_r):
        lowz_1 = (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.2) &
            (mag_r < (13.1 + c_p / 0.32)) &  # 13.1, 0.3
            (np.abs(c_r) < 0.2))
        lowz_2 = (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.5) &
            (mag_g - mag_r > (1.3 + 0.25 * (mag_r - mag_i))) &
            (c_r > 0.45 - (mag_g - mag_r) / 6.0))
        lowz_3 = (
            (mag_r > 16.5) &  # 16.0
            (mag_r < 19.6) &
            (mag_r < (13.5 + c_p / 0.32)) &  # 13.5, 0.3
            (np.abs(c_r) < 0.2))
        return lowz_1 | lowz_2 | lowz_3

    @staticmethod
    def MIDZ(mag_r, mag_i, d_r):
        midz = (
            (mag_i > 17.5) &
            (mag_i < 19.9) &
            ((mag_r - mag_i) < 2.0) &
            (d_r > 0.55) &
            (mag_i < 19.86 + 1.6 * (d_r - 0.9)))  # 19.86, 1.6, 0.8
        return midz

    @staticmethod
    def HIGHZ(mag_r, mag_i, mag_Z, mag_Ks):
        highz = (
            (mag_Z < 19.9) &  # 19.95
            (mag_i > 19.9) &
            (mag_i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((mag_r - mag_Ks) > 1.9 * (mag_r - mag_i)) &
            ((mag_r - mag_i) > 0.98) &
            ((mag_i - mag_Z) > 0.6))
        return highz


@SampleManager.register
class Select_SDSS(reference.Select_SDSS):

    def MAIN_selection(self, bitmask, mag_r):
        is_selected = mag_r < 17.7  # r_pet ~ 17.5
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)

    @staticmethod
    def LOWZ(mag_r, c_p, c_r):
        # we cannot apply the r_psf - r_cmod cut
        lowz = (
            (mag_r > 16.0) &
            (mag_r < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (mag_r < 13.35 + c_p / 0.3))  # 13.5, 0.3
        return lowz

    @staticmethod
    def CMASS(mag_r, mag_i, d_r):
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (mag_i > 17.5) &
            (mag_i < 20.1) &  # 19.9
            (d_r > 0.55) &
            (mag_i < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((mag_r - mag_i) < 2.0))
        return cmass


@SampleManager.register
class Select_DEEP2(reference.Select_DEEP2):

    def colour_selection(self, bitmask, mag_B, mag_Rc, mag_Ic):
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
