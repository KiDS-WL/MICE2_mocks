###############################################################################
#                                                                             #
#   This is a collection of spectroscopic redshift survey selection           #
#   functions that are most relevant for KiDS. Due to differences between     #
#   true and mock galaxy SED some tweaking to literature selections have      #
#   been applied. Whenever there are deviations from the cited literture      #
#   the lines have comment listing the true values.                           #
#                                                                             #
###############################################################################

import inspect
import os
import pickle

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from scipy.stats import gaussian_kde


# folder containing data for the spec. success rate for the deep spec-z samples
success_rate_dir = os.path.join(
    os.path.dirname(__file__),  # location of this script
    "success_rate_data")


class MICE_data(object):
    """
    Base class providing easy access to magnitude columns of MICE2 data tables.
    Filters can be accessed as class methods by there name, e.g.:

    MICE_data.i(suffix) would return the internally defined i-band magnitude
    with a given suffix (true: model magnitude, evo: evolution corrected model
    magnitude, obs: survey photometry realisation). Detects automatically if
    magnification is applied (additional suffix: mag).

    Supports indexing like astropy.table.Table itself.

    Parameters
    ----------
    micetable : astropy.table.Table
        MICE2 data table.
    """

    filter_keys = {}  # map band name to table keyword
    detection_band = None

    def __init__(self, micetable):
        assert(isinstance(micetable, Table))
        self.data = micetable

    def __len__(self):
        return len(self.data)

    def detect_mask(self):
        """
        Returns a mask indicating if an object was observed in the detection
        band of the survey.

        Returns
        -------
        mask : boolean array_like
            Indication whether dection was successful or not.
        """
        # non-detections are either mag==99 or mag==inf/nan
        detect_mag = self._get_filter_data(self.detection_band, "obs")
        detect_mask = np.isfinite(detect_mag) & (detect_mag < 90.0)
        return detect_mask

    def _get_filter_data(self, filter, suffix):
        """
        Returns the data of the requested magnitude.

        Parameters
        ----------
        filter : string
            Filter name as defined in self.filter_keys.
        suffix : string
            Suffix to distinguish different magnitude types (true: model
            magnitude, evo: evolution corrected model magnitude, obs: survey
            photometry realisation).

        Returns
        -------
        data : astropy.table.Table
            Magnitude column data.
        """
        full_key = "%s_%s_mag" % (self.filter_keys[filter], suffix)
        if full_key in self.data.colnames:
            print("selecting column '%s'" % full_key)
            return self.data[full_key]
        full_key = "%s_%s" % (self.filter_keys[filter], suffix)
        if full_key in self.data.colnames:
            print("selecting column '%s'" % full_key)
            return self.data[full_key]
        raise KeyError(
            "%s: filter '%s' does not exist with suffix '%s'" % (
                self.__class__.__name__, self.filter_keys[filter], suffix))

    def __getattr__(self, attr):
        if attr == '__bases__':  # needed for isinstance()
            raise AttributeError
        if attr not in self.filter_keys:
            raise KeyError("unknown filter '%s'" % attr)
        else:  # call _get_filter_data with the right parameters
            return lambda suffix: self._get_filter_data(attr, suffix)

    def __getitem__(self, item):
        # interface for table indexing
        return self.data.__getitem__(item)


class KV450_MICE_data(MICE_data):
    """
    Sub-class of MICE_data defining the KiDS-VIKING specific mock catalogue
    magnitude columns and the detection band.

    Parameters
    ----------
    micetable : astropy.table.Table
        MICE2 data table.
    """

    filter_keys = {
        "u": "sdss_u",
        "g": "sdss_g",
        "r": "sdss_r",
        "i": "sdss_i",
        "z": "sdss_z",
        "y": "des_asahi_full_y",
        "j": "vhs_j",
        "h": "vhs_h",
        "ks": "vhs_ks",
        "B": "lephare_b",
        "V": "lephare_v",
        "R": "lephare_rc",
        "I": "lephare_ic"}
    detection_band = "r"


class DES_MICE_data(MICE_data):
    """
    Sub-class of MICE_data defining the DES specific mock catalogue magnitude
    columns and the detection band.

    Parameters
    ----------
    micetable : astropy.table.Table
        MICE2 data table.
    """

    filter_keys = {
        "g": "des_asahi_full_g",
        "r": "des_asahi_full_r",
        "i": "des_asahi_full_i",
        "z": "des_asahi_full_z",
        "y": "des_asahi_full_y",
        "B": "lephare_b",
        "V": "lephare_v",
        "R": "lephare_rc",
        "I": "lephare_ic"}
    detection_band = "i"


class make_specz(object):
    """
    Base class for spectroscopic selection function. The selection is split
    into different steps (photometryCut, speczSuccess, surveyDetection,
    downsampling, etc.) which can be called in the correct order using the
    __call__() method, which also provides basic statistics about each
    selection process.

    Parameters
    ----------
    micedata : MICE_data
        MICE2 data table on which the selection function is applied.
    """

    # indicates the signature of __call__
    needs_n_z = False  # whether the total number of objects is required
    needs_n_tot = False  # whether the data n(z_spec) is required

    def __init__(self, micedata):
        assert(isinstance(micedata, MICE_data))
        self.data = micedata
        self.mask = np.ones(len(self.data), dtype="bool")
        self.redshift = micedata["z_cgal_v"]

    def stats(self):
        """
        Provide basic statistics on the current state of the selection mask.

        Returns
        -------
        stats : dict
            Contains the classmethod name from which which this method was
            called, the total number of remaining objects and their mean
            redshift.
        """
        stats = {
            "method": inspect.stack()[1][3],  # method name that called stats
            "N_tot": np.count_nonzero(self.mask),
            "z_mean": self.redshift[self.mask].mean()}
        return stats

    def input(self):
        """
        Returns self.stats() to record the input data.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        return self.stats()

    def photometryCut(self):
        """
        Method that applies all magnitude or colour cuts to the input data.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        raise NotImplementedError

    def speczSuccess(self):
        """
        Method that applies all magnitude or colour cuts to the input data.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        raise NotImplementedError

    def surveyDetection(self):
        """
        Method that applies the survey detection cut.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        # update the internal state
        self.mask &= self.data.detect_mask()
        return self.stats()

    def downsampling_N_tot(self, N_tot):
        """
        Method to randomly sample down the remaining MICE2 objects to a given
        number of data objects.

        Parameters
        ----------
        N_tot : int
            Number of objects to sample down to.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        idx_all = np.arange(len(self.data), dtype=np.int64)
        idx_preselect = idx_all[self.mask]
        # shuffle and select the first draw_n objects
        np.random.shuffle(idx_preselect)
        idx_keep = idx_preselect[:N_tot]
        # create a mask with only those entries enabled that have been selected
        mask = np.zeros_like(self.mask)
        mask[idx_keep] = True
        # update the internal state
        self.mask &= mask
        return self.stats()

    def downsampling_n_z(self, n_z):
        """
        Method to randomly sample down the remaining MICE2 objects to a given
        data redshift distribution using Gaussian KDEs.

        Parameters
        ----------
        n_z : array_like
            Spectroscopic redshifts of data objects.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        idx_all = np.arange(len(self.data), dtype=np.int64)
        # only consider objects that are not already masked
        idx_preselect = idx_all[self.mask]
        z_preselect = self.redshift[self.mask]
        # create Gaussian KDEs of the redshift distributions
        kde_spec = gaussian_kde(n_z, 0.05)
        # the accuracy will be governed by the spec-z KDE, therefore select
        # only the same number of photometric objects for its KDE
        idx = np.linspace(0.0, len(z_preselect)).astype(np.int64)
        kde_phot = gaussian_kde(z_preselect[idx], 0.05)
        # compute the rejection probability based on the data to mock redshift
        # probabilty densities
        p_keep = (
            (kde_spec(z_preselect) * len(n_z)) /
            (kde_phot(z_preselect) * len(z_preselect)))
        # reject randomly galaxies
        rand = np.random.uniform(size=len(z_preselect))
        mask_keep = 1.0 - np.minimum(1.0, p_keep) < rand
        # create a mask with only those entries enabled that have been selected
        mask = np.zeros_like(self.mask)
        mask[idx_preselect[mask_keep]] = True
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        """
        Apply the selection function step by step to the input mock catalogue.

        Parameters
        ----------
        pass_survey_selection : bool
            Whether the objects should be rejected which are not detected by
            the imaging survey.

        Returns
        -------
        data : astropy.table.Table
            Input mock data table with selection function mask applied.
        stats : list
            List of statistics from each selection functin step.
        """
        raise NotImplementedError


###############################################################################
#                            WIDE SURVEYS                                     #
###############################################################################


class make_2dFLenS(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.g = micedata.g("obs")
        self.r = micedata.r("obs")
        self.i = micedata.i("obs")
        self.z = micedata.z("obs")
        self.k = micedata.ks("obs")

    def photometryCut(self):
        ###############################################################
        #   based on Blake+16                                         #
        ###############################################################
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.k) < 99.0)
        mask &= (np.abs(self.i) < 99.0) & (np.abs(self.z) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # defining the LOWZ sample
        low_z1 = (
            (self.r > 16.5) &  # 16.0
            (self.r < 19.2) &
            (self.r < (13.1 + c_p / 0.32)) &  # 13.1, 0.3
            (np.abs(c_r) < 0.2))
        low_z2 = ~low_z1 & (
            (self.r > 16.5) &  # 16.0
            (self.r < 19.5) &
            (self.g-self.r > (1.3 + 0.25 * (self.r-self.i))) &
            (c_r > 0.45 - (self.g-self.r) / 6.0))
        low_z3 = ~low_z1 & ~low_z2 & (
            (self.r > 16.5) &  # 16.0
            (self.r < 19.6) &
            (self.r < (13.5 + c_p / 0.32)) &  # 13.5, 0.3
            (np.abs(c_r) < 0.2))
        low_z = low_z1 | low_z2 | low_z3
        # defining the MIDZ sample
        mid_z = (
            (self.i > 17.5) &
            (self.i < 19.9) &
            ((self.r-self.i) < 2.0) &
            (d_r > 0.55) &
            (self.i < 19.86 + 1.6 * (d_r - 0.9)))  # 19.86, 1.6, 0.8
        # defining the HIGHZ sample
        high_z = (
            (self.z < 19.9) &  # 19.95
            (self.i > 19.9) &
            (self.i < 21.8) &
            # the 2dFLenS paper uses r-W1, we must use the substitute r-Ks
            ((self.r-self.k) > 1.9 * (self.r-self.i)) &
            ((self.r-self.i) > 0.98) &
            ((self.i-self.z) > 0.6))
        mask = low_z | mid_z | high_z
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats


class make_GAMA(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.r = micedata.r("obs")

    def photometryCut(self):
        ###############################################################
        #   based on Driver+11                                        #
        ###############################################################
        mask = (self.r < 19.87)  # close to literature, best match to data n(z)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats


class make_SDSS_main(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.r = micedata.r("obs")

    def photometryCut(self):
        ###############################################################
        #   based on Strauss+02                                       #
        ###############################################################
        mask = (self.r < 17.7)  # r_pet~17.5
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats


class make_SDSS_BOSS(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.g = micedata.g("obs")
        self.r = micedata.r("obs")
        self.i = micedata.i("obs")

    def photometryCut(self):
        ###############################################################
        #   based on                                                  #
        #   http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php    #
        #   The selection has changed slightly compared to Dawson+13  #
        ###############################################################
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # defining the LOWZ sample
        # we cannot apply the r_psf - r_cmod cut
        low_z = (
            (self.r > 16.0) &
            (self.r < 20.0) &  # 19.6
            (np.abs(c_r) < 0.2) &
            (self.r < 13.35 + c_p / 0.3))  # 13.5, 0.3
        # defining the CMASS sample
        # we cannot apply the i_fib2, i_psf - i_mod and z_psf - z_mod cuts
        cmass = (
            (self.i > 17.5) &
            (self.i < 20.1) &  # 19.9
            (d_r > 0.55) &
            (self.i < 19.98 + 1.6 * (d_r - 0.7)) &  # 19.86, 1.6, 0.8
            ((self.r-self.i) < 2.0))
        # NOTE: we ignore the CMASS sparse sample
        mask = low_z | cmass
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats


class make_SDSS_BOSS_original(make_SDSS_BOSS):

    def photometryCut(self):
        ###############################################################
        #   based on                                                  #
        #   http://www.sdss3.org/dr9/algorithms/boss_galaxy_ts.php    #
        #   The selection has changed slightly compared to Dawson+13  #
        ###############################################################
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        # cut quantities (unchanged)
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # defining the LOWZ sample (unchanged)
        low_z = (
            (self.r > 16.0) &
            (self.r < 19.6) &
            (np.abs(c_r) < 0.2) &
            (self.r < 13.5 + c_p / 0.3))
        # defining the CMASS sample (unchanged)
        cmass = (
            (self.i > 17.5) &
            (self.i < 19.9) &
            (d_r > 0.55) &
            (self.i < 19.98 + 1.6 * (d_r - 0.8)) &
            ((self.r-self.i) < 2.0))
        # NOTE: we ignore the CMASS sparse sample
        mask = low_z | cmass
        # update the internal state
        self.mask &= mask
        return self.stats()


class make_SDSS_QSO(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)

    def environmentCut(self):
        """
        Method create a fake MICE2 quasar sample. Quasars do not exists in
        MICE2, therefore the assumption is made that quasars sit in the central
        galaxies of the most massive halos. This approach is only justifed by
        compairing the tails of the mock and data n(z) for the combined
        SDSS main, SDSS BOSS and SDSS QSO samples.

        Returns
        -------
        stats : dict
            Statistics dictionaly returned by self.stats().
        """
        is_central = self.data["flag_central"] == 1
        lm_halo = self.data["lmhalo"]
        lm_stellar = self.data["lmstellar"]
        mask = is_central & (lm_stellar > 11.2) & (lm_halo > 13.3)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.environmentCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats


class make_SDSS(make_specz):
    """
    The full SDSS selection is a combination of applying the SDSS main,
    SDSS BOSS and SDSS QSO sample selections on an input MICE2 mock catalogue.

    Parameters
    ----------
    micedata : MICE_data
        MICE2 data table on which the selection function is applied.
    """

    def __init__(self, micedata):
        super().__init__(micedata)

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        main = make_SDSS_main(self.data)
        data, stat = main(pass_survey_selection)
        self.mask = main.mask
        stat[-1]["method"] = "SDSS_main"
        stats.append(stat[-1])
        boss = make_SDSS_BOSS(self.data)
        data, stat = boss(pass_survey_selection)
        self.mask |= boss.mask
        stat[-1]["method"] = "SDSS_BOSS"
        stats.append(stat[-1])
        qsos = make_SDSS_QSO(self.data)
        data, stat = qsos(pass_survey_selection)
        self.mask |= qsos.mask
        stat[-1]["method"] = "SDSS_QSO"
        stats.append(stat[-1])
        return self.data.data[self.mask], stats


class make_WiggleZ(make_specz):

    needs_n_z = True

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.u = micedata.u("obs")
        self.g = micedata.g("obs")
        self.r = micedata.r("obs")
        self.i = micedata.i("obs")
        self.z = micedata.z("obs")

    def photometryCut(self):
        ###############################################################
        #   based on Drinkwater+10                                    #
        ###############################################################
        # colour masking
        colour_mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        colour_mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.u) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.g) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.r) < 99.0) & (np.abs(self.z) < 99.0)
        # photometric cuts
        # we cannot reproduce the FUV, NUV, S/N and position matching cuts
        include = (
            (self.r > 20.0) &
            (self.r < 22.5))
        exclude = (
            (self.g < 22.5) &
            (self.i < 21.5) &
            (self.r-self.i < (self.g-self.r - 0.1)) &
            (self.r-self.i < 0.4) &
            (self.g-self.r > 0.6) &
            (self.r-self.z < 0.7 * (self.g-self.r)))
        mask = include & ~exclude
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, nz_spec, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        stats.append(self.downsampling_n_z(nz_spec))
        return self.data.data[self.mask], stats


###############################################################################
#                            DEEP SURVEYS                                     #
###############################################################################


class make_DEEP2(make_specz):

    needs_n_tot = True

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.B = self.data.B("evo")
        self.R = self.data.R("evo")
        self.I = self.data.I("evo")

    def photometryCut(self):
        ###############################################################
        #   based on Newman+13                                        #
        ###############################################################
        # this modified selection gives the best match to the data n(z) with
        # its cut at z~0.75 and the B-R/R-I distribution (Newman+13, Fig. 12)
        # NOTE: We cannot apply the surface brightness cut and do not apply the
        #       Gaussian weighted sampling near the original colour cuts.
        mask = (
            (self.R > 18.5) &
            (self.R < 24.0) & (  # 24.1
                (self.B-self.R < 2.0 * (self.R-self.I) - 0.4) |  # 2.45, 0.2976
                (self.R-self.I > 1.1) |
                (self.B-self.R < 0.2)))  # 0.5
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        # Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        # with the first and last bin centered on 19 and 24.
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        # paper has given 1 - [sucess rate] in the histogram
        success_R_rate = np.loadtxt(os.path.join(
                success_rate_dir, "DEEP2_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at R > 24.1 being 0
        p_success_R = interp1d(
            success_R_centers, success_R_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_R_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(self.data))
        mask = random_draw < p_success_R(self.R)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, draw_n, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        stats.append(self.speczSuccess())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        stats.append(self.downsampling_N_tot(draw_n))
        return self.data.data[self.mask], stats


class make_DEEP2_original(make_DEEP2):

    def photometryCut(self):
        ###############################################################
        #   based on Newman+13                                        #
        ###############################################################
        # NOTE: We cannot apply the surface brightness cut and do not apply the
        #       Gaussian weighted sampling near the original colour cuts.
        mask = (
            (self.R > 18.5) &
            (self.R < 24.1) & (
                (self.B-self.R < 2.45 * (self.R-self.I) - 0.2976) |
                (self.R-self.I > 1.1) |
                (self.B-self.R < 0.5)))
        # update the internal state
        self.mask &= mask
        return self.stats()


class make_VVDSf02(make_specz):

    needs_n_tot = True

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.I = self.data.I("evo")

    def photometryCut(self):
        ###############################################################
        #   based on LeFèvre+05                                       #
        ###############################################################
        mask = (self.I > 18.5) & (self.I < 24.0)  # 17.5, 24.0
        # NOTE: The oversight of 1.0 magnitudes on the bright end misses 0.2 %
        #       of galaxies.
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # NOTE: We use a redshift-based and I-band based success rate
        #       independently here since we do not know their correlation,
        #       which makes the success rate worse than in reality.
        # Spec-z success rate as function of i_AB read of Figure 16 in
        # LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        # 0.5 mag with the first starting at 17 and the last bin ending at 24.
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_I_rate = np.loadtxt(os.path.join(
                success_rate_dir, "VVDSf02_I_success.txt"))
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(self.data))
        mask = random_draw < p_success_I(self.I)
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+13 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        # NOTE: at z > 1.75 there are only lower limits (due to a lack of
        # spec-z?), thus the success rate is extrapolated as 1.0 at z > 1.75
        success_z_bright_centers, success_z_bright_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_bright_success.txt")).T
        success_z_deep_centers, success_z_deep_rate = np.loadtxt(
            os.path.join(success_rate_dir, "VVDSf02_z_deep_success.txt")).T
        # interpolate the success rates as probability of being selected with
        # the probability in the bright bin at z > 1.75 being 1.0 and the deep
        # bin at z > 4.0 being 0.0
        p_success_z_bright = interp1d(
            success_z_bright_centers, success_z_bright_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_bright_rate[0], 1.0))
        p_success_z_deep = interp1d(
            success_z_deep_centers, success_z_deep_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_z_deep_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(self.data))
        iterator = zip(
            [self.I <= 22.5, self.I > 22.5],
            [p_success_z_bright, p_success_z_deep])
        for m, p_success_z in iterator:
            mask[m] &= random_draw[m] < p_success_z(self.redshift[m])
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, draw_n, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        stats.append(self.speczSuccess())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        stats.append(self.downsampling_N_tot(draw_n))
        return self.data.data[self.mask], stats


class make_zCOSMOS(make_specz):

    needs_n_tot = True

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.I = self.data.I("evo")

    def photometryCut(self):
        ###############################################################
        #   based on Lilly+09                                         #
        ###############################################################
        mask = (self.I > 15.0) & (self.I < 22.5)  # 15.0, 22.5
        # NOTE: This only includes zCOSMOS bright.
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        pickle_file = os.path.join(success_rate_dir, "zCOSMOS.cache")
        if not os.path.exists(pickle_file):
            x = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_z_sampling.txt"))
            y = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_I_sampling.txt"))
            rates = np.loadtxt(os.path.join(
                success_rate_dir, "zCOSMOS_success.txt"))
            p_success_zI = interp2d(x, y, rates, copy=True, kind="linear")
            with open(pickle_file, "wb") as f:
                pickle.dump(p_success_zI, f)
        else:
            with open(pickle_file, "rb") as f:
                p_success_zI = pickle.load(f)
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(self.data))
        object_rates = np.empty_like(random_draw)
        for i, (z, I_AB) in enumerate(zip(self.redshift, self.I)):
            # this must be in a loop since interp2d will create a grid from the
            # input redshifts and magnitudes instead of evaluating pairs of
            # values
            object_rates[i] = p_success_zI(z, I_AB)
        mask = random_draw < object_rates
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, draw_n, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        stats.append(self.speczSuccess())
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        stats.append(self.downsampling_N_tot(draw_n))
        return self.data.data[self.mask], stats


###############################################################################
#                            MOCK SAMPLES                                     #
###############################################################################


class make_idealized(make_specz):
    """
    Idealized spectroscopic mock sample used for cross-correlaton redshift
    tests.

    Parameters
    ----------
    micedata : MICE_data
        MICE2 data table on which the selection function is applied.
    """

    def __init__(self, micedata):
        assert(isinstance(micedata, MICE_data))
        self.data = micedata
        self.mask = np.ones(len(self.data), dtype="bool")
        self.redshift = micedata["z_cgal_v"]
        # magnitudes
        self.r = self.data.r("obs")

    def photometryCut(self):
        mask = (self.r < 24.0)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self, pass_survey_selection=False):
        stats = [self.input()]
        stats.append(self.photometryCut())
        fraction_KiDS = 0.575  # fraction of MICE galaxies after KiDS selection
        fraction_z_spec = 0.10  # desired fraction of spec-z to KiDS galaxies
        fraction_data = fraction_KiDS * fraction_z_spec
        stats.append(self.downsampling_N_tot(
            int(fraction_data * len(self.data))))
        if pass_survey_selection:
            stats.append(self.surveyDetection())
        return self.data.data[self.mask], stats
