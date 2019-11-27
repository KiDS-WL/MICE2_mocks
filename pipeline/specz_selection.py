import inspect
import os
import pickle

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from scipy.stats import gaussian_kde


class MICE_data(object):

    filter_keys = {}  # map band name to table keyword
    detection_band = None

    def __init__(self, micetable):
        assert(isinstance(micetable, Table))
        self.data = micetable

    def __len__(self):
        return len(self.data)

    def detect_mask(self):
        detect_mag = self._get_filter_keyword(self.detection_band, "obs")
        detect_mask = np.isfinite(detect_mag) & (detect_mag < 90.0)
        return detect_mask

    def _get_filter_keyword(self, filter, suffix):
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
        else:
            return lambda suffix: self._get_filter_keyword(attr, suffix)

    def __getitem__(self, item):
        return self.data.__getitem__(item)


class KV450_MICE_data(MICE_data):

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

    needs_n_z = False
    needs_n_tot = False

    def __init__(self, micedata):
        assert(isinstance(micedata, MICE_data))
        self.data = micedata
        self.mask = np.ones(len(self.data), dtype="bool")
        self.redshift = micedata["z_cgal_v"]

    def stats(self):
        stats = {
            "method": inspect.stack()[1][3],  # method name that called stats
            "N_tot": np.count_nonzero(self.mask),
            "z_mean": self.redshift[self.mask].mean()}
        return stats

    def input(self):
        return self.stats()

    def photometryCut(self):
        raise NotImplementedError

    def speczSuccess(self):
        raise NotImplementedError

    def surveyDetection(self):
        # update the internal state
        self.mask &= self.data.detect_mask()
        return self.stats()

    def downsampling_N_tot(self, N_tot):
        idx_all = np.arange(len(self.data), dtype=np.int64)
        idx_preselect = idx_all[self.mask]
        # shuffle and select the first draw_n objects
        np.random.shuffle(idx_preselect)
        idx_keep = idx_preselect[:N_tot]
        # create a new mask with entries enabled that have been selected
        mask = np.zeros_like(self.mask)
        mask[idx_keep] = True
        # update the internal state
        self.mask &= mask
        return self.stats()

    def downsampling_n_z(self, n_z):
        idx_all = np.arange(len(self.data), dtype=np.int64)
        idx_preselect = idx_all[self.mask]
        z_preselect = self.redshift[self.mask]
        # create Gaussian KDEs
        kde_spec = gaussian_kde(n_z, 0.05)
        kde_phot = gaussian_kde(z_preselect, 0.05)
        # compute the rejection probability based on their ratios
        p_keep = (
            (kde_spec(z_preselect) * len(n_z)) /
            (kde_phot(z_preselect) * len(z_preselect)))
        # reject randomly galaxies
        rand = np.random.uniform(size=len(z_preselect))
        mask_keep = 1.0 - np.minimum(1.0, p_keep) < rand
        mask = np.zeros_like(self.mask)
        mask[idx_preselect[mask_keep]] = True
        # update the internal state
        self.mask &= mask
        return self.stats()

    def __call__(self):
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
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.k) < 99.0)
        mask &= (np.abs(self.i) < 99.0) & (np.abs(self.z) < 99.0)
        # cut quantities
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # photometric cuts
        low_z1 = (
            (self.r > 16.5) &
            (self.r < 19.2) &
            (self.r < (13.1 + c_p / 0.32)) &
            (np.abs(c_r) < 0.2))
        low_z2 = ~low_z1 & (
            (self.r > 16.5) &
            (self.r < 19.5) &
            (self.g-self.r > (1.3 + 0.25 * (self.r-self.i))) &
            (c_r > 0.45 - (self.g-self.r) / 6.0))
        low_z3 = ~low_z1 & ~low_z2 & (
            (self.r > 16.5) &
            (self.r < 19.6) &
            (self.r < (13.5 + c_p / 0.32)) &
            (np.abs(c_r) < 0.2))
        low_z = low_z1 | low_z2 | low_z3
        mid_z = (
            (self.i > 17.5) &
            (self.i < 19.9) &
            ((self.r-self.i) < 2.0) &
            (d_r > 0.55) &
            (self.i < 19.86 + 1.6 * (d_r - 0.9)))
        high_z = (
            (self.z < 19.9) &
            (self.i > 19.9) &
            (self.i < 21.8) &
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
        mask = (self.r < 19.87)
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
        mask = (self.r < 17.7)
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
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        # cut quantities
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # photometric cuts
        low_z = (
            (self.r > 16.0) &
            (self.r < 20.0) &
            (np.abs(c_r) < 0.2) &
            (self.r < 13.35 + c_p / 0.3))
        cmass = (
            (self.i > 17.5) &
            (self.i < 20.1) &
            (d_r > 0.55) &
            (self.i < 19.98 + 1.6 * (d_r - 0.7)) &
            ((self.r-self.i) < 2.0))
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


class make_SDSS_BOSS_original(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.g = micedata.g("obs")
        self.r = micedata.r("obs")
        self.i = micedata.i("obs")

    def photometryCut(self):
        # colour masking
        mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        # cut quantities
        c_p = 0.7 * (self.g-self.r) + 1.2 * (self.r-self.i - 0.18)
        c_r = (self.r-self.i) - (self.g-self.r) / 4.0 - 0.18
        d_r = (self.r-self.i) - (self.g-self.r) / 8.0
        # photometric cuts
        low_z = (
            (self.r > 16.0) &
            (self.r < 19.5) &
            (np.abs(c_r) < 0.2) &
            (self.r < 13.5 + c_p / 0.3))
        cmass = (
            (self.i > 17.5) &
            (self.i < 19.9) &
            (d_r > 0.55) &
            (self.i < 19.98 + 1.6 * (d_r - 0.8)) &
            ((self.r-self.i) < 2.0))
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


class make_SDSS_QSO(make_specz):

    def __init__(self, micedata):
        super().__init__(micedata)

    def environmentCut(self):
        is_central = self.data["flag_central"] == 1
        lm_halo = self.data["lmhalo"]
        lm_stel = self.data["lmstellar"]
        mask = is_central & (lm_stel > 11.2) & (lm_halo > 13.3)
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
        # colour masking
        colour_mask = (np.abs(self.g) < 99.0) & (np.abs(self.r) < 99.0)
        colour_mask &= (np.abs(self.r) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.u) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.g) < 99.0) & (np.abs(self.i) < 99.0)
        colour_mask &= (np.abs(self.r) < 99.0) & (np.abs(self.z) < 99.0)
        # photometric cuts
        include = (
            (self.r > 20.0) &
            (self.r < 22.5) &
            (self.u-self.i > 1.5) &
            (self.u-self.i < 2.0) &
            (self.g-self.i > 0.9))
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
        mask = (
            (self.R > 18.5) &                                    # 18.5
            (self.R < 24.0) & (                                  # 24.1
                (self.B-self.R < 2.0 * (self.R-self.I) - 0.4) |  # 2.45, 0.2976
                (self.R-self.I > 1.1) |                          # 1.1
                (self.B-self.R < 0.2)))                          # 0.5
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # Spec-z success rate as function of r_AB for Q>=3 read of Figure 13 in
        # Newman+13 for DEEP2 fields 2-4. Values are binned in steps of 0.2 mag
        # with the first and last bin centered on 19 and 24.
        success_R_bins = np.arange(18.9, 24.1 + 0.01, 0.2)
        success_R_centers = (success_R_bins[1:] + success_R_bins[:-1]) / 2.0
        success_R_rate = 1 - np.array([
            0.069, 0.069, 0.082, 0.082, 0.069, 0.070, 0.044, 0.103, 0.093,
            0.141, 0.093, 0.085, 0.108, 0.101, 0.110, 0.139, 0.161, 0.173,
            0.184, 0.191, 0.261, 0.290, 0.337, 0.393, 0.433, 0.467])
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


class make_VVDSf02(make_specz):

    needs_n_tot = True

    def __init__(self, micedata):
        super().__init__(micedata)
        # magnitudes
        self.I = self.data.I("evo")

    def photometryCut(self):
        mask = (self.I > 18.5) & (self.I < 24.0)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # Spec-z success rate as function of i_AB read of Figure 13 in
        # LeFevre+05 for the VVDS 2h field. Values are binned in steps of
        # 0.5 mag with the first starting at 17 and the last bin ending at 24.
        success_I_bins = np.arange(17.0, 24.0 + 0.01, 0.5)
        success_I_centers = (success_I_bins[1:] + success_I_bins[:-1]) / 2.0
        success_I_rate = np.array([
            0.993, 0.980, 0.980, 0.977, 0.950, 0.960, 0.960,
            0.944, 0.934, 0.917, 0.891, 0.854, 0.785, 0.685])
        # interpolate the success rate as probability of being selected with
        # the probability at I > 24 being 0
        p_success_I = interp1d(
            success_I_centers, success_I_rate, kind="quadratic",
            bounds_error=False, fill_value=(success_I_rate[0], 0.0))
        # Randomly sample objects according to their success rate
        random_draw = np.random.rand(len(self.data))
        mask = random_draw < p_success_I(self.I)
        # Spec-z success rate as function of redshift read of Figure 13a/b in
        # LeFevre+05 for VVDS deep sample. The listing is split by i_AB into
        # ranges (17.5; 22.5] and (22.5; 24.0].
        success_z_bright_centers, success_z_bright_rate = np.transpose([
            [0.000, 0.996], [0.239, 0.958], [0.345, 0.979], [0.447, 0.975],
            [0.539, 0.981], [0.640, 0.983], [0.741, 0.992], [0.843, 0.979],
            [0.940, 0.985], [1.046, 0.983], [1.143, 0.945], [1.240, 0.861],
            [1.346, 0.838], [1.444, 0.686], [1.541, 0.568], [1.647, 0.604],
            [1.741, 1.000]])  # at z > 1.75 there are only lower limits
        success_z_deep_centers, success_z_deep_rate = np.transpose([
            [0.000, 0.888], [0.251, 0.897], [0.343, 0.920], [0.454, 0.945],
            [0.550, 0.962], [0.647, 0.964], [0.749, 0.954], [0.850, 0.933],
            [0.947, 0.912], [1.053, 0.901], [1.150, 0.872], [1.251, 0.869],
            [1.353, 0.869], [1.445, 0.867], [1.546, 0.851], [1.652, 0.861],
            [1.749, 0.836], [1.851, 0.722], [1.953, 0.678], [2.050, 0.691],
            [2.151, 0.686], [2.248, 0.617], [2.354, 0.691], [2.450, 0.697],
            [2.552, 0.657], [2.653, 0.678], [2.754, 0.764], [2.851, 0.802],
            [2.952, 0.905], [3.049, 0.876], [3.150, 0.844], [3.252, 0.789],
            [3.353, 0.787], [3.451, 0.640], [4.003, 0.752]])
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
        mask = (self.I > 15.0) & (self.I < 22.5)
        # update the internal state
        self.mask &= mask
        return self.stats()

    def speczSuccess(self):
        # Spec-z success rate as function of redshift (x) and I_AB (y) read of
        # Figure 3 in Lilly+09 for zCOSMOS bright sample. Do a spline
        # interpolation of the 2D data and save it as pickle on the disk for
        # faster reloads
        wdir = os.path.dirname(__file__)
        pickle_file = os.path.join(wdir, "zCOSMOS_successrate.cache")
        if not os.path.exists(pickle_file):
            x = np.loadtxt(os.path.join(
                wdir, "zCOSMOS_successrate_z_sampling.txt"))
            y = np.loadtxt(os.path.join(
                wdir, "zCOSMOS_successrate_I_sampling.txt"))
            rates = np.loadtxt(os.path.join(
                wdir, "zCOSMOS_successrate_sample_values.txt"))
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
