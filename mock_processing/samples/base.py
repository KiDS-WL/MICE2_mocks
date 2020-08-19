import json
from collections import defaultdict
from os.path import join, dirname

import numpy as np

from mock_processing.core.bitmask import BitMaskManager as BMM
from mock_processing.core.parallel import Schedule
from mock_processing.core.utils import ProgressBar
from mock_processing.matching import DistributionEstimator


class _SampleManager(object):

    _parsers = defaultdict(dict)
    _selectors = defaultdict(dict)
    _samplers = defaultdict(dict)

    def register(self, obj):
        try:
            *other, flavour = obj.__module__.split(".")
            kind, sample = obj.__name__.split("_")
            assert(kind in ("Parser", "Select", "Sample"))
            # insert the object into the register
            if kind == "Parser":
                self._parsers[flavour][sample] = obj
            elif kind == "Select":
                self._selectors[flavour][sample] = obj
            elif kind == "Sample":
                self._samplers[flavour][sample] = obj
            else:
                raise ValueError("invalid object kind: {:}".format(kind))
        except Exception:
            message = "invalid name for registered object: {:}".format(obj)
            raise ImportError(message)
        return obj

    @property
    def flavours(self):
        flavours = (
            set(self._parsers.keys()) |
            set(self._selectors.keys()) |
            set(self._samplers.keys()))
        return sorted(flavours)

    @property
    def samples(self):
        are_implemented = (
            set(self._selectors["reference"].keys()) |
            set(self._samplers["reference"].keys()))
        have_parser = set(self._parsers["reference"].keys())
        # check that there is always an implementation with a config parser
        for sample in are_implemented ^ have_parser:
            message = "sample either lacks a configuration parser or an "
            message += "implementation: {:}".format(sample)
            raise NotImplementedError(message)
        # check that there are no flavoured implementations with out reference
        for flavour in self.flavours:
            flavour_samples = set(self._parsers[flavour].keys())
            for sample in flavour_samples - are_implemented:
                message = "sample is implemented for '{:}' but "
                message += "has no reference implementation: {:}"
                raise NotImplementedError(message.format(flavour, sample))
        return sorted(are_implemented)

    def _check(self, flavour=None, sample=None):
        if flavour is not None and flavour not in self.flavours:
            message = "flavour not implemented: {:}"
            raise NotImplementedError(message.format(flavour))
        if sample is not None and sample not in self.samples:
            message = "sample not implemented: {:}"
            raise NotImplementedError(message.format(sample))

    def get_parser(self, flavour, sample):
        self._check(flavour, sample)
        try:
            return self._parsers[flavour][sample]
        except KeyError:
            return self._parsers["reference"].get(sample, NotImplemented)

    def get_selector(self, flavour, sample):
        self._check(flavour, sample)
        try:
            return self._selectors[flavour][sample]
        except KeyError:
            return self._selectors["reference"].get(sample, NotImplemented)

    def get_sampler(self, flavour, sample):
        self._check(flavour, sample)
        try:
            return self._samplers[flavour][sample]
        except KeyError:
            return self._samplers["reference"].get(sample, NotImplemented)


SampleManager = _SampleManager()


# folder containing data for the spec. success rate for the deep spec-z samples
_SAMPLE_DATA_DIR = join(dirname(__file__), "data")


class BaseSelection(object):

    name = "Generic Sample"
    bit_descriptions = []
    _success_rate_dir = _SAMPLE_DATA_DIR

    def __init__(self, bit_manager):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]

    @property
    def __name__(self):
        return self.__class__.__name__


class Sampler(object):

    name = "Generic Sample"

    @property
    def density_file(self):
        return join(_SAMPLE_DATA_DIR, "{:}.json".format(self.name))


class DensitySampler(Sampler):

    bit_descriptions = ("surface density downsampling",)

    def __init__(self, bit_manager, area, bitmask):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]
        # load sample density in deg^-2
        with open(self.density_file) as f:
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

    def __init__(self, bit_manager, area, bitmask, redshifts):
        self._bits = [
            bit_manager.reserve(desc) for desc in self.bit_descriptions]
        # load sample density in deg^-2 per redshift
        self._sample_dens = DistributionEstimator.from_file(self.density_file)
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
