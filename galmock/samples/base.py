#
# This module implements the selection function and density sample base classes
# as well as the sample manager, which makes the sample selection functions
# accessible by name.
#

import json
import logging
from collections import defaultdict
from os.path import join, dirname

import numpy as np

from galmock.core.bitmask import BitMaskManager as BMM
from galmock.core.parallel import Schedule
from galmock.core.utils import ProgressBar
from galmock.matching import DistributionEstimator


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _SampleManager(object):
    """
    Manager that allows accessing selection functions, density samples and
    configuration parser by sample name.
    """

    _parsers = defaultdict(dict)
    _selectors = defaultdict(dict)
    _samplers = defaultdict(dict)

    def register(self, obj):
        """
        Decorator used to register sample selection function logics. Depending
        on type of the decorated class, it is automatically identified as
        parser, selection function or density samplinger.
        """
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
        """
        Lists the available flavours, i.e. selection function variants, such
        as the the reference implemention and selections that are modified to
        suit a specific input mock catalogue.

        Returns:
        --------
        flavours : list of str
            List of flavour names.
        """
        flavours = (
            set(self._parsers.keys()) |
            set(self._selectors.keys()) |
            set(self._samplers.keys()))
        return sorted(flavours)

    @property
    def samples(self):
        """
        Lists the implemented sample selection functions. Each sample must at
        least have a parser and flavoured implementations require a reference
        implementation.

        Returns:
        --------
        are_implemented : list of str
            List of sample names.
        """
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
        """
        Check whether a specific sample is implemented with an optional
        flavour specified. Raises a NotImplementedError otherwise.

        Parameters:
        -----------
        flavour : str or None
            Flavour name which is checked for existance.
        sample : str or None
            Sample name which is checked for existance.
        """
        if flavour is not None and flavour not in self.flavours:
            message = "flavour not implemented: {:}"
            raise NotImplementedError(message.format(flavour))
        if sample is not None and sample not in self.samples:
            message = "sample not implemented: {:}"
            raise NotImplementedError(message.format(sample))

    def get_parser(self, flavour, sample):
        """
        Returns the configuration file parser for a given sample and flavour.
        Falls back to the reference implementation if the flavour does not
        exist.

        Returns:
        --------
        parser : subclass of galmock.core.config.Parser or NotImplemeted
            Parser for a specific flavour and sample. Returns NotImplemented if
            the sample is not implemented.
        """
        try:
            return self._parsers[flavour][sample]
        except KeyError:
            return self._parsers["reference"].get(sample, NotImplemented)

    def get_selector(self, flavour, sample):
        """
        Returns the selection function for a given sample and flavour. Falls
        back to the reference implementation if the flavour does not exist.

        Returns:
        --------
        parser : subclass of BaseSelection or NotImplemeted
            Selection function for a specific flavour and sample. Returns
            NotImplemented if the sample is not implemented.
        """
        self._check(flavour, sample)
        try:
            return self._selectors[flavour][sample]
        except KeyError:
            message = "no selection implemented for flavour '{:}', "
            message += "falling back to 'reference'"
            logger.warn(message.format(flavour))
            return self._selectors["reference"].get(sample, NotImplemented)

    def get_sampler(self, flavour, sample):
        """
        Returns the density sampler for a given sample and flavour. Falls back
        to the reference implementation if the flavour does not exist.

        Returns:
        --------
        parser : subclass of Sampler or NotImplemeted
            Density sampler for a specific flavour and sample. Returns
            NotImplemented if the sample is not implemented.
        """
        self._check(flavour, sample)
        try:
            return self._samplers[flavour][sample]
        except KeyError:
            message = "no sampler implemented for flavour '{:}', "
            message += "falling back to 'reference'"
            logger.warn(message.format(flavour))
            return self._samplers["reference"].get(sample, NotImplemented)


# instantiate the sample manager once such that the sample implementations can
# be registered in reference.py and the flavour modules.
SampleManager = _SampleManager()


# folder containing data for the spec. success rate for the deep spec-z samples
_SAMPLE_DATA_DIR = join(dirname(__file__), "data")


class BaseSelection(object):
    """
    Base class for sample selection functions that defines the minimum methods
    required.

    Parameters:
    -----------
    bit_manager : galmock.core.bitmask.BitManager
        Each selection function requires a bit manager that manages the bit
        mask values and generates automatic information descriptions for the
        data store column attributes.
    """

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
    """
    Base class for density sampler. Each density sample requires a
    corresponding configuration file in the ./data directory which provides the
    target surface density or the redshift distribution to sample (see
    galmock.matching.DistributionEstimator.from_file()).
    """

    name = "Generic Sample"

    @property
    def density_file(self):
        return join(_SAMPLE_DATA_DIR, "{:}.json".format(self.name))


class DensitySampler(Sampler):
    """
    Base class for a surface density sampler. Establishes the sample density
    and randomly samples down the input mock catalogue (represented by a
    selection bit mask) to this surface density.

    Parameters:
    -----------
    bit_manager : galmock.core.bitmask.BitManager
        Each density sampler requires a bit manager that manages the bit mask
        values and generates automatic information descriptions for the data
        store column attributes.
    area : float
        Area of the mock sample that is passed to the sample, required to
        estimate the surface density.
    bitmask : array-like of uint
        Bit mask that indicates which objects pass specific selection
        steps.
    """

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
        """
        Queries the bit mask manager for the current sample to count how many
        objects currently pass the sample selection.

        Parameters:
        -----------
        bitmask : array-like of uint
            Bit mask that indicates which objects pass specific selection
            steps.

        Returns:
        --------
        n_mocks : int
            Number of objects currently passing the selection.
        """
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
        """
        Returns the reference data sample surface density.
        """
        return self._sample_dens

    @property
    def mock_density(self):
        """
        Returns the mock data surface density.
        """
        return self._mock_dens

    def odds(self):
        """
        Returns the odds of a mock galaxy being sampled.
        """
        return self._sample_dens / self._mock_dens

    def draw(self, n_points):
        """
        Randomly select objects such that the sample surface density is
        achived. The selection is returned as boolean mask.

        Parameters:
        -----------
        n_points : int
            Number of objects for which the selection mask is created.
        
        Returns:
        --------
        mask : bool or array-like
            Boolean mask indicating whether an object has been randomly
            selected.
        """
        random_draw = np.random.rand(n_points)
        mask = random_draw < self.odds()
        return mask

    @Schedule.description("sampling surface density")
    @Schedule.workload(0.10)
    def apply(self, bitmask):
        """
        Apply the density sampler to the selection bit mask. Objects that are
        selected are flagging the bit mask. This modifies the mask in place.

        Parameters:
        -----------
        bitmask : uint or array-like
            Bit mask that indicates which objects pass specific selection
            steps.
        """
        is_selected = self.draw(len(bitmask))
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")


class RedshiftSampler(Sampler):
    """
    Base class for a redshift weighted density sampler. Establishes the sample
    redshift density using an average shifted histogram and randomly samples
    down the input mock catalogue (represented by a selection bit mask) to this
    redshift density.

    Parameters:
    -----------
    bit_manager : galmock.core.bitmask.BitManager
        Each density sampler requires a bit manager that manages the bit mask
        values and generates automatic information descriptions for the data
        store column attributes.
    area : float
        Area of the mock sample that is passed to the sample, required to
        estimate the surface density.
    bitmask : array-like of uint
        Bit mask that indicates which objects pass specific selection
        steps.
    redshifts : array-like of float
        Redshifts of the input mock galaxies.
    """

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
        """
        Queries the bit mask manager for the current sample and measured the
        density of objects currently passing the sample selection as function
        of redshift.

        Parameters:
        -----------
        bitmask : array-like of uint
            Bit mask that indicates which objects pass specific selection
            steps.
        redshift : array-like of float
            Redshifts of the input mock galaxies.
        area : float
            Area of the mock sample that is passed to the sample, required to
            estimate the surface density.

        Returns:
        --------
        n_mocks : int
            Number of objects currently passing the selection.
        """
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
        """
        Returns the the surface density of the reference data sample
        """
        n_objects = self._sample_dens._counts.sum()
        return n_objects * self._sample_dens.normalisation

    @property
    def mock_density(self):
        """
        Returns the mock data surface density.
        """
        n_objects = self._mock_dens._counts.sum()
        return n_objects * self._mock_dens.normalisation

    def odds(self, redshift):
        """
        Returns the odds of a mock galaxy being sampled as function of
        redshift.

        Parameters:
        -----------
        redshift : array-like of float
            Redshifts of the input mock galaxies.
        """
        mock_density = self._mock_dens(redshift)
        # divisions by zero are likely and must be substituted
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.where(
                mock_density == 0.0, 0.0,
                self._sample_dens(redshift) / mock_density)
        return odds

    def draw(self, redshift):
        """
        Randomly select objects based on their redshift such that the sample
        redshift density is achived. The selection is returned as boolean mask.

        Parameters:
        -----------
        redshift : array-like of float
            Redshifts of the input mock galaxies.
        
        Returns:
        --------
        mask : bool or array-like
            Boolean mask indicating whether an object has been randomly
            selected.
        """
        random_draw = np.random.rand(len(redshift))
        mask = random_draw < self.odds(redshift)
        return mask

    @Schedule.description("sampling redshift density")
    @Schedule.workload(0.20)
    def apply(self, bitmask, redshift):
        """
        Apply the density sampler to the selection bit mask. Objects that are
        selected are flagging the bit mask. This modifies the mask in place.

        Parameters:
        -----------
        bitmask : uint or array-like
            Bit mask that indicates which objects pass specific selection
            steps.
        redshift : array-like of float
            Redshifts of the input mock galaxies.
        """
        is_selected = self.draw(redshift)
        BMM.set_bit(bitmask, self._bits[0], condition=is_selected)
        # update the master selection bit
        BMM.update_master(bitmask, sum(self._bits), bit_join="AND")
