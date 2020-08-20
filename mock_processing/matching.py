import os
import json

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from mmaptable import MmapTable

from mock_processing.core.config import (Parameter, ParameterCollection,
                                         ParameterGroup, ParameterListing,
                                         Parser)
from mock_processing.core.parallel import Schedule
from mock_processing.core.utils import expand_path


class DistributionEstimator(object):

    _interpolator = None
    _normalisation = 1.0

    def __init__(
            self, xmin, xmax, width, smooth=10,
            kind="linear", fill_value=None):
        # construct a binning for the data
        self._xmin = float(xmin)
        self._xmax = float(xmax)
        self._width = float(width)
        self._binning = np.arange(
            self._xmin, self._xmax + self._width, self._width)
        # array to accumulate bin counts
        self._counts = np.zeros(len(self._binning) - 1)
        # set interpolation parameters
        self._smooth = int(smooth)
        self._kind = kind
        self._fill_value = float(fill_value)
 
    @classmethod
    def from_file(cls, path, kind="linear", fill_value=None):
        instance = cls.__new__(cls)
        instance._kind = kind
        instance._fill_value = fill_value
        # load the data from the file
        with open(path) as f:
            attr_dict = json.load(f)
        # set the attributes and rerun the interpolation
        for attr, value in attr_dict.items():
            if type(value) is list:
                setattr(instance, attr, np.array(value))
            else:
                setattr(instance, attr, value)
        # reconstruct the binning and the interpolation
        instance._binning = np.arange(
            instance._xmin, instance._xmax + instance._width, instance._width)
        instance.interpolate()
        return instance

    def to_file(self, path):
        # save all required attributes as JSON file
        attrs = (
            "_xmin", "_xmax", "_width", "_smooth", "_kind", "_fill_value",
            "_counts", "_normalisation")
        attr_dict = {}
        for attr in attrs:
            value = getattr(self, attr)
            if type(value) is np.ndarray:
                attr_dict[attr] = list(value)
            else:
                attr_dict[attr] = value
        # do not use pickle for safety reasons
        with open(path, "w") as f:
            json.dump(attr_dict, f,indent=2, sort_keys=True)

    def add_data(self, data):
        self._counts += np.histogram(
            data, bins=self._binning, density=False)[0]
        self._interpolator = None

    @property
    def xmin(self):
        return self._xmin

    @property
    def xmax(self):
        return self._xmax

    @property
    def width(self):
        return self._width

    @property
    def bin_edges(self):
        return self._binning

    @property
    def bin_centers(self):
        return (self._binning[1:] + self._binning[:-1]) / 2.0

    @property
    def smoothing(self):
        return self._smooth

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind):
        self._kind = kind
        self.interpolate()

    @property
    def fill_value(self):
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = value
        self.interpolate()

    @property
    def normalisation(self):
        return self._normalisation
    
    @normalisation.setter
    def normalisation(self, amp):
        self._normalisation = amp

    def interpolate(self, density=False):
        smoothed = np.empty_like(self._counts)
        # compute the rolling sum over the high resolution histogram
        idx_offset = max(1, self._smooth // 2)
        for i in range(len(smoothed)):
            start = max(0, i - idx_offset)
            end = min(len(smoothed), i + idx_offset)
            smoothed[i] = self._counts[start:end].sum()
        # normalize to unity
        bin_centers = self.bin_centers
        if smoothed.any() and density:
            smoothed = smoothed / np.trapz(smoothed, x=bin_centers)
        # interpolate the values
        kwargs = {"kind": self._kind}
        if self._fill_value is not None:
            kwargs.update({
                "bounds_error": False, "fill_value": self._fill_value})
        self._interpolator = interp1d(bin_centers, smoothed, **kwargs)

    def __call__(self, x):
        try:
            interp = self._interpolator(x)
        except TypeError:
            self.interpolate()
            interp = self._interpolator(x)
        return interp * self._normalisation


class MatcherParser(Parser):

    default = ParameterCollection(
        Parameter(
            "input", str, "...",
            "path to input table, must be MmapTable compatible",
            parser=expand_path),
        Parameter(
            "normalize", bool, False,
            "whether each feature space dimension is normalized by the "
            "standard deviation of the values in the dimension"),
        Parameter(
            "max_dist", float, 1.0,
            "maximum distance (Euclidean metric) to consider the nearest "
            "neighbour a match, otherwise the fallback value is assigned"),
        Parameter(
            "every_n", int, 1,
            "take a regular sample of the input data to speed up tree build "
            "and query times"),
        ParameterListing(
            "features", str,
            header=(
                "Mapping between columns of the input data table and the "
                "columns of the mock pipeline data store. The input features "
                "are used to build the search tree, the data store features "
                "are matched against tree. Key must be a valid column name in "
                "the input table, value an existing column name in the data "
                "store.")),
        ParameterListing(
            "observables", str,
            header=(
                "Mapping between columns of the input data table and the "
                "columns of the mock pipeline data store. The observables of "
                "the input data that are assigned to the nearest neighbour of "
                "an object in the data store. Key must be a valid column name "
                "in the input table, value a pair of output column name in "
                "the data store and fall back value if distance > max_dist.")),
        ParameterListing(
            "fallback", str,
            header=(
                "Fallback value to assign if the nearest neighbour distance > "
                "max_dist. If no value is provided the value of the "
                "observable is assigned instead. Key must match one of the "
                "keys in observables, value must have the same data type as "
                "the observables.")),
        ParameterListing(
            "weights", str,
            header=(
                "Weights are used to scale the feature space along specific "
                "axes. This is useful to give a higher or lower significance "
                "to some features when matching. If no weight is provided for "
                "a feature, no scaling is applied. Key must match one of the "
                "keys in observables, value must have the same data type as "
                "the observables.")),
        header=(
            "This configuration file is required for mocks_match_data. It "
            "defines the features of the mock data and an external data set "
            "to match. Based on the nearest neighbour in feature space, a set "
            "of observables from the external data is assigned to the mock "
            "data. Additional distance, cut-offs, fallback values and "
            "weightes for each feature space dimension can be specified."))

    def _run_checks(self):
        # check that there are no fallback values with no matching observable
        if set(self.observables.keys()) < set(self.fallback.keys()):
            message = "fallback values are provided for undefined observables"
            raise KeyError(message)
        # check that there are no weights with no matching observable
        if set(self.observables.keys()) < set(self.weights.keys()):
            message = "weights are provided for undefined observables"
            raise KeyError(message)

    @property
    def feature_names(self):
        return tuple(sorted(self.features.keys()))

    @property
    def observable_names(self):
        return tuple(sorted(self.observables.keys()))


class DataMatcher(object):

    _feature_dtype = "f8"  # default to full precision for feature space
    _tree = None

    def __init__(self, config, logger=None):
        self._config = config
        try:
            if logger is not None:
                logger.info("opening training data: {:}".format(config.input))
            self._table = MmapTable(config.input)
        except Exception as e:
            if logger is not None:
                logger.handleException(e)
            else:
                raise e

    @property
    def config(self):
        return self._config

    @property
    def feature_names(self):
        return tuple(sorted(self._config.features))

    @property
    def observable_names(self):
        return tuple(sorted(self._config.observables))

    def __enter__(self, *args, **kwargs):
        self.initialize()
        return self
    
    def __exit__(self, *args, **kwargs):
        try:
            self._table.close()
        except AttributeError:
            pass

    def _check_training_features(self):
        max_precision = 2
        for path in self.feature_names:
            dtype = self._table[path].dtype
            precision = dtype.itemsize
            # exclude non-numeric and complex data
            if dtype.kind not in "fiu":
                message = "feature must be of integer or floating point type: "
                message += "{:} (type: {:})"
                raise TypeError(message.format(path, dtype.str))
            # avoid rounding errors
            elif dtype.kind in "iu":
                if precision == 1:  # short
                    precision = 2
                elif precision == 2:  # int
                    precision = 4
                else:  # long
                    precision = 8  # exact representation not guaranteed
            max_precision = max(max_precision, precision)
        # set the data type to use for the feature space
        self._feature_dtype = "f{:d}".format(max_precision)

    def _ingest_fallbacks(self):
        self._fallbacks = {}
        for path, value in self._config.fallback.items():
            dtype = self._table[path].dtype
            try:
                self._fallbacks[path] = dtype.type(value)
            except ValueError:
                message = "fallback value for '{:}' of incorrect type: {:}"
                raise ValueError(message.format(path, value))

    def _get_transform(self):
        self._feature_scales = {
            path: 1.0 for path in self.feature_names}
        # normalize the feature space
        if self._config.normalize:
            for path in self.feature_names:
                self._feature_scales[path] /= np.nanstd(self._table[path])
        # apply the weight scaling
        for path, weight in self._config.weights.items():
            self._feature_scales[path] *= weight

    def initialize(self):
        self._check_training_features()
        self._ingest_fallbacks()
        self._get_transform()
        # paths of data columns that will be machted against the training data
        self._data_features = tuple(
            self._config.features[path] for path in self.feature_names)

    def observable_information(self):
        for path in self.observable_names:
            outpath = self._config.observables[path]
            desc = "matched from column '{:}' from file '{:}' (features: {:})"
            # list with output column name, data type and description
            properties = [
                outpath, self._table[path].dtype, {
                    "description": desc.format(
                        path, self._config.input,
                        ", ".join(self.feature_names))}]
            yield properties

    def transform(self, table, training=False):
        if training:
            paths = self.feature_names
        else:
            paths = self._data_features
        scales = [self._feature_scales[path] for path in self.feature_names]
        # build a contiguous array with the feature data
        transformed = None
        for i, (path, scale) in enumerate(zip(paths, scales)):
            if transformed is None:
                transformed = np.empty(
                    (len(table[path]), len(paths)), dtype=self._feature_dtype)
            transformed[:, i] = table[path] * scale
        return transformed

    def check_features(self, table):
        for path in self.feature_names:
            lookup_path = self._config.features[path]
            kind = table[lookup_path].dtype.kind
            if kind != self._table[path].dtype.kind:
                message = "data type is of different kind than training data: "
                message += "{:}"
                raise TypeError(message.format(lookup_path))

    def build_tree(self):
        samples = self._table[::self._config.every_n]  # apply the thinning
        features = self.transform(samples, True)  # apply scaling and weights
        self._tree = cKDTree(features)

    @Schedule.description("deriving features from observables")
    @Schedule.CPUbound
    @Schedule.threads
    def apply(self, threads=-1, **feature_kwargs):
        features = self.transform(feature_kwargs)
        dist_match, idx_match = self._tree.query(features, k=1, n_jobs=threads)
        # check where the fallback values are needed
        fallback_mask = dist_match > self._config.max_dist
        # get the observables
        observables = []
        for path in self.observable_names:
            observable = self._table[path][idx_match]
            if path in self._fallbacks:
                observable[fallback_mask] = self._fallbacks[path]
            observables.append(observable)
        return tuple(observables)
