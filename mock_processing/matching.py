import numpy as np
from scipy.spatial import cKDTree

from memmap_table import MemmapTable


class DataMatcher(object):

    _feature_dtype = "f8"  # default to full precision for feature space
    _tree = None

    def __init__(self, config, logger=None):
        self._config = config
        try:
            if logger is not None:
                logger.info("opening training data: {:}".format(config.input))
            self._table = MemmapTable(config.input)
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
        return tuple(sorted(self._config.feature_names))

    @property
    def observable_names(self):
        return tuple(sorted(self._config.observable_names))

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
                transformed = np.empty((len(table[path]), len(paths)))
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
