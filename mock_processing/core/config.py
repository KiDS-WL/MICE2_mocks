import argparse
import os
import sys
from functools import partial
from tempfile import gettempdir

import toml

from .utils import expand_path


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "default_config")


def load_config(path, logger, description, parser_class, **kwargs):
    full_path = expand_path(path)
    message = "reading {:} configuration file: {:}".format(
        description, full_path)
    logger.info(message)
    try:
        if hasattr(parser_class, "get"):
            return parser_class(full_path, **kwargs).get
        else:
            return parser_class(full_path, **kwargs)
    except OSError as e:
        message = "configuration file not found"
        logger.handleException(e, message)
    except Exception as e:
        message = "malformed configuration file"
        logger.handleException(e, message)


class DumpDefault(argparse.Action):

    _parser_class = lambda *args: None  # dummy

    def __init__(self, *args, nargs=0,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        # verify integrity of the default file
        self._parser_class(self._parser_class._default_path)
        # write the content to the screen
        with open(self._parser_class._default_path) as f:
            for line in f.readlines():
                sys.stdout.write(line)
        parser.exit()


class ParseColumnMap(object):
    """
    Parse a column mapping file used to convert the column names of the input
    files into paths within the data store.

    Parameters:
    -----------
    path : str
        Column map file path
    """

    _default_path = os.path.join(_DEFAULT_CONFIG_PATH, "column_map.toml")

    def __init__(self, path):
        # unpack the potentially nested dictionary by converting nested keys
        # into file system paths
        self._column_map = dict()
        # parse the TOML file and parse it as dictionary
        with open(path) as f:
            self._traverse_dict(toml.load(f))

    @property
    def get(self):
        """
        Get the column mapping data store path -> input file column.
        """
        return self._column_map

    def _traverse_dict(self, subdict, path=""):
        """
        Iterate the directory and insert the values in the column map by
        concatenating nested keywords like file system paths.

        Parameters:
        -----------
        subdict : dict
            Dictionary that maps the input file column names.
        path : str
            path under which the dictionary items are registered in the global
            column map.
        """
        for key, value in subdict.items():
            if type(key) is not str:
                message = "invalid type {:} for set name"
                raise TypeError(message.format(str(type(value))))
            if type(value) is dict:
                self._traverse_dict(value, os.path.join(path, key))
            else:
                if type(value) is list:
                    dtype_tuple = tuple(value)
                else:
                    dtype_tuple = (value, None)
                # remove dummy entries from the default configuration file
                if dtype_tuple[0] == "":
                    continue
                self._column_map[os.path.join(path, key)] = dtype_tuple


class DumpColumnMap(DumpDefault):
    _parser_class = ParseColumnMap


load_column_map = partial(
    load_config, description="column", parser_class=ParseColumnMap)


class ParsePhotometryConfig(object):

    _general_params = {
        "legacy",
        "limit_sigma",
        "no_detect_value",
        "SN_detect",
        "SN_floor",
        # these are filter specific (dictionary) values:
        "limits",
        "PSF"}
    # NOTE: register paramters for new algorithms here
    _algorithm_params = {
        "SExtractor": {
            "phot_autoparams"},
        "GAaP": {
            "aper_min",
            "aper_max"}}
    _default_path = os.path.join(_DEFAULT_CONFIG_PATH, "photometry.toml")

    def __init__(self, config_path):
        # load default to get missing values in user input
        self._parse_config(self._default_path)
        self._parse_config(config_path)
        self.filter_names = sorted(self.PSF.keys())

    def _parse_config(self, config_path):
        with open(config_path) as f:
            config = toml.load(f)
        # determine all allowed top-level parameter names
        known_params = self._general_params.copy()
        for algorithm_name in self._algorithm_params:
            known_params.add(algorithm_name)
        # check for unknown parameters
        diff = set(config.keys()) - known_params
        try:
            param_name = diff.pop()  # successful only if diff is not empty
            message = "unknown photometry paramter: {:}".format(param_name)
            raise ValueError(message)
        except KeyError:
            pass
        # register all parameters as class attributes, algorithms as
        # dictionaries
        for key, value in config.items():
            if key in self._algorithm_params:
                self._parse_algorithm(key, value)
            else:
                setattr(self, key, value)
        # check that each magnitude limit has a corresponding PSF size
        if not self.limits.keys() == self.PSF.keys():
            message = "some filter(s) do not provide pairs of magnitude limit "
            message += "and PSF size"
            raise KeyError(message)

    def _parse_algorithm(self, name, params):
        for key, value in params.items():
            if key not in self._algorithm_params[name]:
                message = "unknown paramter for algorithm '{:}': {:}".format(
                    name, key)
                raise ValueError(message)
        setattr(self, name, params)

    def __str__(self):
        string = ""
        for attr in sorted(self.__dict__):
            if not attr.startswith("_"):
                string += "{:}: {:}\n".format(attr, str(self.__dict__[attr]))
        return string.strip("\n")


class DumpPhotometryConfig(DumpDefault):
    _parser_class = ParsePhotometryConfig


load_photometry_config = partial(
    load_config, description="photometry", parser_class=ParsePhotometryConfig)


class ParseMatchingConfig(object):

    _general_params = {
        "input",
        "normalize",
        "max_dist",
        "every_n",
        # these are filter specific (dictionary) values:
        "features",
        "observables",
        "fallback",
        "weights"}
    # NOTE: register paramters for new algorithms here
    _algorithm_params = {}
    _default_path = os.path.join(_DEFAULT_CONFIG_PATH, "matching.toml")

    def __init__(self, config_path):
        # load default to get missing values in user input
        self._parse_config(self._default_path)
        self._parse_config(config_path)
        self.feature_names = sorted(self.features.keys())
        self.observable_names = sorted(self.observables.keys())

    def _parse_config(self, config_path):
        with open(config_path) as f:
            config = toml.load(f)
        # determine all allowed top-level parameter names
        known_params = self._general_params.copy()
        for algorithm_name in self._algorithm_params:
            known_params.add(algorithm_name)
        # check for unknown parameters
        diff = set(config.keys()) - known_params
        try:
            param_name = diff.pop()  # successful only if diff is not empty
            message = "unknown photometry paramter: {:}".format(param_name)
            raise ValueError(message)
        except KeyError:
            pass
        # register all parameters as class attributes, algorithms as
        # dictionaries
        for key, value in config.items():
            setattr(self, key, value)
        # check that there are no fallback values with no matching observable
        if not self.observables.keys() >= self.fallback.keys():
            message = "fallback values are provided for undefined observables"
            raise KeyError(message)
        # check that there are no weights with no matching observable
        if not self.observables.keys() >= self.weights.keys():
            message = "weights are provided for undefined observables"
            raise KeyError(message)

    def __str__(self):
        string = ""
        for attr in sorted(self.__dict__):
            if not attr.startswith("_"):
                string += "{:}: {:}\n".format(attr, str(self.__dict__[attr]))
        return string.strip("\n")


class DumpMatchingConfig(DumpDefault):
    _parser_class = ParseMatchingConfig


load_matching_config = partial(
    load_config, description="matching", parser_class=ParseMatchingConfig)


class ParseBpzConfig(object):

    _general_params = {
        "BPZpath",
        "BPZenv",
        "BPZtemp",
        "flux",
        "system",
        # these are filter specific (dictionary) values:
        "filters"}
    _param_groups = {
        "prior": {
            "name",
            "filter"},
        "templates": {
            "name",
            "interpolation"},
        "likelihood": {
            "zmin",
            "zmax",
            "dz",
            "odds",
            "min_rms"}}
    _default_path = os.path.join(_DEFAULT_CONFIG_PATH, "BPZ.toml")

    def __init__(self, config_path):
        # load default to get missing values in user input
        self._parse_config(self._default_path)
        self._parse_config(config_path)
        self.filter_names = sorted(self.filters.keys())
        # configure the BPZ executable
        self.BPZpath = expand_path(self.BPZpath)
        if self.BPZenv == "":
            self.BPZenv = "python2"
        else:
            self.BPZenv = expand_path(self.BPZenv)
        if self.BPZtemp == "":
            self.BPZtemp = gettempdir()
        else:
            self.BPZtemp = expand_path(self.BPZtemp)
        if not os.path.isdir(self.BPZpath):
            message = "BPZ install path is not a directory: {:}"
            raise OSError(message.format(self.BPZpath))

    def _parse_config(self, config_path):
        with open(config_path) as f:
            config = toml.load(f)
        # determine all allowed top-level parameter names
        known_params = self._general_params.copy()
        for group_name in self._param_groups:
            known_params.add(group_name)
        # check for unknown parameters
        diff = set(config.keys()) - known_params
        try:
            param_name = diff.pop()  # successful only if diff is not empty
            message = "unknown BPZ paramter: {:}".format(param_name)
            raise ValueError(message)
        except KeyError:
            pass
        # register all parameters as class attributes, algorithms as
        # dictionaries
        for key, value in config.items():
            if key in self._param_groups:
                self._parse_group(key, value)
            elif key == "filters":
                setattr(self, key, {
                    f: expand_path(path) for f, path in value.items()})
            else:
                setattr(self, key, value)
        # check if the prior filter is provided
        if self.prior["filter"] not in self.filters:
            message = "prior filter is not included in the filter list: {:}"
            raise KeyError(message.format(self.prior["filter"]))

    def _parse_group(self, name, params):
        for key, value in params.items():
            if key not in self._param_groups[name]:
                message = "unknown paramter in '{:}': {:}".format(name, key)
                raise ValueError(message)
        setattr(self, name, params)

    def __str__(self):
        string = ""
        for attr in sorted(self.__dict__):
            if not attr.startswith("_"):
                string += "{:}: {:}\n".format(attr, str(self.__dict__[attr]))
        return string.strip("\n")


class DumpBpzConfig(DumpDefault):
    _parser_class = ParseBpzConfig


load_bpz_config = partial(
    load_config, description="BPZ", parser_class=ParseBpzConfig)


class ParseSampleConfig(object):

    def __init__(self, config_path, sample):
        self._sample = sample
        self._default_path = os.path.join(
            _DEFAULT_CONFIG_PATH, "samples", "{:}.toml".format(sample))
        # load default to get missing values in user input
        self._parse_default()
        # get the actual configuration
        self._parse_config(config_path)

    def _parse_default(self):
        # this is needed to figure out the expected parameters and the optional
        # parameters indicated by an empty string
        with open(self._default_path) as f:
            config = toml.load(f)
        if "bitmask" not in config:
            message = "invalid default sample configuration: 'bitmask' field "
            message += "must always be defined"
            raise KeyError(message)
        self._is_required = {
            "bitmask": True}  # list known params and if they are required
        for key, value in config.items():
            if key == "bitmask":
                continue
            elif type(value) is not dict:
                # parameter is not required if it is empty in the default file
                self._is_required[key] = value != ""
            else:  # visit all parameters in group
                self._is_required[key] = {
                    k: v != "" for k, v in value.items()}

    def _parse_config(self, config_path):
        # load the parameter dictionary
        with open(config_path) as f:
            config = toml.load(f)
        self._parse_group(config, self._is_required)
        for key, value in config.items():
            setattr(self, key, value)

    def _parse_group(self, parameters, is_required):
        # check that only the expected parameters exist
        if parameters.keys() != is_required.keys():
            message = "configuration for '{:}' contains invalid parameters"
            raise KeyError(message.format(self._sample))
        # check if all required parameters are set
        for key, value in parameters.items():
            if type(value) is dict:
                self._parse_group(value, is_required[key])
            elif is_required[key] and value == "":
                message = "parameter is required but not set: {:}".format(key)
                raise ValueError(message)
            elif value == "":  # substitute None for optional parameters
                parameters[key] = None

    def __str__(self):
        string = ""
        for attr in sorted(self.__dict__):
            if not attr.startswith("_"):
                string += "{:}: {:}\n".format(attr, str(self.__dict__[attr]))
        return string.strip("\n")


class DumpSampleConfig(argparse.Action):

    _parser_class = lambda *args: None  # dummy

    def __init__(self, *args, nargs=1,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        sample = values[0]
        # write the content to the screen
        default_path = os.path.join(
            _DEFAULT_CONFIG_PATH, "{:}.toml".format(sample))
        with open(default_path) as f:
            for line in f.readlines():
                sys.stdout.write(line)
        parser.exit()


load_sample_config = partial(
    load_config, description="BPZ", parser_class=ParseSampleConfig)
