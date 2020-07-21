import argparse
import os
import sys

import toml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "default_config")


class ParseColumnMap(object):
    """
    Parse a column mapping file used to convert the column names of the input
    files into paths within the data store.

    Parameters:
    -----------
    path : str
        Column map file path
    """

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


class DumpColumnMap(argparse.Action):

    def __init__(self, *args, nargs=0,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        path = os.path.join(_DEFAULT_CONFIG_PATH, "column_map.toml")
        # verify integrity of the default file
        ParseColumnMap(path)
        # write the content to the screen
        with open(path) as f:
            for line in f.readlines():
                sys.stdout.write(line)
        parser.exit()


class ParsePhotometryConfig(object):

    _general_params = {
        "SN_detect", "SN_floor", "limit_sigma", "limits", "PSF"}
    # NOTE: register paramters for new algorithms here
    _algorithm_params = {
        "SExtractor": {"phot_autoparams"},
        "GAaP": {"aper_min", "aper_max"}}

    def __init__(self, config_path):
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
        self.filter_names = sorted(self.PSF.keys())

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


class DumpPhotometryConfig(argparse.Action):

    def __init__(self, *args, nargs=0,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        path = os.path.join(_DEFAULT_CONFIG_PATH, "photometry.toml")
        # verify integrity of the default file
        ParsePhotometryConfig(path)
        # write the content to the screen
        with open(path) as f:
            for line in f.readlines():
                sys.stdout.write(line)
        parser.exit()
