import argparse
import logging
import os
import string
import textwrap

import toml

from galmock.core.utils import expand_path


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_TOML_KEY_CHARACTERS = string.ascii_letters + string.digits + "_-"
_WRAP = 79
_INDENT = 24


def logging_config(logpath, overwrite=False):
    logconfig = {
        "version": 1,
        "disable_existing_loggers": False,

        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
            },
        },

        "formatters": {
            "default": {
                "format": \
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },

        "handlers": {
            "console": {
                "level": "DEBUG",
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
            "file": {
                "level": "DEBUG",
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": logpath,
                "mode": "w" if overwrite else "a",
            },
        },
    }
    return logconfig


class LineComment(object):

    def __init__(self, comment):
        self.comment = comment

    def __str__(self):
        formatted = ""
        # preserve custom white spaces
        for paragraph in self.comment.splitlines():
            if paragraph.strip() == "":
                continue
            for line in textwrap.wrap(
                    paragraph, _WRAP - 2, replace_whitespace=False):
                formatted += "# {:}\n".format(line)
        return formatted.rstrip("\n")


class Parameter(object):

    def __init__(self, name, dtype, value=None, comment=None, parser=None):
        self.name = name
        self.type = dtype
        self.value = value
        self.required = value is not None
        self.parser = parser
        self.comment = comment

    def __str__(self):
        formatted = ""
        # format the identifier
        if all(c in _TOML_KEY_CHARACTERS for c in self.name):
            formatted += self.name
        else:
            formatted += '"{:}"'.format(self.name)
        # format the value
        if self.value is None:
            value = '""'
        elif type(self.value) is bool:
            value = str(self.value).lower()
        elif type(self.value) is str:
            value = '"{:}"'.format(self.value)
        else:
            value = str(self.value)
        formatted += " = {:}".format(value)
        if len(formatted) > _INDENT - 2:
            formatted += "\n"
        else:
            formatted = formatted.ljust(_INDENT)
        # the comment is more complicated
        if self.comment is not None:
            for line in textwrap.wrap(self.comment, _WRAP - _INDENT - 2):
                if formatted.endswith("\n"):
                    formatted += "{:}# {:}\n".format(" " * _INDENT, line)
                else:
                    formatted += "# {:}\n".format(line)
        return formatted.rstrip("\n")


class ParameterListing(object):

    def __init__(self, name, dtype, header=None, parser=None):
        self.name = name
        self.type = dtype
        self.header = header
        self.required = True
        self.parser = parser

    def __str__(self):
        formatted = "\n"
        formatted += "[{:}]\n".format(self.name)
        # add the header
        if self.header is not None:
            formatted += str(LineComment(self.header))
        return formatted.rstrip("\n")

    def get_required(self):
        return []

    def get_optional(self):
        return []

    def is_known(self, name):
        return True


class ParameterGroup(object):

    _allow_groups = False
    _offset_header = False

    def __init__(
            self, name, *entries, header=None, required=True,
            allow_unknown=False):
        self.name = name
        self.entries = entries
        self.header = header
        self.required = required
        self.allow_unknown = allow_unknown
        assert(len(self) == len(set(self.get_known())))
        for entry in self:
            if type(entry) is ParameterGroup and not self._allow_groups:
                raise TypeError("Group may not contain sub-groups")

    def __len__(self):
        return sum(type(entry) is not LineComment for entry in self)

    def __iter__(self):
        for entry in self.entries:
            yield entry

    def __getitem__(self, item):
        for entry in self:
            try:
                if entry.name == item:
                    return entry
            except AttributeError:
                continue
        else:
            raise KeyError("entry not found: {:}".format(item))

    def __str__(self):
        formatted = "\n"
        if self.name is not None:
            formatted += "[{:}]\n".format(self.name)
        if self.header is not None:
            formatted += str(LineComment(self.header)) + "\n"
        if self._offset_header:
            formatted += "\n"
        for entry in self:
            formatted += "{:}\n".format(str(entry))
        return formatted.rstrip("\n")

    def get_known(self):
        return [entry.name for entry in self if type(entry) is not LineComment]

    def get_required(self):
        required = []
        for name in self.get_known():
            entry = self[name]
            if entry.required:
                required.append(name)
        return required

    def get_optional(self):
        optional = []
        for name in self.get_known():
            entry = self[name]
            if not entry.required:
                required.append(name)
        return required

    def is_known(self, name):
        if self.allow_unknown:
            return True  # match anything
        else:
            return name in self.get_known()


class ParameterCollection(ParameterGroup):

    _allow_groups = True
    _offset_header = True

    def __init__(self, *entries, header=None, allow_unknown=False):
        super().__init__(
            None, *entries, header=header, allow_unknown=allow_unknown)

    def __str__(self):
        return super().__str__().lstrip("\n")


class Parser(object):

    default = ParameterCollection()

    def __init__(self, path):
        full_path = expand_path(path)
        message = "reading configuration file: {:}".format(full_path)
        logger.info(message)
        # parse the TOML configuration file
        try:
            with open(full_path) as f:
                config = toml.load(f)
            self._config = self._parse_group(config, self.default)
        except OSError as e:
            logger.exception("configuration file not found")
            raise
        except Exception as e:
            logger.exception("malformed configuration file")
            raise
        # bind the top level entries to the Parser instance
        for name, value in self._config.items():
            setattr(self, name, value)
        # run some configuration specific tests
        self._run_checks()

    @staticmethod
    def _parse_param(param, name, dtype, required, parser):
        if required and type(param) is not dtype:
            message = "expected type {:} but received {:}: {:}"
            raise TypeError(message.format(dtype, type(param), name))
        if not required and param == "":
            return None
        elif parser is not None:
            return parser(param)
        else:
            return param

    def _parse_group(self, group, reference):
        parsed = {}
        for name in reference.get_required():
            if name not in group:
                raise KeyError("missing paramter: {:}".format(name))
        # parse the entries
        for name in group:
            if not reference.is_known(name):
                raise KeyError("unknown parameter: {:}".format(name))
            entry = group[name]
            if type(entry) is dict:
                if getattr(reference, "allow_unknown", False):
                    parsed[name] = entry
                else:
                    parsed[name] = self._parse_group(entry, reference[name])
            elif type(reference) is ParameterListing:
                parsed[name] = self._parse_param(
                    entry, name, reference.type, False, reference.parser)
            else:
                if getattr(reference, "allow_unknown", False):
                    parsed[name] = entry
                else:
                    ref = reference[name]
                    parsed[name] = self._parse_param(
                        entry, name, ref.type, ref.required, ref.parser)
        return parsed

    def _run_checks(self):
        pass

    def __str__(self):
        formatted = ""
        for name in sorted(self._config.keys()):
            formatted += "{:} = {:}\n".format(name, self._config[name])
        return formatted.strip("\n")

    @classmethod
    def get_dump(cls):
        class dump(argparse.Action):

            def __init__(self, *args, nargs=0,**kwargs):
                super().__init__(*args, nargs=nargs, **kwargs)
                self.default = cls.default

            def __call__(
                    self, parser, namespace, values, option_string, **kwargs):
                print(self.default)
                parser.exit()

        return dump


class TableParser(Parser):

    default = ParameterCollection(
        Parameter(
            "index", (str, (str, str)), None,
            "recommended name for a unique galaxy index"),
        ParameterGroup(
            "position",
            LineComment("true galaxy position"),
            Parameter(
                "ra/true", (str, (str, str)), None,
                "recommended name for true right ascension"),
            Parameter(
                "dec/true", (str, (str, str)), None,
                "recommended name for true declination"),
            Parameter(
                "z/true", (str, (str, str)), None,
                "true redshift, required for MICE2 evolution correction"),
            LineComment(
                "observed galaxy position (including redshift space "
                "distortions and lensing)"),
            Parameter(
                "ra/obs", (str, (str, str)), None,
                "recommended name for observed right ascension"),
            Parameter(
                "dec/obs", (str, (str, str)), None,
                "recommended name for observed declination"),
            Parameter(
                "z/obs", (str, (str, str)), None,
                "recommended name for observed redshift"),
            allow_unknown=True),
        ParameterGroup(
            "shape",
            Parameter(
                "axis_ratio", (str, (str, str)), None,
                "b/a axis ratio, required for aperture computation"),
            Parameter(
                "bulge/fraction", (str, (str, str)), None,
                "bulge to total luminosity ratio, required for two-component "
                "effective radius"),
            Parameter(
                "bulge/size", (str, (str, str)), None,
                "projected bulge major axis size in arcsec, required for "
                "two-component effective radius"),
            Parameter(
                "disk/size", (str, (str, str)), None,
                "projected disk major axis size in arcsec, required for "
                "two-component effective radius"),
            allow_unknown=True),
        ParameterGroup(
            "lensing",
            Parameter(
                "gamma1", (str, (str, str)), None),
            Parameter(
                "gamma2", (str, (str, str)), None),
            Parameter(
                "kappa", (str, (str, str)), None,
                "convergence, required to apply flux magnification"),
            allow_unknown=True),
        ParameterGroup(
            "environ",
            Parameter(
                "is_central", (str, (str, str)), None,
                "boolean flag indicating whether a galaxy is the centeral "
                "galaxy of a given halo, required by SDSS QSO spec-z "
                "selection"),
            Parameter(
                "log_M_halo", (str, (str, str)), None,
                "logarithmic halo mass, required by SDSS QSO spec-z "
                "selection"),
            Parameter(
                "log_M_stellar", (str, (str, str)), None,
                "logarithmic stellar mass, required by SDSS QSO spec-z "
                "selection"),
            Parameter(
                "n_sattelites", (str, (str, str)), None),
            allow_unknown=True),
        header=(
            "This file is required for mocks_init_pipeline and establishes a "
            "mapping between column or data set names of the input file and "
            "the pipeline data store:\n"
            "    path/in/data/store = \"input column name\"\n"
            "Data types may be specified using the following syntax:\n"
            "    path/in/data/store = [\"input column name\", \"type code\"]\n"
            "where \"type code\" is a valid python numpy data type identifier "
            "(e.g. \"bool\" for boolean values, \"i8\" for 64 bit integeres "
            "or \"f4\" for 32-bit floating point values)."),
        allow_unknown=True)

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

    @property
    def column_map(self):
        self._column_map = {}
        self._traverse_dict(self._config)
        return self._column_map
