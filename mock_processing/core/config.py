import argparse
import os
import string
import textwrap

import toml

from .logger import DummyLogger
from .utils import expand_path


_TOML_KEY_CHARACTERS = string.ascii_letters + string.digits + "_-"
_WRAP = 79
_INDENT = 24


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

    def __init__(self, path, logger=DummyLogger()):
        full_path = expand_path(path)
        message = "reading configuration file: {:}".format(full_path)
        logger.info(message)
        # parse the TOML configuration file
        try:
            with open(full_path) as f:
                config = toml.load(f)
            self._config = self._parse_group(config, self.default)
        except OSError as e:
            message = "configuration file not found"
            logger.handleException(e, message)
        except Exception as e:
            message = "malformed configuration file"
            logger.handleException(e, message)
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


class Table_Parser(Parser):

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


class Photometry_Parser(Parser):

    default = ParameterCollection(
        Parameter(
            "legacy", bool, False,
            "use legacy mode (van den Busch et al. 2020)"),
        Parameter(
            "limit_sigma", float, 1.0,
            "sigma value of the detection (magnitude) limit with respect to "
            "the sky background"),
        Parameter(
            "no_detect_value", float, 99.0,
            "magnitude value assigned to undetected galaxies"),
        Parameter(
            "SN_detect", float, 1.0,
            "signal-to-noise ratio detection limit"),
        Parameter(
            "SN_floor", float, 0.2,
            "numerical lower limit for signal-to-noise ratio"),
        ParameterGroup(
            "intrinsic",
            Parameter(
                "r_effective", str, "shape/R_effective",
                "path of the effective radius column in the data store"),
            Parameter(
                "flux_frac", float, 0.5,
                "this defines the effective radius by setting the fraction of "
                "the total flux/luminosity for which the radius of the source "
                "is computed"),
            header=None),
        ParameterListing(
            "limits", float,
            header=(
                "numerical values for the magnitude limits, the keys must be "
                "the same as in column map file used for "
                "mocks_init_pipeline")),
        ParameterListing(
            "PSF", float,
            header=(
                "numerical values for the PSF FWHM in arcsec, the keys must "
                "be the same as in column map file used for "
                "mocks_init_pipeline")),
        ParameterGroup(
            "GAaP",
            Parameter(
                "aper_min", float, 0.7,
                "GAaP lower aperture size limit in arcsec"),
            Parameter(
                "aper_max", float, 2.0,
                "GAaP upper aperture size limit in arcsec"),
            header=None),
        ParameterGroup(
            "SExtractor",
            Parameter(
                "phot_autoparams", float, 2.5,
                "MAG_AUTO-like scaling factor for Petrosian radius, here "
                "applied to intrinsic galaxy size derived from effective "
                "radius"),
            header=None),
        header=(
            "This configuration file is required for mocks_apertures and "
            "mocks_photometry. It defines the free parameters of the aperture "
            "and photometry methods as well as observational PSF sizes and "
            "magnitude limits."))

    def _run_checks(self):
        if set(self.limits.keys()) != set(self.PSF.keys()):
            message = "some filter(s) do not provide pairs of magnitude limit "
            message += "and PSF size"
            raise KeyError(message)

    @property
    def filter_names(self):
        return tuple(sorted(self.limits.keys()))


class Matching_Parser(Parser):

    default = ParameterCollection(
        Parameter(
            "input", str, "...",
            "path to input table, must be MemmapTable compatible",
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


class BPZ_Parser(Parser):

    default = ParameterCollection(
        Parameter(
            "BPZpath", str, "~/src/bpz-1.99.3",
            "path of the bpz installation to use",
            parser=expand_path),
        Parameter(
            "BPZenv", str, None,
            "path to a python2 environment used when calling BPZ (leave blank "
            "to use the default python2 interpreter)",
            parser=expand_path),
        Parameter(
            "BPZtemp", str, None,
            "temporary directory to use for bpz (I/O intense, e.g. a ram disk, "
            "if blank, defaults to data store)",
            parser=expand_path),
        Parameter(
            "flux", bool, False,
            "whether the input columns are fluxes or magnitudes"),
        Parameter(
            "system", str, "AB",
            "photometric system, must be \"AB\" or \"Vega\""),
        ParameterListing(
            "filters", str,
            header=(
                "Mapping between filter keys (same as used in the column map "
                "file for mocks_init_pipeline) and paths to transmission "
                "curve files compatible with BPZ (do not need to be in the "
                "BPZpath/FILTER directory)."),
            parser=expand_path),
        ParameterGroup(
            "prior",
            Parameter(
                "name", str, "hdfn_gen",
                "template-magnitude prior module in the bpz_path directory "
                "(filename: prior_[name].py)"),
            Parameter(
                "filter", str, "i",
                "filter key (one of those listed in the [filters] section) "
                "based on which the prior is evaluated"),
            header=None),
        ParameterGroup(
            "templates",
            Parameter(
                "name", str, "CWWSB4",
                "name of the template .list file in the bpz_path/SED "
                "directory"),
            Parameter(
                "interpolation", int, 10,
                "introduces n points of interpolation between the templates "
                "in the color space"),
            header=None),
        ParameterGroup(
            "likelihood",
            Parameter(
                "zmin", float, 0.01,
                "minimum redshift probed"),
            Parameter(
                "zmax", float, 7.00,
                "maximum redshift probed"),
            Parameter(
                "dz", float, 0.01,
                "redshift resolution, intervals are logarithmic: (1+z)*dz"),
            Parameter(
                "odds", float, 0.68,
                "redshift confidence limits"),
            Parameter(
                "min_rms", float, 0.0,
                "intrinsic scatter of the photo-z in dz/(1+z)"),
            header=None),
        header=(
            "This configuration file is required for mocks_BPZ. It defines "
            "the transmission curves, galaxy templates and Bayesian prior "
            "required to run the BPZ photometric reshift code."))

    def _run_checks(self):
        if self.prior["filter"] not in self.filters:
            message = "prior filter is not included in the filter list: {:}"
            raise KeyError(message.format(self.prior["filter"]))

    @property
    def filter_names(self):
        return tuple(sorted(self.filters.keys()))
