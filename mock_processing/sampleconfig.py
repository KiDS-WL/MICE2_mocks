import argparse

from .core.config import (LineComment, Parameter, ParameterGroup,
                           ParameterCollection, Parser)


REGISTERED_SAMPLES = {}


def register(parser):
    """
    Register selection function parsers.
    """
    REGISTERED_SAMPLES[parser.name] = parser
    return parser


class DumpConfig(argparse.Action):

    def __init__(self, *args, nargs=1,**kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        sample = values[0]
        try:
            sample_parser = REGISTERED_SAMPLES[sample]
        except KeyError:
            message = "sample '{:}' does not implement a configuration"
            raise parser.error(message.format(sample))
        print(sample_parser.default)
        parser.exit()


def make_bitmask(sample):
    param = Parameter(
        "bitmask", str, "samples/" + sample,
        "path at which the sample selection bit mask is saved in the data "
        "store")
    return param


header = (
    "This configuration file is required for mocks_select_sample. It "
    "defines the output column name of the selection bit mask and the "
    "data columns for the selection functions (such as magnitudes or "
    "redshift).")
selection_header = (
    "Mapping form keyword argument names in selection function to "
    "column path in the data store. Optional arguments can be "
    "left blank.")
param_redshift = Parameter("redshift", str, "...", "(observed) redshifts")
param_sdss = {}
for key in ("g", "r", "i", "Z", "Ks"):
    param_sdss[key] = Parameter(
        "mag_" + key, str, "...", "SDSS {:}-band magnitude".format(key))
param_johnson = {}
for key in ("B", "Rc", "Ic"):
    param_johnson[key] = Parameter(
        "mag_" + key, str, "...", "Johnson {:}-band magnitude".format(key))


@register
class Parser_KiDS(Parser):

    name = "KiDS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            Parameter(
                "recal_weight", str, "lensing/recal_weight",
                "path of lensing weight in the data store"),
            LineComment(
                "setting the following paramter will remove objects with "
                "non-detections"),
            Parameter(
                "prior_magnitude", str, None,
                "path of BPZ prior magnitude in the data store"),
            header=selection_header),
        header=header)


@register
class Parser_2dFLenS(Parser):

    name = "2dFLenS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            param_sdss["Z"],
            param_sdss["Ks"],
            header=selection_header),
        header=header)


@register
class Parser_GAMA(Parser):

    name = "GAMA"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["r"],
            header=selection_header),
        header=header)


@register
class Parser_SDSS(Parser):

    name = "SDSS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            Parameter(
                "is_central", str, "environ/is_central",
                "flag indicating if it is central host galaxy"),
            Parameter(
                "lmhalo", str, "environ/log_M_halo",
                "logarithmic halo mass"),
            Parameter(
                "lmstellar", str, "environ/log_M_stellar",
                "logarithmic stellar mass"),
            header=selection_header),
        header=header)


@register
class Parser_WiggleZ(Parser):

    name = "WiggleZ"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_sdss["g"],
            param_sdss["r"],
            param_sdss["i"],
            param_sdss["Z"],
            header=selection_header),
        header=header)


@register
class Parser_DEEP2(Parser):

    name = "DEEP2"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_johnson["B"],
            param_johnson["Rc"],
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
class Parser_VVDSf02(Parser):

    name = "VVDSf02"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
class Parser_zCOSMOS(Parser):

    name = "zCOSMOS"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_redshift,
            param_johnson["Ic"],
            header=selection_header),
        header=header)


@register
class Parser_Sparse24mag(Parser):

    name = "Sparse24mag"
    default = ParameterCollection(
        make_bitmask(name),
        ParameterGroup(
            "selection",
            param_sdss["r"],
            header=selection_header),
        header=header)
