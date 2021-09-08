from argparse import Action

from galmock.core.config import Parameter
from galmock.samples.base import SampleManager


# define some commonly used statements in the configuration file language
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
for key in ("u", "g", "r", "i", "Z", "Y", "J", "H", "Ks"):
    param_sdss[key] = Parameter(
        "mag_" + key, str, "...", "SDSS {:}-band magnitude".format(key))
param_johnson = {}
for key in ("B", "Rc", "Ic"):
    param_johnson[key] = Parameter(
        "mag_" + key, str, "...", "Johnson {:}-band magnitude".format(key))


def make_bitmask_parameter(sample):
    """
    Generates a parameter definition for the sample configuration file. It sets
    the name of the output column in the data store for a bit mask that is used
    to store the sample selection bits.

    Parameters:
    -----------
    sample : str
        Name of the sample described by the bit mask.
    
    Returns:
    --------
    param : galmock.core.config.Parameter
        The generated paramter definition for the sample configuration file.
    """
    param = Parameter(
        "bitmask", str, "samples/" + sample,
        "path at which the sample selection bit mask is saved in the data "
        "store")
    return param


class DumpConfig(Action):
    """
    An argparse.Action class that is used to generate the default layout of the
    sample configuration file (--dump). Two additional parameters are parsed,
    flavour and sample that are required to locate the correct configuration
    parser.
    """

    def __init__(self, *args, nargs=2, **kwargs):
        super().__init__(*args, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        flavour = values[0]
        sample = values[1]
        try:
            sample_parser = SampleManager.get_parser(flavour, sample)
        except NotImplementedError as e:
            raise parser.error(str(e))
        print(sample_parser.default)
        parser.exit()
