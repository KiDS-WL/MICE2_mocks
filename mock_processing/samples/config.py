from argparse import Action

from mock_processing.core.config import Parameter
from .base import SampleManager


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
    param = Parameter(
        "bitmask", str, "samples/" + sample,
        "path at which the sample selection bit mask is saved in the data "
        "store")
    return param


class DumpConfig(Action):

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
