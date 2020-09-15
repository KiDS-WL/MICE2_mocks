#
# This module implements selection functions and density samplers for different
# galaxy samples with modifications specific to the Flagship galaxy mock
# catalogue.
#
# Each class here should be subclassing an existsing reference implementation
# (see reference.py) overwrite their methods as necessary.
#

import numpy as np

from galmock.core.bitmask import BitMaskManager as BMM
from galmock.samples import reference
from galmock.samples.base import SampleManager
