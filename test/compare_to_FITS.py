#!/usr/bin/env python3
import sys

import numpy as np
from astropy.io import fits as pyfits

from memmap_table import MemmapTable


with pyfits.open(sys.argv[1]) as fits:
    fits_data = fits[1].data
    fits_col = sys.argv[3]

    with MemmapTable(sys.argv[2]) as table:
        table_col = sys.argv[4]

        all_equal = np.all(table[table_col] == fits_data[fits_col])
        print("{:} matches {:}:".format(fits_col, table_col), all_equal)

        if not all_equal:
            print(table[table_col] - fits_data[fits_col])
