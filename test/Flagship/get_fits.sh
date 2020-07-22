#!/usr/bin/env bash

../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/Flagship_KiDS \
    -o ~/TEST/Flagship_query.fits \
    -q "position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0"

# topcat ~/TEST/Flagship_query.fits ~/TEST/MICE2_query_reference.fits
