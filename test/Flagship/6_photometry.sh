#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
else
    suffix=""
fi

../../scripts/mocks_photometry \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c photometry.toml \
    --method GAaP \
    --mag mags/lensed \
    --real mags/K1000
