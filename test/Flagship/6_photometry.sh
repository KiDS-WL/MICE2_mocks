#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test"
else
    suffix=""
fi

../../scripts/mocks_photometry \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c ../KV450_photometry.toml \
    --method GAaP \
    --mag mags/lensed \
    --real mags/KV450
