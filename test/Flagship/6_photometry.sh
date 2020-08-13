#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test"
elif [ "$1" == "all" ]
then
    suffix=""
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"all\""
    exit 1;
fi

../../scripts/mocks_photometry \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c config/photometry.toml \
    --method GAaP \
    --mag mags/lensed \
    --real mags/K1000new \
    --threads ${2:-64}
echo
../../scripts/mocks_photometry \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c config/photometry.toml \
    --method SExtractor \
    --mag mags/lensed \
    --real mags/K1000 \
    --threads ${2:-64}
echo
