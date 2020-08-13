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

../../scripts/mocks_apertures \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c config/photometry.toml \
    --method GAaP \
    --threads ${2:-64}
echo
../../scripts/mocks_apertures \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c config/photometry_old.toml \
    --method SExtractor \
    --threads ${2:-64}
echo
