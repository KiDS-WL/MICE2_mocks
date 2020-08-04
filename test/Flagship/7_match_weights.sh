#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
else
    suffix=""
fi

../../scripts/mocks_match_data \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c matching.toml \
    --threads 64
