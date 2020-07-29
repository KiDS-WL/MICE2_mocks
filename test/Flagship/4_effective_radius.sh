#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
else
    suffix=""
fi

../../scripts/mocks_effective_radius \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    --threads 60
