#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
else
    suffix=""
fi

../../scripts/Flagship_flux_to_magnitudes \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    --flux flux/model \
    --mag mags/model
