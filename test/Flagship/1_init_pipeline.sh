#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
elif [ "$1" == "all" ]
then
    suffix=""
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"all\""
    exit 1;
fi

../../scripts/mocks_init_pipeline \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -i /net/home/fohlen13/jlvdb/DATA/Flagship_downloads/Flagship_v1-8-4_deep_ugrizYJHKs_shapes_halos_WL${suffix}.fits \
    -c Flagship.toml \
    --purge
