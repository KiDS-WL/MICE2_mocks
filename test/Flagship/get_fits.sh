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

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/Flagship_query${suffix}.fits \
    -q "position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0"
