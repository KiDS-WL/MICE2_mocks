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

../../scripts/mocks_effective_radius \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    --threads ${2:-64}
echo
