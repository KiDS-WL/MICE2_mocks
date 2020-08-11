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

export hostname=$HOSTNAME
../../scripts/mocks_BPZ \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c config/BPZ.toml \
    --mag mags/K1000 \
    --zphot BPZ/K1000 \
    --threads ${2:-64}
echo
