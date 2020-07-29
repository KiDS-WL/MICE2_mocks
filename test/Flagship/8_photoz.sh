#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test_every7"
else
    suffix=""
fi

export hostname=$HOSTNAME
../../scripts/mocks_BPZ \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c BPZ.toml \
    --mag mags/K1000 \
    --zphot BPZ/K1000 \
    --threads 64
