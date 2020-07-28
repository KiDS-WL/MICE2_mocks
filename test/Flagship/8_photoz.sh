#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test"
else
    suffix=""
fi

export hostname=$HOSTNAME
../../scripts/mocks_BPZ \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -c BPZ.toml \
    --mag mags/KV450 \
    --zphot BPZ/KV450 \
    --threads 64
