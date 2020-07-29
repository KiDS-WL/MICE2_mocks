#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

export hostname=$HOSTNAME
../../scripts/mocks_BPZ \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -c BPZ.toml \
    --mag mags/KV450 \
    --zphot BPZ/KV450 \
    --threads 32
