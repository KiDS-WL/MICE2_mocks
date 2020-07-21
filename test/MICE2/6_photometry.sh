#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

../../scripts/mocks_photometry \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -c ../KV450_legacy_photometry.toml \
    --method SExtractor \
    --mag mags/lensed \
    --real mags/KV450
