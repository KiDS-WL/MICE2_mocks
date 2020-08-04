#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

../../scripts/mocks_match_data \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -c matching.toml \
    --threads 32
