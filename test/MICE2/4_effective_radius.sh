#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

../../scripts/mocks_effective_radius \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_deep \
    --threads 60
