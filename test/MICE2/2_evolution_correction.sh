#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

../../scripts/MICE2_evolution_correction \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    --mag mags/model \
    --evo mags/evolved
