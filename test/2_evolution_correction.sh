#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

if [ "$(whoami)" == "janluca" ]
then
    ../scripts/MICE_evolution_correction \
        /home/janluca/dev/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL \
        --mag mags/model \
        --evo mags/evolved
else
    ../scripts/MICE_evolution_correction \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap \
        --mag mags/model \
        --evo mags/evolved
fi
