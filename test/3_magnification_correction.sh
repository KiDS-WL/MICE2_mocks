#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

if [ "$(whoami)" == "janluca" ]
then
    ../scripts/mocks_magnification \
        /home/janluca/dev/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL \
        --mag mags/evolved \
        --lensed mags/lensed
else
    ../scripts/mocks_magnification \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
        --mag mags/evolved \
        --lensed mags/lensed
fi
