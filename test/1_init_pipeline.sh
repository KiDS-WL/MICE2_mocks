#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

if [ "$(whoami)" == "janluca" ]
then
    ../scripts/mocks_init_pipeline \
        /home/janluca/dev/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL \
        -i /home/janluca/dev/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
else
    ../scripts/mocks_init_pipeline \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
        -i /net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
fi
