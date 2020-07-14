#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
    config=MICE2.toml
    data=/net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL.fits
elif [ "$1" == "FS" ]
then
    table="FS"
    config=Flagship.toml
    data=/net/home/fohlen12/jlvdb/DATA/Flagship_KiDS/Flagship_v1-8-4_deep_ugrizYJHKs_shapes_halos_WL.fits
else
    table="256th"
    config=MICE2.toml
    data=/net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL.fits
fi

../scripts/mocks_init_pipeline \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -i $data \
    -c $config \
    --purge
