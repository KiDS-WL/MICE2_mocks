#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
elif [ "$1" == "test" ]
then
    table="256th"
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"deep\""
    exit 1;
fi

../../scripts/mocks_init_pipeline \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -i /net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_${table}_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
    -c config/MICE2.toml \
    --purge
