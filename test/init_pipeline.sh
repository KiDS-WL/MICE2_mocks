#!/usr/bin/env bash

if [ -f /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits ]
then
    ../scripts/mocks_init_pipeline \
        /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL \
        -i /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
else
    ../scripts/mocks_init_pipeline \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap \
        -i /net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
fi
