#!/usr/bin/env bash

mocks_init_pipeline \
    -i /net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
    -c MICE2.toml \
    -o /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap \
    --purge
