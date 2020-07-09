#!/usr/bin/env bash

if [ -f /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits ]
then
    #export PYTHONPATH=$(python -c "import os; print(os.path.abspath('../..'))"):$PYTHONPATH
    ../pipeline/mocks_init_pipeline \
        /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL \
        -i /home/janluca/dev/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
else
    ../pipeline/mocks_init_pipeline \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap \
        -i /net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits \
        -c MICE2.toml \
        --purge
fi
