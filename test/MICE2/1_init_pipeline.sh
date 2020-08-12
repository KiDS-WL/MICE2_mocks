#!/usr/bin/env bash

if [ "$1" == "all" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_all
    input=/net/home/fohlen12/jlvdb/DATA/MICE2_KV_full/MICE2_all_uBgVrRciIcYJHKs_shapes_halos_WL.fits
elif [ "$1" == "deep" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_crop
    input=/net/home/fohlen12/jlvdb/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits
elif [ "$1" == "test" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse
    input=/net/home/fohlen12/jlvdb/DATA/MICE2_KV_full/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits
else
    echo "ERROR: invalid setup \"$1\", must be \"test\", \"deep\" or \"all\""
    exit 1;
fi

../../scripts/mocks_init_pipeline $DS \
    -i $input \
    -c config/MICE2.toml \
    --purge
echo
