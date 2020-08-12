#!/usr/bin/env bash

if [ "$1" == "all" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_all
    prefix=/net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_all
elif [ "$1" == "deep" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_crop
    prefix=/net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_deep
elif [ "$1" == "test" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse
    prefix=/net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_test
else
    echo "ERROR: invalid setup \"$1\", must be \"test\", \"deep\" or \"all\""
    exit 1;
fi

# this selects 24.4 deg^2
footprint='position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0'

../../scripts/mocks_datastore_query $DS --verify \
    -o ${prefix}.fits \
    -q "$footprint"
echo

../../scripts/mocks_datastore_query $DS \
    -o ${prefix}_KV450.fits \
    -q "${footprint}"' AND samples/KiDS & 1'
echo

for sample in 2dFLenS GAMA SDSS
do
    ../../scripts/mocks_datastore_query $DS \
        -o ${prefix}_${sample}.fits \
        -q "${footprint}"' AND samples/'"${sample}"' & 1'
    echo
done

../../scripts/mocks_datastore_query $DS \
    -o ${prefix}_BOSS.fits \
    -q "${footprint}"' AND samples/SDSS & 12'
echo

if [ "$1" == "deep" ]
then
    for sample in WiggleZ DEEP2 VVDSf02 zCOSMOS
    do
        ../../scripts/mocks_datastore_query $DS \
            -o ${prefix}_${sample}.fits \
            -q "${footprint}"' AND samples/'"${sample}"' & 1'
        echo
    done
fi
