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

# this selects 24.4 deg^2
footprint='position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0'

../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} --verify \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}.fits \
    -q "$footprint"
echo

# topcat /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}.fits /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_reference.fits

../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_KV450.fits \
    -q "${footprint}"' AND samples/KiDS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_2dFLenS.fits \
    -q "${footprint}"' AND samples/2dFLenS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_GAMA.fits \
    -q "${footprint}"' AND samples/GAMA & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_SDSS.fits \
    -q "${footprint}"' AND samples/SDSS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_BOSS.fits \
    -q "${footprint}"' AND samples/SDSS & 12'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_WiggleZ.fits \
    -q "${footprint}"' AND samples/WiggleZ & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_DEEP2.fits \
    -q "${footprint}"' AND samples/DEEP2 & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_VVDSf02.fits \
    -q "${footprint}"' AND samples/VVDSf02 & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}_zCOSMOS.fits \
    -q "${footprint}"' AND samples/zCOSMOS & 1'
echo
