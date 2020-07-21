#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
else
    table="256th"
fi

../../scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -o ~/TEST/MICE2_query_${table}.fits \
    -q "position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0"

# topcat ~/TEST/MICE2_query_${table}.fits ~/TEST/MICE2_query_reference.fits
