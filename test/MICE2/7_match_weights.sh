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

../../scripts/mocks_match_data \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -c config/matching.toml \
    --threads ${2:-32}
