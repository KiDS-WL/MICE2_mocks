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

export hostname=$HOSTNAME
../../scripts/mocks_BPZ \
    /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
    -c BPZ.toml \
    --mag mags/KV450 \
    --zphot BPZ/KV450 \
    --threads ${2:-32}
