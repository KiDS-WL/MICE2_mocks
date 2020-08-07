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

for sample in KiDS 2dFLenS GAMA SDSS DEEP2 VVDSf02 zCOSMOS
do
    ../../scripts/mocks_select_sample \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
        --sample $sample --area 5156.6 \
        -c samples/${sample}.toml
done
