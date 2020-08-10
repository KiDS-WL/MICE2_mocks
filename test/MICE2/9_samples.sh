#!/usr/bin/env bash

if [ "$1" == "deep" ]
then
    table="deep"
    area=859.44
elif [ "$1" == "test" ]
then
    table="256th"
    area=5156.6
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"deep\""
    exit 1;
fi

for sample in KiDS 2dFLenS GAMA SDSS WiggleZ DEEP2 VVDSf02 zCOSMOS
do
    ../../scripts/mocks_select_sample \
        /net/home/fohlen12/jlvdb/DATA/MICE2_test_memmap_${table} \
        --sample $sample --area $area \
        -c samples/${sample}.toml \
        --threads ${2:-32}
    echo
done
