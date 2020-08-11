#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test"
    area=24.4
elif [ "$1" == "all" ]
then
    suffix=""
    area=5156.6
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"all\""
    exit 1;
fi

for sample in KiDS #2dFLenS GAMA SDSS WiggleZ DEEP2 VVDSf02 zCOSMOS
do
    ../../scripts/mocks_select_sample \
        /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
        --sample $sample --area $area \
        -c samples/${sample}.toml \
        --threads ${2:-64}
    echo
done
