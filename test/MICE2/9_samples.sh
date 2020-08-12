#!/usr/bin/env bash

if [ "$1" == "all" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_all
    area=5156.6
elif [ "$1" == "deep" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_crop
    area=859.44
elif [ "$1" == "test" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse
    area=5156.6
else
    echo "ERROR: invalid setup \"$1\", must be \"test\", \"deep\" or \"all\""
    exit 1;
fi

for sample in KiDS 2dFLenS GAMA SDSS
do
    ../../scripts/mocks_select_sample $DS \
        --sample $sample --area $area \
        -c samples/${sample}.toml \
        --threads ${2:-32}
    echo
done

if [ "$1" == "deep" ]
then
    for sample in WiggleZ DEEP2 VVDSf02 zCOSMOS
    do
        ../../scripts/mocks_select_sample $DS \
            --sample $sample --area $area \
            -c samples/${sample}.toml \
            --threads ${2:-32}
        echo
    done
fi
