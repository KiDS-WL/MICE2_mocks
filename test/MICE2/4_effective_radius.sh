#!/usr/bin/env bash

if [ "$1" == "all" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_all
elif [ "$1" == "deep" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_crop
elif [ "$1" == "test" ]
then
    DS=/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse
else
    echo "ERROR: invalid setup \"$1\", must be \"test\", \"deep\" or \"all\""
    exit 1;
fi

../../scripts/mocks_effective_radius $DS \
    -c config/photometry.toml \
    --threads ${2:-32}
echo
