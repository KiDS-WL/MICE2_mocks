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

export hostname=$HOSTNAME
../../scripts/mocks_BPZ $DS \
    -c config/BPZ.toml \
    --mag mags/KV450 \
    --zphot BPZ/KV450 \
    --threads ${2:-32}
echo
