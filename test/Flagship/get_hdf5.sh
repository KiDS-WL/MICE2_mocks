#!/usr/bin/env bash

if [ "$1" == "test" ]
then
    suffix="_test"
elif [ "$1" == "all" ]
then
    suffix="_all"
else
    echo "ERROR: invalid setup \"$1\", must be \"test\" or \"all\""
    exit 1;
fi

# this selects 24.4 deg^2
footprint='position/ra/obs >= 40.0 AND position/ra/obs < 45.0 AND position/dec/obs >= 10.0 AND position/dec/obs < 15.0'

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} --verify \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}.hdf5 \
    -q "$footprint"
echo

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_K1000.hdf5 \
    -q "${footprint}"' AND samples/KiDS & 1'
echo

exit 0;

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_2dFLenS.hdf5 \
    -q "${footprint}"' AND samples/2dFLenS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_GAMA.hdf5 \
    -q "${footprint}"' AND samples/GAMA & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_SDSS.hdf5 \
    -q "${footprint}"' AND samples/SDSS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_BOSS.hdf5 \
    -q "${footprint}"' AND samples/SDSS & 12'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_WiggleZ.hdf5 \
    -q "${footprint}"' AND samples/WiggleZ & 1'
echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_DEEP2.hdf5 \
#     -q "${footprint}"' AND samples/DEEP2 & 1'
# echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_VVDSf02.hdf5 \
#     -q "${footprint}"' AND samples/VVDSf02 & 1'
# echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_zCOSMOS.hdf5 \
#     -q "${footprint}"' AND samples/zCOSMOS & 1'
# echo
