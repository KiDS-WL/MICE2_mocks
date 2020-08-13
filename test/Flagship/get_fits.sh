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
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${table}.fits \
    -q "$footprint"
echo

# topcat /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_${table}.fits /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_reference.fits

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_K1000.fits \
    -q "${footprint}"' AND samples/KiDS & 1'
echo

../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_2dFLenS.fits \
    -q "${footprint}"' AND samples/2dFLenS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_GAMA.fits \
    -q "${footprint}"' AND samples/GAMA & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_SDSS.fits \
    -q "${footprint}"' AND samples/SDSS & 1'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_BOSS.fits \
    -q "${footprint}"' AND samples/SDSS & 12'
echo
../../scripts/mocks_datastore_query \
    /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
    -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_WiggleZ.fits \
    -q "${footprint}"' AND samples/WiggleZ & 1'
echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_DEEP2.fits \
#     -q "${footprint}"' AND samples/DEEP2 & 1'
# echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_VVDSf02.fits \
#     -q "${footprint}"' AND samples/VVDSf02 & 1'
# echo
# ../../scripts/mocks_datastore_query \
#     /net/home/fohlen13/jlvdb/DATA/Flagship_KiDS${suffix} \
#     -o /net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/Flagship_query${suffix}_zCOSMOS.fits \
#     -q "${footprint}"' AND samples/zCOSMOS & 1'
# echo
