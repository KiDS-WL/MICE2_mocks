#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for WiggleZ derived from the MICE2 galaxy mock    #
#   catalogue.                                                                #
#                                                                             #
###############################################################################

# data paths
DATADIR=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/KiDS_VIKING_magnification_on
mkdir -p ${DATADIR}
CHUNKDIR=${DATADIR}/CHUNKS

# static file names
export MOCKout=${DATADIR}/MICE2_WiggleZ.fits

echo "==> process the mock chucks"
# Process each chunk sequentially and collect the output logs for debugging.
for file in ${CHUNKDIR}/*/MICE2_all.fits; do
    mocks_MICE_specz_sample \
        -s $file --s-type KV450 \
        --survey WiggleZ \
        -d ${HOME}/DATA/KV450/SPECZ/${survey}_masked.fits --d-z-spec z_spec \
        -o $(dirname $file)/$(basename ${MOCKout})
done
echo ""

echo "==> merge the mock chunks"
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename ${MOCKout}) \
    -o ${MOCKout}
echo ""
