#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for VVDSf02 derived from the MICE2 galaxy mock    #
#   catalogue.                                                                #
#                                                                             #
###############################################################################

# data paths
DATADIR=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/KiDS_VIKING_magnification_off
mkdir -p ${DATADIR}
CHUNKDIR=${DATADIR}/CHUNKS

# static file names
export MOCKout=${DATADIR}/MICE2_VVDSf02.fits

echo "==> process the mock chucks"
# Process each chunk sequentially and collect the output logs for debugging.
n_chunks=$(ls ${CHUNKDIR}/*/MICE2_all.fits | wc -l)
for file in ${CHUNKDIR}/*/MICE2_all.fits; do
    mocks_MICE_specz_sample \
        -s $file --s-type KV450 \
        --survey VVDSf02 \
        --n-data $(python -c "print(int(4194 / 0.462910 * 5156.625 / ${n_chunks}))") \
        -o $(dirname $file)/$(basename ${MOCKout})
done
echo ""

echo "==> merge the mock chunks"
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename ${MOCKout}) \
    -o ${MOCKout}
echo ""
