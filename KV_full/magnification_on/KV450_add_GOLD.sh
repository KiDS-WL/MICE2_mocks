#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for KV450 derived from the MICE2 galaxy mock      #
#   catalogue. The catalogue contains a KiDS like photometry, lensfit         #
#   weights and BPZ photo-z.                                                  #
#                                                                             #
#   ARG1: Number of threads to use for parallel processing of data chunks     #
#                                                                             #
###############################################################################

THREADS=$(echo ${1:-$(nproc)})


function process_chunk(){
    logfile=${logdir}/$(basename $1 .fits).log
    ${PIPEDIR}/KV_add_GOLD.sh \
        $1 \
        $(basename $MOCKoutfull) \
        $(basename $MOCKout) > ${logfile} 2>&1
    echo "processed: $1"
}
export -f process_chunk


# data paths
export PIPEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATADIR=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/KiDS_VIKING_magnification_on
mkdir -p ${DATADIR}
CHUNKDIR=${DATADIR}/CHUNKS

# static file names
MOCKraw=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/MICE2_all_uBgVrRciIcYJHKs_shapes_halos_WL.fits
export MOCKoutfull=${DATADIR}/MICE2_all.fits
export MOCKout=${DATADIR}/MICE2_KV450.fits

echo "==> add GOLD flags"
# Process each chunk sequentially and collect the output logs for debugging.
export logdir=${CHUNKDIR}/logs_GOLD
mkdir -p $logdir
echo "writing logs to: ${logdir}"
find ${CHUNKDIR} -name "KV_full*.fits" -type f | \
    xargs -n 1 -P ${THREADS} bash -c 'process_chunk "$@"' --
echo ""

echo "==> merge the mock chunks"
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename $MOCKoutfull) \
    -o ${MOCKoutfull}
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename $MOCKout) \
    -o ${MOCKout}
echo ""
