#/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create 100 line of sight realisations of the MICE2 DEEP2, VVDSf02 and     #
#   zCOSMOS spectroscopic mock catalogues to estimate the impact of sample    #
#   variance on the DIR redshift calibration.                                 #
#                                                                             #
###############################################################################

# data paths
DATADIR=${HOME}/DATA/MICE2_DES/DES_sigma_12
OUTROOT=${HOME}/DATA/MICE2_DES/SPECZ_sigma_12_dithers
mkdir -p ${OUTROOT}

# constant parameters
RAname=ra_gal_mag
DECname=dec_gal_mag

echo "==> generate footprints for 100 dithers"
# STOMP for each deep field that covers 100x the masked CC data footprint.
# This is slightly less area then the DIR spec-z catalogues cover:
# DEEP2: 0.82 sqdeg, VVDSf02: 0.51 sqdeg, zCOSMOS: 1.73 sqdeg
mocks_generate_footprint \
    -b 6 22.9 35 40 \
    --survey DEEP2_dithers \
    --footprint-file ${OUTROOT}/dither_footprint.txt -a \
    --n-pointings 100 --pointings-ra 5 \
    --pointings-file ${OUTROOT}/DEEP2_dithers.txt \
    -o ${OUTROOT}/DEEP2_dithers.map
mocks_generate_footprint \
    -b 6 16.5 40 45 \
    --survey VVDSf02_dithers \
    --footprint-file ${OUTROOT}/dither_footprint.txt -a \
    --n-pointings 100 --pointings-ra 5 \
    --pointings-file ${OUTROOT}/VVDSf02_dithers.txt \
    -o ${OUTROOT}/VVDSf02_dithers.map
mocks_generate_footprint \
    -b 6 24 45 55 \
    --survey zCOSMOS_dithers \
    --footprint-file ${OUTROOT}/dither_footprint.txt -a \
    --n-pointings 100 --pointings-ra 10 \
    --pointings-file ${OUTROOT}/zCOSMOS_dithers.txt \
    -o ${OUTROOT}/zCOSMOS_dithers.map
echo

# Split the STOMP masks according to their line of sight boundaries and then
# mask the data catalogues to the mask footprint. The relevant file is
# MICE2_all_CC.fits on which the spec-z selection function will be applied.
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> split ${survey} into dithers"
    # This script is from the stomp_tools package
    pointings_split_data \
        ${OUTROOT}/${survey}_dithers \
        --pointing-file ${OUTROOT}/${survey}_dithers.txt \
        --ref-file ${DATADIR}/MICE2_all_CC.fits \
        --test-file ${DATADIR}/MICE2_DES_CC.fits \
        --ra $RAname --dec $DECname \
        --map-file ${OUTROOT}/${survey}_dithers_r16384.map
    echo
done

# Simplify the output data structure and apply the spec-z survey selection
# functions. Read the number of objects in the DIR spec-z catalogues to sample
# down the mocks accordingly.
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> apply ${survey} selection function"
    n_obj=$(data_table_shape ${HOME}/DATA/DES/mcal_specz_${survey}_mag.fits)
    statfile=${OUTROOT}/${survey}_dithers/${survey}_selection.stats
    test -e $statfile && rm -v $statfile
    for fits in ${OUTROOT}/${survey}_dithers/${survey}_dither*/pointing_spec.cat; do
        mocks_MICE_specz_sample \
            -s $fits --s-type DES \
            --n-data ${n_obj} --survey ${survey} \
            --pass-phot-detection --stats-file a \
            -o ${OUTROOT}/${survey}_dithers/$(basename $(dirname $fits))_specz.fits
        mv $(dirname $fits)/pointing_phot.cat \
           ${OUTROOT}/${survey}_dithers/$(basename $(dirname $fits))_DES.fits
        rm -r $(dirname $fits)
    done
    echo
done
