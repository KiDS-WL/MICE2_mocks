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

# For each survey create a footprint that covers 100x the masked CC data
# footprint. This is slightly less area then the DIR spec-z catalogues cover:
# DEEP2: 0.82 sqdeg, VVDSf02: 0.51 sqdeg, zCOSMOS: 1.73 sqdeg
echo "==> generate footprints for 100 dithers"
test -e ${OUTROOT}/dither_footprint.txt && rm ${OUTROOT}/dither_footprint.txt
mocks_generate_footprint \
    -b 35.0 40.0 6.0 22.9 \
    --survey DEEP2_dithers \
    -f ${OUTROOT}/dither_footprint.txt \
    -p ${OUTROOT}/DEEP2_dithers.txt \
    --grid 5 20
mocks_generate_footprint \
    -b 40.0 45.0 6.0 16.5 \
    --survey VVDSf02_dithers \
    -f ${OUTROOT}/dither_footprint.txt \
    -p ${OUTROOT}/VVDSf02_dithers.txt \
    --grid 5 20
mocks_generate_footprint \
    -b 45.0 55.0 6.0 24.0 \
    --survey zCOSMOS_dithers \
    -f ${OUTROOT}/dither_footprint.txt \
    -p ${OUTROOT}/zCOSMOS_dithers.txt \
    --grid 5 20
echo

# Split the data catalogues according to the pointing boundaries. The relevant
# file is MICE2_all_CC.fits on which the spec-z selection function will be
# applied.
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> split ${survey} into dithers"
    data_table_to_pointings \
        -i ${DATADIR}/MICE2_all_CC.fits \
        -p ${OUTROOT}/${survey}_dithers.txt \
        --ra $RAname --dec $DECname \
        -o ${OUTROOT}/${survey}_dithers
    echo
done

# Apply the spec-z survey selection functions. Read the number of objects in
# the DIR spec-z catalogues to sample down the mocks accordingly.
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> apply ${survey} selection function"
    n_obj=$(data_table_shape ${HOME}/DATA/DES/mcal_specz_${survey}_mag.fits)
    statfile=${OUTROOT}/${survey}_dithers/${survey}_selection.stats
    test -e $statfile && rm -v $statfile
    for fits in ${OUTROOT}/${survey}_dithers/${survey}*.fits; do
        # overwriting the input file since we don't need it anymore
        mocks_MICE_specz_sample \
            -s $fits --s-type DES \
            --n-data ${n_obj} --survey ${survey} \
            --pass-phot-detection --stats-file a \
            -o $fits
    done
    echo
done
