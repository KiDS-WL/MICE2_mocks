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
    ${PIPEDIR}/KV_chunk.sh \
        $1 \
        $(basename $MOCKoutfull) \
        $(basename $MOCKout) > ${logfile} 2>&1
    echo "processed: $1"
}
export -f process_chunk


# data paths
export PIPEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATADIR=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/KiDS_VIKING
mkdir -p ${DATADIR}
CHUNKDIR=${DATADIR}/CHUNKS

# static file names
MOCKraw=/net/home/${HOSTNAME}/jlvdb/DATA/MICE2_KV_full/MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL.fits
export MOCKoutfull=${DATADIR}/MICE2_all.fits
export MOCKout=${DATADIR}/MICE2_KV450.fits
# KV450 data table for check plots
dataKV450=/net/home/${HOSTNAME}/jlvdb/DATA/KV450/KiDS_VIKING/KV450_north.cat

# constant parameters
RAname=ra_gal_mag
DECname=dec_gal_mag
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5  # the original value is 1.0, however a slightly larger values
            # yields smaller photometric uncertainties and a better match in
            # the spec-z vs phot-z distribution between data and mocks

export BPZPATH=~/src/bpz-1.99.3

echo "==> split MICE2 footprint into chunks"
test -e ${DATADIR}/footprint.txt && rm ${DATADIR}/footprint.txt
# Create data chunks that can be processed in parallel.
test -e ${CHUNKDIR} && rm -r ${CHUNKDIR}
mkdir -p ${CHUNKDIR}
data_table_split_rows \
    -i ${MOCKraw} \
    --n-splits 960 \
    -o ${CHUNKDIR}/KV_full.fits \

for fits in ${CHUNKDIR}/KV_full_*.fits; do
    subdir=${fits::-5}
    mkdir ${subdir}
    mv $fits ${subdir}/
done
echo ""

echo "==> process the mock chucks"
# Process each chunk sequentially and collect the output logs for debugging.
export logdir=${CHUNKDIR}/logs
mkdir -p $logdir
echo "writing logs to: ${logdir}"
find ${CHUNKDIR} -name "KV_full*.fits" -type f | \
    xargs -n 1 -P ${THREADS} bash -c 'process_chunk "$@"' --
echo ""

# This is what will be limited by memory.
echo "==> merge the mock chunks"
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename $MOCKoutfull) \
    -o ${MOCKoutfull}
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename $MOCKout) \
    -o ${MOCKout}
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename ${MOCKoutfull%.*}_CC.fits) \
    -o ${MOCKoutfull%.*}_CC.fits
data_table_vstack \
    -i ${CHUNKDIR}/*/$(basename ${MOCKout%.*}_CC.fits) \
    -o ${MOCKout%.*}_CC.fits
echo ""

###############################################################################
# create check plots
###############################################################################

echo "==> plot aperture statistics"
plot_extended_object_sn \
    -i ${MOCKout} \
    --filters \
        sdss_u \
        sdss_g \
        sdss_r \
        sdss_i \
        sdss_z \
        des_asahi_full_y \
        vhs_j \
        vhs_h \
        vhs_ks
echo ""

echo "==> plot magnitude statistics"
plot_photometry_realisation \
    -s ${MOCKout} \
    --s-filters \
        sdss_u_obs_mag \
        sdss_g_obs_mag \
        sdss_r_obs_mag \
        sdss_i_obs_mag \
        sdss_z_obs_mag \
        des_asahi_full_y_obs_mag \
        vhs_j_obs_mag \
        vhs_h_obs_mag \
        vhs_ks_obs_mag \
    --s-errors \
        sdss_u_obserr_mag \
        sdss_g_obserr_mag \
        sdss_r_obserr_mag \
        sdss_i_obserr_mag \
        sdss_z_obserr_mag \
        des_asahi_full_y_obserr_mag \
        vhs_j_obserr_mag \
        vhs_h_obserr_mag \
        vhs_ks_obserr_mag \
    -d ${dataKV450} \
    --d-filters \
        MAG_GAAP_u \
        MAG_GAAP_g \
        MAG_GAAP_r \
        MAG_GAAP_i \
        MAG_GAAP_Z \
        MAG_GAAP_Y \
        MAG_GAAP_J \
        MAG_GAAP_H \
        MAG_GAAP_Ks \
    --d-errors \
        MAGERR_GAAP_u \
        MAGERR_GAAP_g \
        MAGERR_GAAP_r \
        MAGERR_GAAP_i \
        MAGERR_GAAP_Z \
        MAGERR_GAAP_Y \
        MAGERR_GAAP_J \
        MAGERR_GAAP_H \
        MAGERR_GAAP_Ks \
    --d-extinct \
        EXTINCTION_u \
        EXTINCTION_g \
        EXTINCTION_r \
        EXTINCTION_i \
        EXTINCTION_Z \
        EXTINCTION_Y \
        EXTINCTION_J \
        EXTINCTION_H \
        EXTINCTION_Ks
echo ""

echo "==> plot weight statistics"
plot_draw_property \
    -s ${MOCKout} \
    --s-prop recal_weight \
    --s-filters \
        sdss_u_obs_mag \
        sdss_g_obs_mag \
        sdss_r_obs_mag \
        sdss_i_obs_mag \
        sdss_z_obs_mag \
        des_asahi_full_y_obs_mag \
        vhs_j_obs_mag \
        vhs_h_obs_mag \
        vhs_ks_obs_mag \
    -d ${dataKV450} \
    --d-prop recal_weight \
    --d-filters \
        MAG_GAAP_u \
        MAG_GAAP_g \
        MAG_GAAP_r \
        MAG_GAAP_i \
        MAG_GAAP_Z \
        MAG_GAAP_Y \
        MAG_GAAP_J \
        MAG_GAAP_H \
        MAG_GAAP_Ks
echo ""

echo "==> plot photo-z statistics"
plot_bpz_wrapper \
    -s ${MOCKout} \
    --s-z-true z_cgal_v \
    -d ${dataKV450} \
    --d-zb Z_B
echo ""
