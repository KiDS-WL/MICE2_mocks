#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for KV450 derived from the MICE2 galaxy mock      #
#   catalogue. The catalogue contains a KiDS like photometry, lensfit         #
#   weights and BPZ photo-z.                                                  #
#                                                                             #
#   This version of the catalogue has magnification disabled.                 #
#                                                                             #
###############################################################################

# data paths
DATADIR=${HOME}/DATA/MICE2_KV450/KV450_magnification_off
mkdir -p ${DATADIR}
MAPDIR=${HOME}/CC/STOMP_MAPS/MICE2_KV450
mkdir -p ${MAPDIR}

# static file names
MOCKraw=${HOME}/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits
MOCKmasked=${DATADIR}/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL_masked.fits
MOCKoutfull=${DATADIR}/MICE2_all.fits
MOCKout=${DATADIR}/MICE2_KV450.fits
# KV450 data table for check plots
dataKV450=${HOME}/DATA/KV450/KiDS_VIKING/KV450_north.cat

# constant parameters
RAname=ra_gal
DECname=dec_gal
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5

export BPZPATH=~/src/bpz-1.99.3

echo "==> generate base masks for KV450 footprint"
test -e ${DATADIR}/footprint.txt && rm ${DATADIR}/footprint.txt
# STOMP map that encompasses the data downloaded from COSMOHUB
mocks_generate_footprint \
    -b  0.00 30.00 30.00 60.00 \
    --survey MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL \
    --footprint-file ${DATADIR}/footprint.txt -a \
    -o ${MAPDIR}/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.map
# STOMP map that masks the data to ~343 sqdeg (effective KV450 area).
# Create a pointing list of 440 pointings (20x22) with ~0.7 sqdeg each (mean
# pointing area in KV450 CC data).
mocks_generate_footprint \
    -b  6.00 24.00 35.00 55.00 \
    --survey KV450 \
    --footprint-file ${DATADIR}/footprint.txt -a \
    -o ${MAPDIR}/MICE2_KV450.map \
    --pointings-file ${DATADIR}/pointings_KV450.txt \
    --n-pointings 440 --pointings-ra 20
echo ""

echo "==> mask MICE2 to KV450 footprint"
# apply the STOMP map to the MICE2 catalogue
data_table_mask \
    -i ${MOCKraw} \
    -s ${MAPDIR}/MICE2_KV450_r16384.map \
    --ra $RAname --dec $DECname \
    -o ${MOCKmasked}
echo ""

echo "==> apply evolution correction"
# automatically applied to any existing MICE2 filter column
mocks_MICE_mag_evolved \
    -i ${MOCKmasked} \
    -o ${DATADIR}/magnitudes_evolved.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> compute point source S/N correction"
# Compute the effective radius (that contains 50% of the luminosity), compute
# the observational size using the PSFs, scale this with a factor of 2.5
# (similar to what sextractor would do) to get a mock aperture. Finally
# calculate a correction factor for the S/N based on the aperture area compared
# to a point source (= PSF area).
mocks_extended_object_sn \
    -i ${MOCKoutfull} \
    --bulge-ratio bulge_fraction --bulge-size bulge_length \
    --disk-size disk_length --ba-ratio bulge_axis_ratio \
    --psf $PSFs \
    --filters \
        sdss_u \
        sdss_g \
        sdss_r \
        sdss_i \
        sdss_z \
        des_asahi_full_y \
        vhs_j \
        vhs_h \
        vhs_ks \
    --scale 2.5 --flux-frac 0.5 \
    -o ${DATADIR}/apertures.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> generate photometry realisation"
# Based on the KiDS limiting magnitudes, calcalute the mock galaxy S/N and
# apply the aperture size S/N correction to obtain a KiDS-like magnitude
# realisation.
mocks_photometry_realisation \
    -i ${MOCKoutfull} \
    --filters \
        sdss_u_evo \
        sdss_g_evo \
        sdss_r_evo \
        sdss_i_evo \
        sdss_z_evo \
        des_asahi_full_y_evo \
        vhs_j_evo \
        vhs_h_evo \
        vhs_ks_evo \
    --limits $MAGlims \
    --significance $MAGsig \
    --sn-factors \
        sn_factor_sdss_u \
        sn_factor_sdss_g \
        sn_factor_sdss_r \
        sn_factor_sdss_i \
        sn_factor_sdss_z \
        sn_factor_des_asahi_full_y \
        sn_factor_vhs_j \
        sn_factor_vhs_h \
        sn_factor_vhs_ks \
    -o ${DATADIR}/magnitudes_observed.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> assign galaxy weights"
# Assign lensfit weights by matching mock galaxies in 9-band magnitude space to
# their nearest neighbour KV450 galaxies using the super-user catalogues which
# contain objects with recal_weight<=0. Mock galaxies that do not have a
# nearest neighbour within --r-max (Minkowski distance) are assigned the
# --fallback values.
mocks_draw_property \
    -s ${MOCKoutfull} \
    --s-attr \
        sdss_u_obs \
        sdss_g_obs \
        sdss_r_obs \
        sdss_i_obs \
        sdss_z_obs \
        des_asahi_full_y_obs \
        vhs_j_obs \
        vhs_h_obs vhs_ks_obs \
    --s-prop recal_weight \
    -d ${HOME}/DATA/KV450/recal_weights.fits \
    --d-attr \
        MAG_GAAP_u \
        MAG_GAAP_g \
        MAG_GAAP_r \
        MAG_GAAP_i \
        MAG_GAAP_Z \
        MAG_GAAP_Y \
        MAG_GAAP_J \
        MAG_GAAP_H \
        MAG_GAAP_Ks \
    --d-prop weight \
    --r-max 1.0 \
    --fallback 0.0 \
    -t ${HOME}/DATA/KV450/recal_weights.tree.pickle \
    -o ${DATADIR}/recal_weights.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/recal_weights.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> compute photo-zs"
# Run BPZ on the mock galaxy photometry using the KV450 setup (prior: NGVS,
# templates: Capak CWWSB), the prior limited to 0.7 < z < 1.43.
mocks_bpz_wrapper \
    -i ${MOCKoutfull} \
    --filters \
        sdss_u_obs \
        sdss_g_obs \
        sdss_r_obs \
        sdss_i_obs \
        sdss_z_obs \
        des_asahi_full_y_obs \
        vhs_j_obs \
        vhs_h_obs \
        vhs_ks_obs \
    --errors \
        sdss_u_obserr \
        sdss_g_obserr \
        sdss_r_obserr \
        sdss_i_obserr \
        sdss_z_obserr \
        des_asahi_full_y_obserr \
        vhs_j_obserr \
        vhs_h_obserr \
        vhs_ks_obserr \
    --z-true z_cgal_v \
    --z-min 0.06674 \
    --z-max 1.42667 \
    --templates CWWSB_capak \
    --prior NGVS \
    --prior-filter sdss_i_obs \
    -o ${DATADIR}/photoz.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/recal_weights.fits \
       ${DATADIR}/photoz.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> apply final KV450 selection"
# select objects with recal_weight>0 and M_0<90
data_table_filter \
    -i ${MOCKoutfull} \
    --rule recal_weight gg 0.0 \
    --rule M_0 ll 90.0 \
    -o ${MOCKout}
echo ""

echo "==> reduce number of columns"
# create copies of the output catalogues with a minimal subset of colums
data_table_copy \
    -i ${MOCKoutfull} \
    -c unique_gal_id $RAname $DECname \
       z_cgal z_cgal_v Z_B \
       recal_weight flag_central \
       lmhalo lmstellar \
       sdss_i_evo \
       lephare_b_evo \
       lephare_v_evo \
       lephare_rc_evo \
       lephare_ic_evo \
       sdss_u_obs \
       sdss_g_obs \
       sdss_r_obs \
       sdss_i_obs \
       sdss_z_obs \
       des_asahi_full_y_obs \
       vhs_j_obs \
       vhs_h_obs \
       vhs_ks_obs \
    -o ${MOCKoutfull%.*}_CC.fits
data_table_copy \
    -i ${MOCKout} \
    -c unique_gal_id $RAname $DECname \
       z_cgal z_cgal_v Z_B \
       recal_weight flag_central \
       lmhalo lmstellar \
       sdss_i_evo \
       lephare_b_evo \
       lephare_v_evo \
       lephare_rc_evo \
       lephare_ic_evo \
       sdss_u_obs \
       sdss_g_obs \
       sdss_r_obs \
       sdss_i_obs \
       sdss_z_obs \
       des_asahi_full_y_obs \
       vhs_j_obs \
       vhs_h_obs \
       vhs_ks_obs \
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
        sdss_u_obs \
        sdss_g_obs \
        sdss_r_obs \
        sdss_i_obs \
        sdss_z_obs \
        des_asahi_full_y_obs \
        vhs_j_obs \
        vhs_h_obs \
        vhs_ks_obs \
    --s-errors \
        sdss_u_obserr \
        sdss_g_obserr \
        sdss_r_obserr \
        sdss_i_obserr \
        sdss_z_obserr \
        des_asahi_full_y_obserr \
        vhs_j_obserr \
        vhs_h_obserr \
        vhs_ks_obserr \
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
        sdss_u_obs \
        sdss_g_obs \
        sdss_r_obs \
        sdss_i_obs \
        sdss_z_obs \
        des_asahi_full_y_obs \
        vhs_j_obs \
        vhs_h_obs \
        vhs_ks_obs \
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
