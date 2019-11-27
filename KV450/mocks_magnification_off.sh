#!/usr/bin/env bash

DATADIR=${HOME}/DATA/MICE2_KV450/KV450_magnification_off
mkdir -p ${DATADIR}

MAPDIR=${HOME}/CC/STOMP_MAPS/MICE2_KV450
mkdir -p ${MAPDIR}

MOCKraw=${HOME}/DATA/MICE2_KV450/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.fits
MOCKmasked=${DATADIR}/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL_masked.fits
MOCKoutfull=${DATADIR}/MICE2_all.fits
MOCKout=${DATADIR}/MICE2_KV450.fits

dataKV450=${HOME}/DATA/KV450/KiDS_VIKING/KV450_north.cat

RAname=ra_gal
DECname=dec_gal
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5

export BPZPATH=~/src/bpz-1.99.3

echo "==> generate base masks for KV450 footprint"
test -e ${DATADIR}/footprint.txt && rm ${DATADIR}/footprint.txt
mocks_generate_footprint \
    -b  0.00 30.00 30.00 60.00 \
    --survey MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL \
    --footprint-file ${DATADIR}/footprint.txt -a \
    -o ${MAPDIR}/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL.map
mocks_generate_footprint \
    -b  6.00 24.00 35.00 55.00 \
    --survey KV450 \
    --footprint-file ${DATADIR}/footprint.txt -a \
    -o ${MAPDIR}/MICE2_KV450.map \
    --pointings-file ${DATADIR}/pointings_KV450.txt \
    --n-pointings 440 --pointings-ra 20
echo ""

echo "==> mask MICE2 to KV450 footprint"
data_table_mask \
    -i ${MOCKraw} \
    -s ${MAPDIR}/MICE2_KV450_r16384.map \
    --ra $RAname --dec $DECname \
    -o ${MOCKmasked}
echo ""

echo "==> apply evolution correction"
mocks_MICE_mag_evolved \
    -i ${MOCKmasked} \
    -o ${DATADIR}/magnitudes_evolved.fits
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> compute point source S/N correction"
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
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> generate photometry realisation"
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
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> assign galaxy weights"
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
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/recal_weights.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> compute photo-zs"
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
data_table_filter \
    -i ${MOCKoutfull} \
    --rule recal_weight gg 0.0 \
    --rule M_0 ll 90.0 \
    -o ${MOCKout}
echo ""

echo "==> reduce number of columns"
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
