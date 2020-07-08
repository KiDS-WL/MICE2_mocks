#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for KV450 derived from the MICE2 galaxy mock      #
#   catalogue. The catalogue contains a KiDS like photometry, lensfit         #
#   weights and BPZ photo-z.                                                  #
#                                                                             #
#   This version of the catalogue applies magnification to the MICE2 model    #
#   magnitudes based on the convergence.                                      #
#                                                                             #
###############################################################################

# data paths
DATADIR=${HOME}/TEST/MICE2_pipeline/$1
mkdir -p ${DATADIR}

# static file names
MOCKmasked=${DATADIR}/MICE2_masked.fits
MOCKoutfull=${DATADIR}/MICE2_all.fits
MOCKout=${DATADIR}/MICE2_KV450.fits
# KV450 data table for check plots
dataKV450=${HOME}/TEST/MICE2_pipeline/KV450_sample.fits

# constant parameters
RAname=ra_gal_mag
DECname=dec_gal_mag
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.0  # the original value is 1.0, however a slightly larger values
            # yields smaller photometric uncertainties and a better match in
            # the spec-z vs phot-z distribution between data and mocks

export BPZPATH=~/src/bpz-1.99.3

function evo() {
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
}

function mag() {
    echo "==> apply flux magnification"
    # based on mock convergence field, apply to all KiDS bands and Johnson filters
    # for deep field spec-z mocks
    mocks_flux_magnification \
        -i ${MOCKoutfull} \
        --filters \
            sdss_u_evo \
            lephare_b_evo \
            sdss_g_evo \
            lephare_v_evo \
            sdss_r_evo \
            lephare_rc_evo \
            sdss_i_evo \
            lephare_ic_evo \
            sdss_z_evo \
            des_asahi_full_y_evo \
            vhs_j_evo \
            vhs_h_evo \
            vhs_ks_evo \
        --convergence kappa \
        -o ${DATADIR}/magnitudes_magnified.fits
    # update the combined data table
    data_table_hstack \
        -i ${MOCKmasked} \
        ${DATADIR}/magnitudes_evolved.fits \
        ${DATADIR}/magnitudes_magnified.fits \
        -o ${MOCKoutfull}
    echo ""
}

function aper() {
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
            sdss_z des_asahi_full_y \
            vhs_j \
            vhs_h \
            vhs_ks \
        --flux-frac 0.5 \
        --aper-min 0.7 --aper-max 2.0 \
        --threads $(($(nproc) / 2)) \
        -o ${DATADIR}/apertures.fits
    # update the combined data table
    data_table_hstack \
        -i ${MOCKmasked} \
        ${DATADIR}/magnitudes_evolved.fits \
        ${DATADIR}/magnitudes_magnified.fits \
        ${DATADIR}/apertures.fits \
        -o ${MOCKoutfull}
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
}

function phot() {
    echo "==> generate photometry realisation"
    Based on the KiDS limiting magnitudes, calcalute the mock galaxy S/N and
    apply the aperture size S/N correction to obtain a KiDS-like magnitude
    realisation.
    mocks_photometry_realisation \
        -i ${MOCKoutfull} \
        --filters \
            sdss_u_evo_mag \
            sdss_g_evo_mag \
            sdss_r_evo_mag \
            sdss_i_evo_mag \
            sdss_z_evo_mag \
            des_asahi_full_y_evo_mag \
            vhs_j_evo_mag \
            vhs_h_evo_mag \
            vhs_ks_evo_mag \
        --limits $MAGlims \
        --aperture 2.0 \
        --aperture-areas \
            area_aper_sdss_u \
            area_aper_sdss_g \
            area_aper_sdss_r \
            area_aper_sdss_i \
            area_aper_sdss_z \
            area_aper_des_asahi_full_y \
            area_aper_vhs_j \
            area_aper_vhs_h \
            area_aper_vhs_ks \
        --significance $MAGsig \
        -o ${DATADIR}/magnitudes_observed.fits
    # update the combined data table
    data_table_hstack \
        -i ${MOCKmasked} \
        ${DATADIR}/magnitudes_evolved.fits \
        ${DATADIR}/magnitudes_magnified.fits \
        ${DATADIR}/apertures.fits \
        ${DATADIR}/magnitudes_observed.fits \
        -o ${MOCKoutfull}
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
}

function weights() {
    echo "==> assign galaxy weights"
    # Assign lensfit weights by matching mock galaxies in 9-band magnitude space to
    # their nearest neighbour KV450 galaxies using the super-user catalogues which
    # contain objects with recal_weight<=0. Mock galaxies that do not have a
    # nearest neighbour within --r-max (Minkowski distance) are assigned the
    # --fallback values.

    mocks_draw_property \
        -s ${MOCKoutfull} \
        --s-attr \
            sdss_u_obs_mag \
            sdss_g_obs_mag \
            sdss_r_obs_mag \
            sdss_i_obs_mag \
            sdss_z_obs_mag \
            des_asahi_full_y_obs_mag \
            vhs_j_obs_mag \
            vhs_h_obs_mag \
            vhs_ks_obs_mag \
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
        --threads $(($(nproc) / 2)) \
        -o ${DATADIR}/recal_weights.fits
    # update the combined data table
    data_table_hstack \
        -i ${MOCKmasked} \
        ${DATADIR}/magnitudes_evolved.fits \
        ${DATADIR}/magnitudes_magnified.fits \
        ${DATADIR}/apertures.fits \
        ${DATADIR}/magnitudes_observed.fits \
        ${DATADIR}/recal_weights.fits \
        -o ${MOCKoutfull}
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
}

function bpz(){
    echo "==> compute photo-zs"
    # Run BPZ on the mock galaxy photometry using the KV450 setup (prior: NGVS,
    # templates: Capak CWWSB), the prior limited to 0.7 < z < 1.43.
    mocks_bpz_wrapper \
        -i ${MOCKoutfull} \
        --filters \
            sdss_u_obs_mag \
            sdss_g_obs_mag \
            sdss_r_obs_mag \
            sdss_i_obs_mag \
            sdss_z_obs_mag \
            des_asahi_full_y_obs_mag \
            vhs_j_obs_mag \
            vhs_h_obs_mag \
            vhs_ks_obs_mag \
        --errors \
            sdss_u_obserr_mag \
            sdss_g_obserr_mag \
            sdss_r_obserr_mag \
            sdss_i_obserr_mag \
            sdss_z_obserr_mag \
            des_asahi_full_y_obserr_mag \
            vhs_j_obserr_mag \
            vhs_h_obserr_mag \
            vhs_ks_obserr_mag \
        --z-min 0.06674 \
        --z-max 1.42667 \
        --templates CWWSB_capak \
        --prior NGVS \
        --prior-filter sdss_i_obs_mag \
        --threads $(($(nproc) / 2)) \
        -o ${DATADIR}/photoz.fits
    # update the combined data table
    data_table_hstack \
        -i ${MOCKmasked} \
        ${DATADIR}/magnitudes_evolved.fits \
        ${DATADIR}/magnitudes_magnified.fits \
        ${DATADIR}/apertures.fits \
        ${DATADIR}/magnitudes_observed.fits \
        ${DATADIR}/recal_weights.fits \
        ${DATADIR}/photoz.fits \
        -o ${MOCKoutfull}
    echo ""

    echo "==> plot photo-z statistics"
    plot_bpz_wrapper \
        -s ${MOCKout} \
        --s-z-true z_cgal_v \
        -d ${dataKV450} \
        --d-zb Z_B
    echo ""
}

function selection() {
    echo "==> apply final KV450 selection"
    # select objects with recal_weight>0 and M_0<90
    data_table_filter \
        -i ${MOCKoutfull} \
        --rule recal_weight gg 0.0 \
        --rule M_0 ll 90.0 \
        -o ${MOCKout}
    echo ""
}

evo
mag
aper
phot
weights
bpz
selection
