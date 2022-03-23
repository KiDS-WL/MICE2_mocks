#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for KiDS-VIKING data for a single chunk of        #
#   MICE2 data.                                                               #
#                                                                             #
#   ARG1: Data chunk fits file                                                #
#   ARG2: File name for table with all MICE2 objects                          #
#   ARG3: File name for table with KiDS-VIKING selected objects               #
#                                                                             #
###############################################################################

# data paths
DATADIR=$(dirname $1)
THREADS=1

# static file names
MOCKmasked=$1
MOCKoutfull=${DATADIR}/$2
MOCKout=${DATADIR}/$3

# constant parameters
RAname=ra_gal
DECname=dec_gal
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5  # the original value is 1.0, however a slightly larger values
            # yields smaller photometric uncertainties and a better match in
            # the spec-z vs phot-z distribution between data and mocks

export BPZPATH=${HOME}/src/bpz-1.99.3
export BPZPYTHON=${HOME}/BPZenv/bin/python2

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
        sdss_z des_asahi_full_y \
        vhs_j \
        vhs_h \
        vhs_ks \
    --scale 2.5 --flux-frac 0.5 \
    --threads ${THREADS} \
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
        vhs_h_obs \
        vhs_ks_obs \
    --s-prop recal_weight \
    -d /net/home/fohlen11/jlvdb/DATA/KV450/recal_weights.fits \
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
    -t /net/home/fohlen11/jlvdb/DATA/KV450/recal_weights.tree.pickle \
    --threads ${THREADS} \
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
    --z-min 0.06674 \
    --z-max 1.42667 \
    --templates CWWSB_capak \
    --prior NGVS \
    --prior-filter sdss_i_obs \
    --threads ${THREADS} \
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
