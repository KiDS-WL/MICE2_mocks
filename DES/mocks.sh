#!/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create a mock catalogue for DESy1 derived from the MICE2 galaxy mock      #
#   catalogue with enabled magnification. The catalogue contains a DES like   #
#   photometry, DES selection flags, lensift weights and BPZ photo-z.         #
#                                                                             #
###############################################################################

# data paths
DATADIR=${HOME}/DATA/MICE2_DES/DES_sigma_12
mkdir -p ${DATADIR}

# static file names
MOCKraw=${HOME}/DATA/MICE2_DES/MICE2_deep_BgVrRciIcY_shapes_halos_WL.fits
MOCKmasked=${DATADIR}/MICE2_deep_BgVrRciIcY_shapes_halos_WL_masked.fits
MOCKoutfull=${DATADIR}/MICE2_all.fits
MOCKout=${DATADIR}/MICE2_DES.fits
# DES data table for check plots
dataDES=${HOME}/DATA/DES/mcal_photo_100th_mags.fits

# constant parameters
BOUNDS="35 55 6 24"  # footprint that is used for mocks: RAmin/max DECmin/max
RAname=ra_gal_mag
DECname=dec_gal_mag
PSFs="    1.25  1.07  0.97  0.89  1.07"  # from Drlica-Wagner+17/18
MAGlims="23.4  23.2  22.5  21.8  20.1"   # from Drlica-Wagner+17/18
MAGsig=12.0  # the original value is 10.0, however a slightly larger values
             # yields smaller photometric uncertainties and a better match in
             # the spec-z vs phot-z distribution between data and mocks

export BPZPATH=~/src/bpz-1.99.3

echo "==> generate base footprint for DES"
test -e ${DATADIR}/footprint.txt && rm ${DATADIR}/footprint.txt
# Create bounds of ~343 sqdeg (effective KV450 area). This does not match the
# true DES footprint but is sufficeint for the use case. Create a pointing list
# of 120 pointings (10x12). Only of interest for a potential DES mock
# cross-correlation analysis.
mocks_generate_footprint \
    -b $BOUNDS \
    --survey DES \
    -f ${DATADIR}/footprint.txt \
    -p ${DATADIR}/pointings_DES.txt \
    --grid 10 12
echo ""

echo "==> mask MICE2 to DES footprint"
# apply the bounds to the MICE2 catalogue
data_table_mask_ra_dec \
    -i ${MOCKraw} \
    -b $BOUNDS \
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

echo "==> apply flux magnification"
# based on mock convergence field, apply to all KiDS bands and Johnson filters
# for deep field spec-z mocks
mocks_flux_magnification \
    -i ${MOCKoutfull} \
    --filters \
        lephare_b_evo \
        des_asahi_full_g_evo \
        lephare_v_evo \
        des_asahi_full_r_evo \
        lephare_rc_evo \
        des_asahi_full_i_evo \
        lephare_ic_evo \
        des_asahi_full_z_evo \
        des_asahi_full_y_evo \
    --convergence kappa \
    -o ${DATADIR}/magnitudes_magnified.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/magnitudes_magnified.fits \
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
        des_asahi_full_g \
        des_asahi_full_r \
        des_asahi_full_i \
        des_asahi_full_z \
        des_asahi_full_y \
    --scale 2.5 --flux-frac 0.5 \
    -o ${DATADIR}/apertures.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/magnitudes_magnified.fits \
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
        des_asahi_full_g_evo_mag \
        des_asahi_full_r_evo_mag \
        des_asahi_full_i_evo_mag \
        des_asahi_full_z_evo_mag \
        des_asahi_full_y_evo_mag \
    --limits $MAGlims \
    --significance $MAGsig \
    --sn-factors \
        sn_factor_des_asahi_full_g \
        sn_factor_des_asahi_full_r \
        sn_factor_des_asahi_full_i \
        sn_factor_des_asahi_full_z \
        sn_factor_des_asahi_full_y \
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

echo "==> assign galaxy weights"
# Assign lensfit weights by matching mock galaxies in 4-band magnitude space to
# their nearest neighbour DES galaxies. Additionally clone flags_select which
# is essential to obtain the correct object selection with flags_select==0.
# Mock galaxies that do not have a nearest neighbour within --r-max (Minkowski
# distance) are assigned the --fallback values.
mocks_draw_property \
    -s ${MOCKoutfull} \
    --s-attr \
        des_asahi_full_g_obs_mag \
        des_asahi_full_r_obs_mag \
        des_asahi_full_i_obs_mag \
        des_asahi_full_z_obs_mag \
    --s-prop flags_select LF_weight \
    -d ${HOME}/DATA/DES/flags_select.fits \
    --d-attr \
        MAG_g \
        MAG_r \
        MAG_i \
        MAG_z \
    --d-prop flags_select LF_weight \
    -t ${HOME}/DATA/DES/flags_select.tree.pickle \
    --r-max 1.0 \
    --fallback -99 0.0 \
    -o ${DATADIR}/weights.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/magnitudes_magnified.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/weights.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> compute photo-zs"
# Run BPZ on the mock galaxy photometry using the KV450 setup (prior: NGVS,
# templates: Capak CWWSB), assuming this is close enough to the DES photo-z.
# The prior limited to 0.7 < z < 1.43.
mocks_bpz_wrapper \
    -i ${MOCKoutfull} \
    --filters \
        des_asahi_full_g_obs_mag \
        des_asahi_full_r_obs_mag \
        des_asahi_full_i_obs_mag \
        des_asahi_full_z_obs_mag \
    --errors \
        des_asahi_full_g_obserr_mag \
        des_asahi_full_r_obserr_mag \
        des_asahi_full_i_obserr_mag \
        des_asahi_full_z_obserr_mag \
    --z-true z_cgal_v \
    --z-min 0.06674 \
    --z-max 1.42667 \
    --templates CWWSB_capak \
    --prior NGVS \
    --prior-filter des_asahi_full_i_obs_mag \
    -o ${DATADIR}/photoz.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/magnitudes_magnified.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/weights.fits \
       ${DATADIR}/photoz.fits \
    -o ${MOCKoutfull}
echo ""

echo "==> apply final DES selection"
# select objects with flags_select==0 and M_0<90
data_table_filter \
    -i ${MOCKoutfull} \
    --rule flags_select eq 0 \
    --rule M_0 ll 90.0 \
    -o ${MOCKout}
echo ""

echo "==> reduce number of columns"
# create copies of the output catalogues with a minimal subset of colums
data_table_copy \
    -i ${MOCKoutfull} \
    -c unique_gal_id $RAname $DECname \
       z_cgal z_cgal_v Z_B \
       flags_select LF_weight \
       flag_central lmhalo lmstellar \
       des_asahi_full_i_evo_mag \
       lephare_b_evo_mag \
       lephare_v_evo_mag \
       lephare_rc_evo_mag \
       lephare_ic_evo_mag \
       des_asahi_full_g_obs_mag \
       des_asahi_full_r_obs_mag \
       des_asahi_full_i_obs_mag \
       des_asahi_full_z_obs_mag \
       des_asahi_full_y_obs_mag \
    -o ${MOCKoutfull%.*}_CC.fits
data_table_copy \
    -i ${MOCKout} \
    -c unique_gal_id $RAname $DECname \
       z_cgal z_cgal_v Z_B \
       flags_select LF_weight \
       flag_central lmhalo lmstellar \
       des_asahi_full_i_evo_mag \
       lephare_b_evo_mag \
       lephare_v_evo_mag \
       lephare_rc_evo_mag \
       lephare_ic_evo_mag \
       des_asahi_full_g_obs_mag \
       des_asahi_full_r_obs_mag \
       des_asahi_full_i_obs_mag \
       des_asahi_full_z_obs_mag \
       des_asahi_full_y_obs_mag \
    -o ${MOCKout%.*}_CC.fits
echo ""

###############################################################################
# create check plots
###############################################################################

echo "==> plot aperture statistics"
plot_extended_object_sn \
    -i ${MOCKout} \
    --filters \
        des_asahi_full_g \
        des_asahi_full_r \
        des_asahi_full_i \
        des_asahi_full_z \
        des_asahi_full_y
echo ""

echo "==> plot magnitude statistics"
plot_photometry_realisation \
    -s ${MOCKout} \
    --s-filters \
        des_asahi_full_g_obs_mag \
        des_asahi_full_r_obs_mag \
        des_asahi_full_i_obs_mag \
        des_asahi_full_z_obs_mag \
    --s-errors \
        des_asahi_full_g_obserr_mag \
        des_asahi_full_r_obserr_mag \
        des_asahi_full_i_obserr_mag \
        des_asahi_full_z_obserr_mag \
    -d ${dataDES} \
    --d-filters \
        MAG_g \
        MAG_r \
        MAG_i \
        MAG_z \
    --d-errors \
        MAGERR_g \
        MAGERR_r \
        MAGERR_i \
        MAGERR_z
echo ""

echo "==> plot weight statistics"
plot_draw_property \
    -s ${MOCKout} \
    --s-prop flags_select \
    --s-filters \
        des_asahi_full_g_obs_mag \
        des_asahi_full_r_obs_mag \
        des_asahi_full_i_obs_mag \
        des_asahi_full_z_obs_mag \
    -d ${dataDES} \
    --d-prop flags_select \
    --d-filters \
        MAG_g MAG_r MAG_i MAG_z
plot_draw_property \
    -s ${MOCKout} \
    --s-prop LF_weight \
    --s-filters \
        des_asahi_full_g_obs_mag \
        des_asahi_full_r_obs_mag \
        des_asahi_full_i_obs_mag \
        des_asahi_full_z_obs_mag \
    -d ${dataDES} \
    --d-prop LF_weight \
    --d-filters \
        MAG_g MAG_r MAG_i MAG_z
echo ""

echo "==> plot photo-z statistics"
plot_bpz_wrapper \
    -s ${MOCKout} \
    --s-z-true z_cgal_v \
    -d ${dataDES} \
    --d-zb mode_z
echo ""
