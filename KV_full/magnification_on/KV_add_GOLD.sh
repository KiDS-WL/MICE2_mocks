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
RAname=ra_gal_mag
DECname=dec_gal_mag
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5  # the original value is 1.0, however a slightly larger values
            # yields smaller photometric uncertainties and a better match in
            # the spec-z vs phot-z distribution between data and mocks

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
    --s-prop Flag_SOM_Fid \
    -d /net/home/fohlen11/jlvdb/DATA/K1000/Flag_SOM_Fid_A.fits \
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
    --d-prop Flag_SOM_Fid_A \
    --r-max 1.0 \
    --fallback 0.0 \
    -t /net/home/fohlen11/jlvdb/DATA/KV450/Flag_SOM_Fid_A.tree.pickle \
    --threads ${THREADS} \
    -o ${DATADIR}/GOLD_flags.fits
# update the combined data table
data_table_hstack \
    -i ${MOCKmasked} \
       ${DATADIR}/magnitudes_evolved.fits \
       ${DATADIR}/magnitudes_magnified.fits \
       ${DATADIR}/apertures.fits \
       ${DATADIR}/magnitudes_observed.fits \
       ${DATADIR}/recal_weights.fits \
       ${DATADIR}/GOLD_flags.fits \
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
