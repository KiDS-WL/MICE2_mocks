#/usr/bin/env bash

###############################################################################
#                                                                             #
#   Create 100 line of sight realisations of the MICE2 DEEP2, VVDSf02 and     #
#   zCOSMOS spectroscopic mock catalogues to estimate the impact of sample    #
#   variance on the DIR redshift calibration. Additionally create 100         #
#   photometry realisations and photo-zs for Wright+20.                       #
#                                                                             #
#   This version of the catalogue has magnification enabled.                  #
#                                                                             #
###############################################################################

# data paths
DATADIR=${HOME}/DATA/MICE2_KV450/KV450_magnification_on
MOCKmasked=${DATADIR}/MICE2_deep_uBgVrRciIcYJHKs_shapes_halos_WL_masked.fits
OUTROOT=${HOME}/DATA/MICE2_KV450/REALISATIONS_magnification_on
mkdir -p ${OUTROOT}

# constant parameters
Nrealisations=100
RAname=ra_gal_mag
DECname=dec_gal_mag
PSFs="    1.0  0.9  0.7  0.8  1.0   1.0  0.9  1.0  0.9"
MAGlims="25.5 26.3 26.2 24.9 24.85 24.1 24.2 23.3 23.2"
MAGsig=1.5

export BPZPATH=~/src/bpz-1.99.3

# STOMP for each deep field that covers 100x the masked CC data footprint.
# This is slightly less area then the DIR spec-z catalogues cover:
# DEEP2: 0.82 sqdeg, VVDSf02: 0.51 sqdeg, zCOSMOS: 1.73 sqdeg
echo "==> generate footprints for 100 survey dithers"
mocks_generate_footprint \
    -b 35.0 40.0  6.0 22.9 \
    --survey DEEP2_dithers \
    --footprint-file ${OUTROOT}/dither_footprint.txt -a \
    --n-pointings 100 --pointings-ra 5 \
    --pointings-file ${OUTROOT}/DEEP2_dithers.txt \
    -o ${OUTROOT}/DEEP2_dithers.map
mocks_generate_footprint \
    -b 40.0 45.0  6.0 16.5 \
    --survey VVDSf02_dithers \
    --footprint-file ${OUTROOT}/dither_footprint.txt -a \
    --n-pointings 100 --pointings-ra 5 \
    --pointings-file ${OUTROOT}/VVDSf02_dithers.txt \
    -o ${OUTROOT}/VVDSf02_dithers.map
mocks_generate_footprint \
    -b 45.0 55.0  6.0 24.0 \
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
        --test-file ${DATADIR}/MICE2_KV450_CC.fits \
        --ra $RAname --dec $DECname \
        --map-file ${OUTROOT}/${survey}_dithers_r16384.map
    echo
done

# Pick the STOMP map of the 50th sight realisation (for each survey) as basis
# for creating photometry realisations.
echo "==> prepare the patches needed for the photometry realizations"
survey=DEEP2
cp -vf ${OUTROOT}/${survey}_dithers/${survey}_dithers_37p5_18p1/pointing_stomp.map \
       ${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base_r16384.map
survey=VVDSf02
cp -vf ${OUTROOT}/${survey}_dithers/${survey}_dithers_42p5_15p2/pointing_stomp.map \
       ${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base_r16384.map
survey=zCOSMOS
cp -vf ${OUTROOT}/${survey}_dithers/${survey}_dithers_49p5_8p6/pointing_stomp.map \
       ${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base_r16384.map
# Go back to the raw MICE2 input table and apply the mock pipeline steps
# needed to generate a photometry realisation (see ./mocks*.sh).
for survey in DEEP2 VVDSf02 zCOSMOS; do
    patchcat=${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base.fits
    echo "--> mask MICE2 to KV450 footprint"
    data_table_mask \
        -i ${MOCKmasked} \
        -s ${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base_r16384.map \
        --ra $RAname --dec $DECname \
        -o ${patchcat}
    echo "--> apply evolution correction"
    mocks_MICE_mag_evolved \
        -i ${patchcat} \
        -o ${OUTROOT}/temp.fits
    data_table_hstack \
        -i ${patchcat} ${OUTROOT}/temp.fits \
        -o ${patchcat}
    echo "--> apply flux magnification"
    mocks_flux_magnification \
        -i ${patchcat} \
        --filters \
            sdss_u_evo sdss_g_evo sdss_r_evo \
            sdss_i_evo sdss_z_evo des_asahi_full_y_evo \
            vhs_j_evo vhs_h_evo vhs_ks_evo \
        --convergence kappa \
        -o ${OUTROOT}/temp.fits
    data_table_hstack \
        -i ${patchcat} ${OUTROOT}/temp.fits \
        -o ${patchcat}
    echo "--> compute point source S/N correction"
    mocks_extended_object_sn \
        -i ${patchcat} \
        --bulge-ratio bulge_fraction --bulge-size bulge_length \
        --disk-size disk_length --ba-ratio bulge_axis_ratio \
        --psf $PSFs \
        --filters \
            sdss_u sdss_g sdss_r \
            sdss_i sdss_z des_asahi_full_y \
            vhs_j vhs_h vhs_ks \
        --scale 2.5 --flux-frac 0.5 \
        -o ${OUTROOT}/temp.fits
    data_table_hstack \
        -i ${patchcat} ${OUTROOT}/temp.fits \
        -o ${patchcat}
    echo
done

# Based on the previously computed base catalogue create 100 realisations
# of the photometry and according photo-zs (see ./mocks*.sh).
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> generate photometry realizations and apply ${survey} selection"
    mkdir -p ${OUTROOT}/${survey}_phot_samples
    patchcat=${OUTROOT}/MICE2_all_CC_${survey}_photnoise_base.fits
    n_obj=$(data_table_shape ${HOME}/DATA/KV450/SPECZ/DIR_${survey}.fits)
    statfile=${OUTROOT}/${survey}_phot_samples/${survey}_selection.stats
    test -e $statfile && rm -v $statfile
    for i_phot in $(seq 1 $Nrealisations); do
        echo "--> creating realisation $i_phot / $Nrealisations"
        # Create a photometry realisation, do not forget to set a different
        # seed for every iteration
        mocks_photometry_realisation \
            -i ${patchcat} \
            --filters \
                sdss_u_evo_mag sdss_g_evo_mag sdss_r_evo_mag \
                sdss_i_evo_mag sdss_z_evo_mag des_asahi_full_y_evo_mag \
                vhs_j_evo_mag vhs_h_evo_mag vhs_ks_evo_mag \
            --limits $MAGlims \
            --significance $MAGsig \
            --sn-factors \
                sn_factor_sdss_u sn_factor_sdss_g sn_factor_sdss_r \
                sn_factor_sdss_i sn_factor_sdss_z sn_factor_des_asahi_full_y \
                sn_factor_vhs_j sn_factor_vhs_h sn_factor_vhs_ks \
            --seed KV450_$i_phot \
            -o ${OUTROOT}/temp.fits
        data_table_hstack \
            -i ${patchcat} ${OUTROOT}/temp.fits \
            -o ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_all.fits
        # Run BPZ on the photometry realisation.
        mocks_bpz_wrapper \
            -i ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_all.fits \
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
            --z-true z_cgal_v \
            --z-min 0.06674 \
            --z-max 1.42667 \
            --templates CWWSB_capak \
            --prior NGVS \
            --prior-filter sdss_i_obs_mag \
            -o ${OUTROOT}/temp.fits
        data_table_hstack \
            -i ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_all.fits \
               ${OUTROOT}/temp.fits \
            -o ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_all.fits
        # Apply the spec-z survey selection function, sample down to the number
        # of objects in the DIR spec-z catalogues.
        mocks_MICE_specz_sample \
            -s ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_all.fits \
            --s-type KV450 --n-data ${n_obj} \
            --survey ${survey} --pass-phot-detection --stats-file a \
            -o ${OUTROOT}/${survey}_phot_samples/${survey}_phot_samples_${i_phot}_specz.fits
    done
done
rm -v ${OUTROOT}/temp.fits

# Simplify the output data structure and apply the spec-z survey selection
# functions. Read the number of objects in the DIR spec-z catalogues to sample
# down the mocks accordingly.
for survey in DEEP2 VVDSf02 zCOSMOS; do
    echo "==> apply ${survey} selection function"
    n_obj=$(data_table_shape ${HOME}/DATA/KV450/SPECZ/DIR_${survey}.fits)
    statfile=${OUTROOT}/${survey}_dithers/${survey}_selection.stats
    test -e $statfile && rm -v $statfile
    for fits in ${OUTROOT}/${survey}_dithers/${survey}_dither*/pointing_spec.cat; do
        pointing_id=$(basename $(dirname $fits))
        echo "--> processing pointing $pointing_id"
        mocks_MICE_specz_sample \
            -s $fits \
            --s-type KV450 --n-data ${n_obj} \
            --survey ${survey} --pass-phot-detection --stats-file a \
            -o ${OUTROOT}/${survey}_dithers/$(basename $(dirname $fits))_specz.fits
        mv -v $(dirname $fits)/pointing_phot.cat \
              ${OUTROOT}/${survey}_dithers/$(basename $(dirname $fits))_KV450.fits
        rm -r $(dirname $fits)
    done
    echo
done
