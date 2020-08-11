#!/usr/bin/env bash

./1_init_pipeline.sh $1 $2
./2_flux_to_magnitudes.sh $1 $2
./3_magnification_correction.sh $1 $2
./4_effective_radius.sh $1 $2
./5_apertures.sh $1 $2
./6_photometry.sh $1 $2
./7_match_weights.sh $1 $2
./8_photoz.sh $1 $2
./9_samples.sh $1 $2

if [ "$HOSTNAME" == "fohlen12" ]
then
    ./get_fits.sh $1 $2
else
    ./get_hdf5.sh $1 $2
fi
