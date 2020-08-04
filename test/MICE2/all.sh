#!/usr/bin/env bash

./1_init_pipeline.sh $1
./2_evolution_correction.sh $1
./3_magnification_correction.sh $1
./4_effective_radius.sh $1
./5_apertures.sh $1
./6_photometry.sh $1
./7_match_weights.sh $1
./8_photoz.sh $1

./get_fits.sh $1
