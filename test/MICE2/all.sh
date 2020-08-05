#!/usr/bin/env bash

./1_init_pipeline.sh $1 ${2:-32}
./2_evolution_correction.sh $1 ${2:-32}
./3_magnification_correction.sh $1 ${2:-32}
./4_effective_radius.sh $1 ${2:-32}
./5_apertures.sh $1 ${2:-32}
./6_photometry.sh $1 ${2:-32}
./7_match_weights.sh $1 ${2:-32}
./8_photoz.sh $1 ${2:-32}

./get_fits.sh $1 ${2:-32}
