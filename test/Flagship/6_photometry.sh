#!/usr/bin/env bash

../../scripts/mocks_photometry \
    /net/home/fohlen12/jlvdb/DATA/Flagship_KiDS \
    -c ../KV450_photometry.toml \
    --method GAaP \
    --mag mags/lensed \
    --real mags/KV450
