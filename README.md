# MICE2_mocks

This Repository provides code to create galaxy mock catalogues based on
[MICE](http://maia.ice.cat/mice/) galaxy catalogues.


## Requirements

The pipeline is written in Python3 and requires the following non-standard
packages:
- `numpy` and `scipy`
- `astropy>=3.0` (recommended for the improved astropy.table performance)
- `matplotlib>=2.0` for the plotting scripts

> Currently the code also depends on a wrapper for the STOMP library. A python3
version can be downloaded [here](https://github.com/jlvdb/astro-stomp3) and
installed following
[these](https://github.com/morriscb/the-wizz/wiki/Stomp-Installation)
instructions.

Additionally, the wrapper scripts in `./KV450` and `./DES` make use of
external packages that provide convenience functions to handling data table
files and STOMP pixel maps:
- [jlvdb/table_tools](https://github.com/jlvdb/table_tools) (script calls
starting with `data_table_`)
- [jlvdb/stomp_tools](https://github.com/jlvdb/stomp_tools) (script calls
starting with `stomp_`)

To be able to compute photometric redshifts
[BPZ](http://www.stsci.edu/~dcoe/BPZ/) is requried.


## Instructions

Starting from the MICE2 base catalogues, the pipeline allows to model various
observational selection functions:

- Spectroscopic surveys: GAMA, SDSS (main sample, BOSS and QSOs), 2dFLenS,
WiggleZ, DEEP2, zCOSMOS and VVDS (2h field)
- Photometric surveys: Examples to create 450 sqdeg of KiDS-VIKING (KV450,
`./KV450`) and the Dark Energy Survey (DES, `./DES`) are included.

The pipeline allows to attach realistic photometry realisations to the MICE2
catalogues, photometric redshifts, galaxy weights, and spectroscopic success
rates for some of the included spectroscopic selection functions.


### Data Access

The MICE2 base catalogues can be downloaded from
[COSMO HUB](https://cosmohub.pic.es/). Recommended column selections (in SQL
query stile using 'expert mode') for KV450 are

```
SELECT
    `unique_gal_id`, `ra_gal`, `dec_gal`, `z_cgal`, `z_cgal_v`,
    `sdss_u_true`, `lephare_b_true`, `sdss_g_true`,
    `lephare_v_true`, `sdss_r_true`, `lephare_rc_true`,
    `sdss_i_true`, `lephare_ic_true`, `sdss_z_true`,
    `des_asahi_full_y_true`, `vhs_j_true`,
    `vhs_h_true`, `vhs_ks_true`,
    `bulge_fraction`, `bulge_length`, `bulge_axis_ratio`,
    `disk_length`, `disk_axis_ratio`
FROM micecatv2_0_view WHERE `dec_gal` <= 30 AND `ra_gal` >= 30 AND `ra_gal` <= 60
```
and for DES correspondingly
```
SELECT
    `unique_gal_id`, `ra_gal`, `dec_gal`, `z_cgal`, `z_cgal_v`,
    `sdss_u_true`, `lephare_b_true`, `sdss_g_true`,
    `lephare_v_true`, `sdss_r_true`, `lephare_rc_true`,
    `sdss_i_true`, `lephare_ic_true`, `sdss_z_true`,
    `des_asahi_full_y_true`, `vhs_j_true`,
    `vhs_h_true`, `vhs_ks_true`,
    `bulge_fraction`, `bulge_length`, `bulge_axis_ratio`,
    `disk_length`, `disk_axis_ratio`
FROM micecatv2_0_view WHERE `dec_gal` <= 30 AND `ra_gal` >= 30 AND `ra_gal` <= 60
```
This selection uses the most complete patch of MICE (`30 <= RA <= 60` and
`0 <= DEC <= 30`). Some of these columns are only needed to additionaly select
spectoscopic samples.


### Creating Photometric Catalogues

The wrapper scripts in `./KV450` and `./DES` show exemplary how to create
mock catalogues matched to observational data. These steps include:

1. Applying the MICE2 evolution correction.
2. Correcting the model magnitudes for magnification.
3. Computing observational galaxy sizes based on the point spread function
(this allows a size contribution to the photometric uncertainties).
4. Adding a photometry realization based on the observational limiting
magnitudes
5. Assigning galaxy weights by nearest neighbour matching between mock and data
in magnitude space
6. Computing photometric redshifts with BPZ


### Creating Spectroscopic Catalogues

[coming soon]
