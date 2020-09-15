# MICE2_mocks

This Repository provides code to derive survey specific mock cataloges from an
existing input simulated galaxy mock catalogue such as the Euclid Flagship or
MICE2. Even though is pipeline is writting for the Kilo-Degree Survey, all
methods should be applicable to other reference data sets.

The processing pipeline offers a variety of featues, such as:

- adding realistic photometry realisations (mimicking real world observations),
- photometric redshifts (using BPZ) and
- selecting spectroscopic samples such as SDSS BOSS, GAMA, zCOSMOS and more.

The functionality can easily be extended to further needs.

The pipeline is based on an internal, memory-map based data store that is
created when the pipeline is initialsed. This data layout allows extremely
efficient data access, allowing parallel reading and writeing operations. All
data processing is logged and data columns in the data store are marked with
desciptive attributes. Once all computations are complete, the data store can
be queried with logical expressions to select output (sub-)samples. Supported
in- and output formats are (currently) CSV, FITS, HDF5 and Parquet.

The pipeline can be accessed by dedicated command line scripts or directly
through a Python interface.


## Requirements

The pipeline is written in Python3 and requires the additional packages
summarized in the `requirements.txt` file.

To be able to compute photometric redshifts
[BPZ](http://www.stsci.edu/~dcoe/BPZ/) and a python2 environment is requried,
see `requirements_BPZ_py2.txt`.

The root directory of this repository should be included in the `$PYTHONPATH`.
Optionally, the `scripts` can be included in `$PATH` to directly accessing the
command line tools.


## Instructions

Examples for how to set up and run the pipeline from python can be found in
`MICE2/MICE2.py` and `Flagship/Flagship.py` directories. This includes samples
of the configuration files (photometry, photo-z, spectroscopic reference
samples) and the transmission curves required by BPZ (to be unpacked from the
`filters.tar.gz` files).

All pipeline methods are implemented in the main class, `galmock.GalaxyMock`.
Alternatively, these methods can be used by calling command line tools in the
`scripts` directory. Type `[scriptname] --help` to obtain more information
about the usage of each of these scripts.

Many of the processing steps are configured using configuration files. Default
versions of these files can be generated using the command line tools by typing
`[scriptname] --dump`, which will print an commented configuration file to
the standard output.


### Data Access

The MICE2 and Flagship base catalogues can be downloaded from
[COSMO HUB](https://cosmohub.pic.es/). Recommended column selections (in SQL
query stile using 'expert mode') to reproduce KiDS ugrizYJHKs data are for
MICE2:
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
and for Flagship:
```
SELECT
    `halo_id`, `galaxy_id`, `ra_gal`, `dec_gal`, `ra_gal_mag`, `dec_gal_mag`,
    `true_redshift_gal`, `observed_redshift_gal`,
    `kappa`, `gamma1`, `gamma2`, `ellipticity`,
    `log_stellar_mass`, `halo_lm`, `kind`, `halo_n_sats`,
    `kids_u`, `kids_g`, `kids_r`, `kids_i`, `lsst_z`,
    `lsst_y`, `2mass_h`, `2mass_j`, `2mass_ks`,
    `bulge_fraction`, `bulge_length`, `bulge_axis_ratio`, `bulge_angle`,
    `disk_length`, `disk_axis_ratio`, `disk_angle`
FROM flagship_mock_1_8_4_s
```


## Citing galmock

Papers utilizing the `galmock` pipeline should provide a link back to this
repository. It is also requested that users cite
> van den Busch et al. 2020


## Maintainers

Jan Luca van den Busch (Ruhr-Universität Bochum, Germany) - [https://github.com/jlvdb]()
