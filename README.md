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


## Citing galmock (or MICE2_mocks)

Papers utilizing the `galmock` pipeline should provide a link back to this
repository. It is also requested that users cite
> van den Busch et al. 2020


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
```sql
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
```sql
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


## Usage examples

This is a summary on how to create KiDS-like mock data based on a Flagship
input catalogue, selected according to the SQL query from the previous section
and using the configuration files in `Flagship/config`.

### From the command line

Read the input simulation into the data store:
```bash
scripts/mocks_init_pipeline path/to/datastore \
    -i /path/to/Flagship.fits --config Flagship/config/Flagship.toml
```

Run some preprocessing that is specific to the input simulation (here most
importantly converting fluxes to magnitudes):
```bash
scripts/mocks_prepare_Flagship path/to/datastore \
    --flux flux/model --mag mags/model \
    --gal-idx index_galaxy --is-central environ/is_central
```

Correct the model magnitudes for magnification induced by gravitational
lensing:
```bash
scripts/mocks_magnification path/to/datastore \
    --mag mags/model --lensed mags/lensed
```

Compute an intrisinc size for each galaxy:
```bash
scripts/mocks_effective_radius path/to/datastore \
    --config Flagship/config/photometry.toml
```

Construct apertures based on the galaxy size and the KiDS median PSF size to
obtain a more realistic signal-to-noise distribution:
```bash
scripts/mocks_apertures path/to/datastore \
    --config Flagship/config/photometry.toml
```

Add a photometry realisation based on the KiDS median limiting magnitudes:
```bash
scripts/mocks_photometry path/to/datastore \
    --config Flagship/config/photometry.toml \
    --mag mags/lensed --real mags/K1000
```

Derive `lensfit` shape weights by matching to the KiDS data magnitude space:
```bash
scripts/mocks_match_data path/to/datastore \
    --config Flagship/config/matching.toml
```
> NOTE: This requires an unfiltered version of the KiDS cosmic shear catalogue.

Add photometric redshifts using BPZ. This requires the Raichoor NVGS prior and
the Capak CWWSB SED templates. The filter curves are already shipped with the
pipeline in `Flagship/filters.tar.gz`.
```bash
scripts/mocks_BPZ path/to/datastore \
    --config Flagship/config/BPZ.toml \
    --mag mags/KV450 --zphot BPZ/K1000
```
> NOTE: Depending on the current working directory the paths to the filter
transmission curves may need to be modified.

Run the KiDS selection function:
```bash
scripts/mocks_select_sample path/to/datastore \
    --sample KiDS --config Flagship/samples/KiDS.toml --area 5156.6
```
> NOTE: The exact area of the input simulation catalogue is required for same
sample selection functions.

Export the subset of KiDS galaxies to a fits file:
```bash
scripts/mocks_datastore_query path/to/datastore \
    --query 'samples/KiDS & 1' \
    -o path/to/output.fits --verify
```
> NOTE: The `--verify` flag can be used to verify the check sums of the data
store.

All these operations can be applied from within Python using the corresponding
class methods of `galmock.Galmock`, for an example see `Flagship/Flagship.py`.


### Advanced examples

The underlying data store of the pipeline can be manipulated manually as well.
The recommend ways to access the data is via `galmock.GalaxyMock.datastore`,
which is a wrapper for the underlying `mmaptable.MmapTable` format.

The following example shows how to add new columns to an existing pipeline data
store:

```python
import os
import sys

import numpy as np
from astropy.io import fits as pyfits

from galmock import GalaxyMock


# create some dummy FITS file
fits_input_path = "testfile.fits"
col1 = pyfits.Column(
    name="mycol1", format="E", array=np.arange(0, len(gm), dtype="f4"))
col2 = pyfits.Column(
    name="mycol2", format="J", array=np.arange(len(gm), 0, -1, dtype="i4"))
cols = pyfits.ColDefs([col1, col2])
hdu = pyfits.BinTableHDU.from_columns(cols)
hdu.writeto(fits_input_path, overwrite=True)


# open an existing pipeline data store
datastore_path = "/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse"
with GalaxyMock(datastore_path, readonly=False) as gm:

    # we prefer the native byteorder in the data store
    numpy_byteorder_prefix = "<" if sys.byteorder == "little" else ">"

    # open the file again and copy the data by column into the datastore    
    with pyfits.open(fits_input_path) as fits:
        data = fits[1].data  # get the data from the first extension
        print("input data:")
        print(data)

        names = []
        for column in data.columns:
            name = column.name
            dtype = column.dtype.str
            # get the data type in native byte order (optional)
            dtype = dtype.replace("<", numpy_byteorder_prefix)
            dtype = dtype.replace(">", numpy_byteorder_prefix)

            # create the new column
            names.append(name)
            gm.datastore.add_column(name, dtype, overwrite=True)
            # assign the data, data store columns behave like numpy arrays
            gm.datastore[name][:] = column.array[:]

        # print a preview of the newly added columns
        print("inserted data:")
        # the data store supports fancy indexing, here with a list of column names
        print(gm.datastore[names])

    # delete the test file
    os.remove(fits_input_path)
```

The effect can be verified using the data store information script:

```bash
scripts/mocks_datastore_info /net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse -c
```

The data store automatically adds check sums for the new columns which can be
checked and verified by: 

```bash
scripts/mocks_datastore_verify /net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse
```

Finally, we can select the new columns in the data store and write a row subset
to a FITS file:

```bash
scripts/mocks_datastore_query \
    /net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse \
    --query 'mycol1 > mycol2' --columns mycol1 mycol2 \
    --output mydata.fits
```



## Maintainers

Jan Luca van den Busch (Ruhr-Universität Bochum, Germany) - [https://github.com/jlvdb]()
