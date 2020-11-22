#!/usr/bin/env python3
import os

import mmaptable
import numpy as np
from astropy.io import fits as pyfits

from galmock import GalaxyMock
from galmock.core.utils import ProgressBar
from galmock.photometry import PhotometryParser


configuration = PhotometryParser(
    os.path.join(os.path.dirname(__file__), "config/photometry.toml"))
non_detect = configuration.photometry["no_detect_value"]

print("loading ImSim table")
with pyfits.open("/net/home/fohlen12/jlvdb/IMPORT/SimPhoto_v1.1_Flagship_ra35-55_dec5-15.fits") as fits:
    data = fits[1].data
    print("sorting indices")
    order = np.argsort(data["index"])
    index_filt = data["index"][order]

print("creating filtered data store")
basepath = "/net/home/fohlen13/jlvdb/DATA"
outpath = os.path.join(basepath, "Flagship_KiDS_ImSim_test")
with mmaptable.MmapTable(os.path.join(basepath, "Flagship_KiDS")) as dsin:
    # create the columns in the filtered data store
    with mmaptable.MmapTable(outpath, len(index_filt), mode="w+") as dsout:
        for column in dsin.colnames:
            dsout.add_column(
                column, dsin[column].dtype, attr=dsin[column].attr,
                overwrite=True)

    with GalaxyMock(outpath, readonly=False) as dsout:
        # read the data in chunks and apply the index filter
        row_filt = 0
        pbar = ProgressBar(n_rows=len(dsin))
        for row_all, index in enumerate(dsin["index"]):
            if index == index_filt[row_filt]:
                dsout.datastore[row_filt] = dsin[row_all].to_records()
                row_filt += 1
            if row_all % 100 == 0:
                pbar.update(100)
            if row_filt == len(dsout):
                break
        pbar.close()

        print("ingest the additional table columns")
        keys = ("u", "g", "r", "i", "Z", "Y", "J", "H", "Ks")
        # add common columns
        for colpath in ("mags/sim", "mags/sim_fudge"):
            desc = "name of the KiDS tile the image simulation is based on"
            key = "KiDS_tile"
            print("   ", key)
            col = dsout.datastore.add_column(
                os.path.join(colpath, key),
                dtype=data[key].dtype, attr={"description": desc},
                overwrite=True)
            col[:] = data[key][order]
            key = "mags/sim/MAG_AUTO"
            print("   ", key)
            col = dsout.datastore.add_column(
                os.path.join(colpath, "MAG_AUTO"),
                dtype=data[key].dtype, attr={
                    "description": "Source Extractor MAG_AUTO"},
                overwrite=True)
            # filter bad values
            values = data[key][order]
            values[np.isnan(values)] = non_detect
            values[np.isinf(values)] = non_detect
            col[:] = np.minimum(values, non_detect)
        # add the magnitudes
        for mag_type in ("sim", "sim_fudge"):
            for mag_key in keys:
                # magnitude
                key = "mags/sim/MAG_GAAP_0p7_{:}".format(mag_key)
                print("   ", key)
                desc = "{:}-band GAaP magnitude measured in 0.7\" apertures"
                col = dsout.datastore.add_column(
                    "mags/{:}/{:}".format(mag_type, mag_key),
                    dtype=data[key].dtype, attr={
                        "description": desc.format(mag_key)},
                    overwrite=True)
                # filter bad values
                values = data[key][order]
                values[np.isnan(values)] = non_detect
                values[np.isinf(values)] = non_detect
                col[:] = np.minimum(values, non_detect)
                detect_mask = values == non_detect
                # error
                if mag_type == "sim":
                    key = "mags/sim/MAGERR_GAAP_0p7_{:}".format(mag_key)
                else:
                    key = "mags/sim/MAGERR_GAAP_fudge_{:}".format(mag_key)
                print("   ", key)
                desc = "error of {:}-band GAaP magnitude"
                col = dsout.datastore.add_column(
                    "mags/{:}/{:}_err".format(mag_type, mag_key),
                    dtype=data[key].dtype, attr={
                        "description": desc.format(mag_key)},
                    overwrite=True)
                # filter bad values
                values = data[key][order]
                values[detect_mask] = configuration.limits[mag_key]
                values[values <= 0.0] = configuration.limits[mag_key]
                col[:] = np.minimum(values, configuration.limits[mag_key])

        # update the check sum to match the subset
        print()
        dsout.verify(recalc=True)
