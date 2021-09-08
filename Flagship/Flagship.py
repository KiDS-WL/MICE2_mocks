#!/usr/bin/env python3
import argparse
import logging.config
import os
import subprocess
import shlex
from collections import OrderedDict
from multiprocessing import cpu_count

from galmock import GalaxyMock
from galmock.core.config import logging_config


parser = argparse.ArgumentParser(
    description="Pipeline to create realistic KiDS and spectroscopic mock "
                "samples from the Flagship galaxy mock catalogue.",
    epilog="Job IDs are automatically ordered numerically.")
parser.add_argument(
    "type", choices=("all", "test"), help="MICE2 data sample type")
parser.add_argument(
    "jobID", nargs="*",
    help="select a set of job IDs to process the mock data, options are: "
         "1: create datastore - 2: prepare Flagship - 3: add magnification - "
         "4: effective radius - 5: add apertures - 6: add photometry - "
         "7: match data - 8: BPZ photo-z - 9: select samples - out: export "
         "samples - all: run all jobs in sequence")
parser.add_argument(
    "--threads", type=int, default=cpu_count(),
    help="maximum number of threads to use (default: %(default)s)")
parser.add_argument(
    "--format", default="hdf5",
    help="file format of output files (default: %(default)s)")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="display debugging messages")

all_jobs = tuple(str(i) for i in range(1, 10)) + ("out",)


def main():
    args = parser.parse_args()
    if "all" in args.jobID:
        args.jobID = all_jobs
    args.jobID = set(args.jobID)
    args.verbose = "-v" if args.verbose else ""

    # configure the data paths and sample selections
    base_path = "/net/home/fohlen13/jlvdb/DATA/Flagship_HDF5/Flagship_v1-8-4_deep_ugrizYJHKs_shapes_halos_WL{:}.hdf5"
    samples = [
        "KiDS", "2dFLenS", "GAMA", "SDSS", "WiggleZ",
        "DEEP2", "VVDSf02", "zCOSMOS"]
    # configure the output files and sample selection
    query = "{ra:} >= 40 AND {ra:} < 45 AND {dec:} >= 10 AND {dec:} < 15"
    query = query.format(ra="position/ra/obs", dec="position/dec/obs")
    query_sample = query + " AND samples/{:} & {:d}"
    output_base = "/net/home/fohlen13/jlvdb/TEST/MOCK_pipeline/Flagship_query_{:}{:}." + args.format

    if args.type == "all":  # all
        input_file = base_path.format("")
        datastore = "/net/home/fohlen13/jlvdb/DATA/Flagship_KiDS"
        area = 5156.6

    else:  # test
        input_file = base_path.format("_test")
        datastore = "/net/home/fohlen13/jlvdb/DATA/Flagship_KiDS_test"
        # sample density insufficient for some samples
        area = 24.400

    # create a logger for the pipeline
    overwrite = "1" in args.jobID
    logging.config.dictConfig(
        logging_config(datastore + ".log", overwrite=overwrite,
        verbose=args.verbose))

    # check for unknown jobs
    for jobID in args.jobID - set(all_jobs):
        raise parser.error("invalid job ID: {:}".format(jobID))

    # run the jobs
    if "1" in args.jobID:
        mocks = GalaxyMock.create(
            datastore, input=input_file, purge=True, index="index",
            columns="config/Flagship.toml", threads=args.threads)
    else:
        mocks = GalaxyMock(datastore, readonly=False, threads=args.threads)

    with mocks:
        if "2" in args.jobID:
            mocks.prepare_Flagship(
                flux="flux/model", mag="mags/model", gal_idx="index_galaxy",
                is_central="environ/is_central")
        if "3" in args.jobID:
            mocks.magnification(
                mag="mags/model", lensed="mags/lensed")
        if "4" in args.jobID:
            mocks.effective_radius(
                config="config/photometry.toml")
        if "5" in args.jobID:
            mocks.apertures(
                config="config/photometry.toml")
        if "6" in args.jobID:
            mocks.photometry(
                config="config/photometry.toml", mag="mags/lensed",
                real="mags/K1000")
        if "7" in args.jobID:
            mocks.match_data(
                config="config/matching.toml")
        if "8" in args.jobID:
            os.environ["hostname"] = os.uname()[1]
            mocks.BPZ(
                config="config/BPZ.toml", mag="mags/K1000", zphot="BPZ/K1000")
        if "9" in args.jobID:
            for sample in samples:
                mocks.select_sample(
                    config="samples/{:}.toml".format(sample), type="Flagship",
                    sample=sample, area=area)
        if "out" in args.jobID:
            mocks.query(
                output=output_base.format(args.type, ""), verify=True,
                format=args.format, query=query)
            # get the remaining samples
            for sample in samples:
                mocks.query(
                    output=output_base.format(args.type, "_" + sample),
                    format=args.format, query=query_sample.format(sample, 1))
            # get the BOSS sample
            mocks.query(
                output=output_base.format(args.type, "_BOSS"),
                format=args.format, query=query_sample.format("SDSS", 12))


if __name__ == "__main__":
    main()
