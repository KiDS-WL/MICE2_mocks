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
                "samples from the MICE2 galaxy mock catalogue.",
    epilog="Job IDs are automatically ordered numerically.")
parser.add_argument(
    "type", choices=("all", "deep", "test"), help="MICE2 data sample type")
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
    "--format", default="fits",
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
    base_path = "/net/home/fohlen12/jlvdb/DATA/{:}/MICE2_"
    base_path += "{:}_uBgVrRciIcYJHKs_shapes_halos_WL.fits"
    samples = ["KiDS", "2dFLenS", "GAMA", "SDSS"]
    # configure the output files and sample selection
    query = "{ra:} >= 40 AND {ra:} < 45 AND {dec:} >= 10 AND {dec:} < 15"
    query = query.format(ra="position/ra/obs", dec="position/dec/obs")
    query_sample = query + " AND samples/{:} & {:d}"
    output_base = "/net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/"
    output_base += "MICE2_query_{:}{:}." + args.format

    if args.type == "all":  # all
        input_file = base_path.format("MICE2_KV_full", "all")
        datastore = "/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_all"
        samples += ["WiggleZ", "DEEP2", "VVDSf02", "zCOSMOS"]
        area = 5156.6

    elif args.type == "deep":  # deep
        input_file = base_path.format("MICE2_KV450", "deep")
        datastore = "/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_crop"
        samples += ["WiggleZ", "DEEP2", "VVDSf02", "zCOSMOS"]
        area = 859.44

    else:  # test
        input_file = base_path.format("MICE2_KV_full", "256th")
        datastore = "/net/home/fohlen12/jlvdb/DATA/MICE2_KiDS_sparse"
        # sample density insufficient for some samples
        area = 5156.6

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
            columns="config/MICE2.toml", threads=args.threads)
    else:
        mocks = GalaxyMock(datastore, readonly=False, threads=args.threads)

    with mocks:
        if "2" in args.jobID:
            mocks.prepare_MICE2(
                mag="mags/model", evo="mags/evolved")
        if "3" in args.jobID:
            mocks.magnification(
                mag="mags/evolved", lensed="mags/lensed")
        if "4" in args.jobID:
            mocks.effective_radius(
                config="config/photometry.toml")
        if "5" in args.jobID:
            mocks.apertures(
                config="config/photometry.toml")
        if "6" in args.jobID:
            mocks.photometry(
                config="config/photometry.toml", mag="mags/lensed",
                real="mags/KV450")
        if "7" in args.jobID:
            mocks.match_data(
                config="config/matching.toml")
        if "8" in args.jobID:
            os.environ["hostname"] = os.uname()[1]
            mocks.BPZ(
                config="config/BPZ.toml", mag="mags/KV450", zphot="BPZ/KV450")
        if "9" in args.jobID:
            for sample in samples:
                mocks.select_sample(
                    config="samples/{:}.toml".format(sample), type="MICE2",
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
