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


# mapping between the job ID (see commandline parser) and the script signature
job_map = {
    "1": "create",
    "2": "prepare_MICE2",
    "3": "magnification",
    "4": "effective_radius",
    "5": "apertures",
    "6": "photometry",
    "7": "match_data",
    "8": "BPZ",
    "9": "select_sample",
    "out": "query"}

# generate a help message for the commandline parser
job_help_str = "select a set of job IDs to process the mock data, "
job_help_str += "options are: {{{:}}} or 'all' to run all jobs".format(
    ", ".join(
        "{:}:{:}".format(ID, job_map[ID]) for ID in sorted(job_map.keys())))


parser = argparse.ArgumentParser(
    description="Pipeline to create realistic KiDS and spectroscopic mock "
                "samples from the MICE2 galaxy mock catalogue.",
    epilog="Job IDs are automatically ordered numerically.")
parser.add_argument(
    "type", choices=("all", "deep", "test"), help="MICE2 data sample type")
parser.add_argument(
    "jobID", nargs="*", help=job_help_str)
parser.add_argument(
    "--threads", type=int, default=cpu_count(),
    help="maximum number of threads to use (default: %(default)s)")
parser.add_argument(
    "--format", default="fits",
    help="file format of output files (default: %(default)s)")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="display debugging messages")


def main():
    args = parser.parse_args()
    if "all" in args.jobID:
        args.jobID = job_map.keys()
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
    for jobID in args.jobID - set(job_map.keys()):
        raise parser.error("invalid job ID: {:}".format(jobID))

    # run the jobs
    if "1" in args.jobID:
        mocks = GalaxyMock.create(
            datastore, input=input_file, purge=True,
            columns="config/MICE2.toml", threads=args.threads)
    else:
        mocks = GalaxyMock(datastore, readonly=False, threads=args.threads)

    with mocks:
        if "2" in args.jobID:
            getattr(mocks, job_map["2"])(
                mag="mags/model", evo="mags/evolved")
        if "3" in args.jobID:
            getattr(mocks, job_map["3"])(
                mag="mags/evolved", lensed="mags/lensed")
        if "4" in args.jobID:
            getattr(mocks, job_map["4"])(
                config="config/photometry.toml")
        if "5" in args.jobID:
            getattr(mocks, job_map["5"])(
                config="config/photometry.toml")
        if "6" in args.jobID:
            getattr(mocks, job_map["6"])(
                config="config/photometry.toml", mag="mags/lensed",
                real="mags/KV450")
        if "7" in args.jobID:
            getattr(mocks, job_map["7"])(
                config="config/matching.toml")
        if "8" in args.jobID:
            os.environ["hostname"] = os.uname()[1]
            getattr(mocks, job_map["8"])(
                config="config/BPZ.toml", mag="mags/KV450", zphot="BPZ/KV450")
        if "9" in args.jobID:
            for sample in samples:
                getattr(mocks, job_map["9"])(
                    config="samples/{:}.toml".format(sample), type="MICE2",
                    sample=sample, area=area)
        if "out" in args.jobID:
            getattr(mocks, job_map["out"])(
                output=output_base.format(args.type, ""), verify=True,
                format=args.format, query=query)
            # get the remaining samples
            for sample in samples:
                getattr(mocks, job_map["out"])(
                    output=output_base.format(args.type, "_" + sample),
                    format=args.format, query=query_sample.format(sample, 1))
            # get the BOSS sample
            getattr(mocks, job_map["out"])(
                output=output_base.format(args.type, "_BOSS"),
                format=args.format, query=query_sample.format("SDSS", 12))


if __name__ == "__main__":
    main()
