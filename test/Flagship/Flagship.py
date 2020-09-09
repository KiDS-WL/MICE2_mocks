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
    "2": "prepare_Flagship",
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
                "samples from the Flagship galaxy mock catalogue.",
    epilog="Job IDs are automatically ordered numerically.")
parser.add_argument(
    "type", choices=("all", "test"), help="MICE2 data sample type")
parser.add_argument(
    "jobID", nargs="*", help=job_help_str)
parser.add_argument(
    "--threads", type=int, default=cpu_count(),
    help="maximum number of threads to use (default: %(default)s)")
parser.add_argument(
    "--format", default="hdf5",
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
    for jobID in args.jobID - set(job_map.keys()):
        raise parser.error("invalid job ID: {:}".format(jobID))

    # run the jobs
    if "1" in args.jobID:
        mocks = GalaxyMock.create(
            datastore, input=input_file, purge=True,
            columns="config/Flagship.toml", threads=args.threads)
    else:
        mocks = GalaxyMock(datastore, readonly=False, threads=args.threads)

    with mocks:
        if "2" in args.jobID:
            getattr(mocks, job_map["2"])(
                flux="flux/model", mag="mags/model", gal_idx="index_galaxy",
                is_central="environ/is_central")
        if "3" in args.jobID:
            getattr(mocks, job_map["3"])(
                mag="mags/model", lensed="mags/lensed")
        if "4" in args.jobID:
            getattr(mocks, job_map["4"])(
                config="config/photometry.toml")
        if "5" in args.jobID:
            getattr(mocks, job_map["5"])(
                config="config/photometry_old.toml")
        if "6" in args.jobID:
            getattr(mocks, job_map["6"])(
                config="config/photometry.toml", mag="mags/lensed",
                real="mags/K1000")
        if "7" in args.jobID:
            getattr(mocks, job_map["7"])(
                config="config/matching.toml")
        if "8" in args.jobID:
            os.environ["hostname"] = os.uname()[1]
            getattr(mocks, job_map["8"])(
                config="config/BPZ.toml", mag="mags/K1000", zphot="BPZ/K1000")
        if "9" in args.jobID:
            for sample in samples:
                getattr(mocks, job_map["9"])(
                    config="samples/{:}.toml".format(sample), type="Flagship",
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
