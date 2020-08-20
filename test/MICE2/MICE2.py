#!/usr/bin/env python3
import argparse
import os
import subprocess
import shlex
from collections import OrderedDict
from multiprocessing import cpu_count

from mock_processing import jobs


# mapping between the job ID (see commandline parser) and the script signature
job_map = {
    "1": "mocks_init_pipeline {:} "
         "-i {input:} -c {columns:} --purge",
    "2": "MICE2_evolution_correction {:} "
         "--mag {mag:} --evo {evo:} --threads {threads:}",
    "3": "mocks_magnification {:} "
         "--mag {mag:} --lensed {lensed:} --threads {threads:}",
    "4": "mocks_effective_radius {:} "
         "-c {config:} --threads {threads:}",
    "5": "mocks_apertures {:} "
         "-c {config:} --method SExtractor --threads {threads:}",
    "6": "mocks_photometry {:} "
         "-c {config:} --method SExtractor --mag {mag:} --real {real:} "
         "--threads {threads:}",
    "7": "mocks_match_data {:} "
         "-c {config:} --threads {threads:}",
    "8": "mocks_BPZ {:} "
         "-c {config:} --mag {mag:} --zphot {zphot:} --threads {threads:}",
    "9": "mocks_select_sample {:} "
         "-c {config:} --area {area:} --sample {sample:} --threads {threads:}",
    "out": "mocks_datastore_query {:} "
           "-o {output:} -q '{query:}' --format {format:}"}

# generate a help message for the commandline parser
job_help_str = "select a set of job IDs to process the mock data, "
job_help_str += "options are: {{{:}}} or 'all' to run all jobs".format(
    ", ".join(
    "{:}:{:}".format(ID, job_map[ID].split()[0])
    for ID in sorted(job_map.keys())))


parser = argparse.ArgumentParser(
    description="Pipeline to create realistic KiDS and spectroscopic mock "
                "samples from the MICE2 galaxy mock catalogue.",
    epilog="Job IDs are automatically ordered numerically.")
parser.add_argument(
    "type", choices=("all", "deep", "test"), help="MICE2 data sample type")
parser.add_argument(
    "jobID", nargs="*", help=job_help_str)
parser.add_argument(
    "--threads", type=int, default=cpu_count() // 2,
    help="maximum number of threads to use (default: %(default)s)")
parser.add_argument(
    "--format", default="fits",
    help="file format of output files (default: %(default)s)")


def call(schema, *args, **kwargs):
    # get the script directory and combine it with the script schema
    command = os.path.normpath(
        os.path.join(jobs.__file__, "..", "..", "scripts", schema))
    # format the schema with the provided arguments and keyword arguments
    command = command.format(*args, **kwargs)
    # execute the command
    subprocess.call(shlex.split(command))
    print()


def main():
    args = parser.parse_args()
    if "all" in args.jobID:
        args.jobID = job_map.keys()
    args.jobID = set(args.jobID)

    # configure the data paths and sample selections
    base_path = "/net/home/fohlen12/jlvdb/DATA/{:}/MICE2_{:}_uBgVrRciIcYJHKs_shapes_halos_WL.fits"
    samples = ["KiDS", "2dFLenS", "GAMA", "SDSS"]
    # configure the output files and sample selection
    query = "{ra:} >= 40 AND {ra:} < 45 AND {dec:} >= 10 AND {dec:} < 15"
    query = query.format(ra="position/ra/obs", dec="position/dec/obs")
    query_sample = query + " AND samples/{:} & {:d}"
    output_base = "/net/home/fohlen12/jlvdb/TEST/MOCK_pipeline/MICE2_query_{:}{:}." + args.format

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

    # check for unknown jobs
    for jobID in args.jobID - set(job_map.keys()):
        raise parser.error("invalid job ID: {:}".format(jobID))

    # run the jobs
    if "1" in args.jobID:
        call(
            job_map["1"], datastore, input=input_file,
            columns="config/MICE2.toml")
    if "2" in args.jobID:
        call(
            job_map["2"], datastore, mag="mags/model", evo="mags/evolved",
            threads=args.threads)
    if "3" in args.jobID:
        call(
            job_map["3"], datastore, mag="mags/evolved", lensed="mags/lensed",
            threads=args.threads)
    if "4" in args.jobID:
        call(
            job_map["4"], datastore, config="config/photometry.toml",
            threads=args.threads)
    if "5" in args.jobID:
        call(
            job_map["5"], datastore, config="config/photometry.toml",
            method="SExtractor", threads=args.threads)
    if "6" in args.jobID:
        call(
            job_map["6"], datastore, config="config/photometry.toml",
            method="SExtractor", mag="mags/lensed", real="mags/KV450",
            threads=args.threads)
    if "7" in args.jobID:
        call(
            job_map["7"], datastore, config="config/matching.toml",
            threads=args.threads)
    if "8" in args.jobID:
        os.environ["hostname"] = os.uname()[1]
        call(
            job_map["8"], datastore, config="config/BPZ.toml",
            mag="mags/KV450", zphot="BPZ/KV450", threads=args.threads)
    if "9" in args.jobID:
        for sample in samples:
            call(
                job_map["9"], datastore, sample=sample, area=area,
                config="samples/{:}.toml".format(sample),
                threads=args.threads)
    if "out" in args.jobID:
        call(
            job_map["out"] + " --verify", datastore,
            output=output_base.format(args.type, ""),
            format=args.format, query=query)
        # get the remaining samples
        for sample in samples:
            call(
                job_map["out"], datastore,
                output=output_base.format(args.type, "_" + sample),
                format=args.format, query=query_sample.format(sample, 1))
        # get the BOSS sample
        call(
            job_map["out"], datastore,
            output=output_base.format(args.type, "_BOSS"),
            format=args.format, query=query_sample.format("SDSS", 12))


if __name__ == "__main__":
    main()
