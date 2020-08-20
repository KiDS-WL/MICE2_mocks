import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from tempfile import TemporaryDirectory
from time import sleep

import numpy as np

from galmock.core.config import (Parameter, ParameterCollection,
                                         ParameterGroup, ParameterListing,
                                         Parser)
from galmock.core.parallel import Schedule
from galmock.core.utils import expand_path


class BpzParser(Parser):

    default = ParameterCollection(
        Parameter(
            "BPZpath", str, "~/src/bpz-1.99.3",
            "path of the bpz installation to use",
            parser=expand_path),
        Parameter(
            "BPZenv", str, None,
            "path to a python2 environment used when calling BPZ (leave blank "
            "to use the default python2 interpreter)",
            parser=expand_path),
        Parameter(
            "BPZtemp", str, None,
            "temporary directory to use for bpz (I/O intense, e.g. a ram disk, "
            "if blank, defaults to data store)",
            parser=expand_path),
        Parameter(
            "flux", bool, False,
            "whether the input columns are fluxes or magnitudes"),
        Parameter(
            "system", str, "AB",
            "photometric system, must be \"AB\" or \"Vega\""),
        ParameterListing(
            "filters", str,
            header=(
                "Mapping between filter keys (same as used in the column map "
                "file for mocks_init_pipeline) and paths to transmission "
                "curve files compatible with BPZ (do not need to be in the "
                "BPZpath/FILTER directory)."),
            parser=expand_path),
        ParameterGroup(
            "prior",
            Parameter(
                "name", str, "hdfn_gen",
                "template-magnitude prior module in the bpz_path directory "
                "(filename: prior_[name].py)"),
            Parameter(
                "filter", str, "i",
                "filter key (one of those listed in the [filters] section) "
                "based on which the prior is evaluated"),
            header=None),
        ParameterGroup(
            "templates",
            Parameter(
                "name", str, "CWWSB4",
                "name of the template .list file in the bpz_path/SED "
                "directory"),
            Parameter(
                "interpolation", int, 10,
                "introduces n points of interpolation between the templates "
                "in the color space"),
            header=None),
        ParameterGroup(
            "likelihood",
            Parameter(
                "zmin", float, 0.01,
                "minimum redshift probed"),
            Parameter(
                "zmax", float, 7.00,
                "maximum redshift probed"),
            Parameter(
                "dz", float, 0.01,
                "redshift resolution, intervals are logarithmic: (1+z)*dz"),
            Parameter(
                "odds", float, 0.68,
                "redshift confidence limits"),
            Parameter(
                "min_rms", float, 0.0,
                "intrinsic scatter of the photo-z in dz/(1+z)"),
            header=None),
        header=(
            "This configuration file is required for mocks_BPZ. It defines "
            "the transmission curves, galaxy templates and Bayesian prior "
            "required to run the BPZ photometric reshift code."))

    def _run_checks(self):
        if self.prior["filter"] not in self.filters:
            message = "prior filter is not included in the filter list: {:}"
            raise KeyError(message.format(self.prior["filter"]))

    @property
    def filter_names(self):
        return tuple(sorted(self.filters.keys()))


class BpzManager(object):

    def __init__(self, config, logger=None):
        self._config = config
        # create a temporary directory
        self._tempdir = TemporaryDirectory(
            prefix=os.path.join(self.config.BPZtemp, "BPZ_"))
        # construct paths to data directories
        self._ab_path = os.path.join(self.tempdir, "AB")
        self._filter_path = os.path.join(self.tempdir, "FILTER")
        self._input_path = os.path.join(self.tempdir, "INPUT")
        self._output_path = os.path.join(self.tempdir, "OUTPUT")
        # create in- and output file templates
        self._columns_file = os.path.join(
            self.tempdir, "input.columns")
        self._input_template = os.path.join(
            self._input_path, "thread{:}.dat")
        self._output_template = os.path.join(
            self._output_path, "thread{:}.dat")
        # the BPZ output data columns of interest
        self._output_dtype = np.dtype([
            ("ID", "i8"), ("Z_B", "f4"), ("Z_B_MIN", "f4"), ("Z_B_MAX", "f4"),
            ("T_B", "f4"), ("ODDS", "f4"), ("Z_ML", "f4"), ("T_ML", "f4"),
            ("CHI-SQUARED", "f4"), ("M_0", "f4")])
        prefix = "BPZ best fit"
        self._output_description = OrderedDict([
            ("Z_B", prefix),
            ("Z_B_MIN", "{:} redshift lower {:.1%}-confidence interval".format(
                prefix, self.config.likelihood["odds"])),
            ("Z_B_MAX", "{:} redshift upper {:.1%}-confidence interval".format(
                prefix, self.config.likelihood["odds"])),
            ("T_B", "{:} template".format(prefix)),
            ("ODDS", "Probability contained in the main BPZ posterior peak"),
            ("Z_ML", "{:} maximum likelihood redshift".format(prefix)),
            ("T_ML", "{:} maximum likelihood template".format(prefix)),
            ("CHI-SQUARED", "{:} chi squared".format(prefix)),
            ("M_0", "reference magnitude of the PBZ prior")])
        # run the initialization
        if logger is not None:
            message = "setting up working directory: {:}"
            logger.debug(message.format(self.tempdir))
        self._init_environment()
        self._init_tempdir()
        self._check_prior_template()
        if logger is not None:
            logger.debug("installing transmission profiles")
        self._install_filters()
        self._write_columns_file()
        self._create_AB_files()

    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.cleanup()

    def cleanup(self):
        self._restore_environment()
        self._tempdir.cleanup()

    def _init_environment(self):
        self._restore_values = {}
        # save any original values to restore later
        for key in ("NUMERIX", "BPZPATH"):
            try:
                self._restore_values[key] = os.environ[key]
            except KeyError:
                self._restore_values[key] = None
        # set the required values
        os.environ["BPZPATH"] = self.tempdir
        os.environ["NUMERIX"] = "numpy"

    def _init_tempdir(self):
        for root, dirs, files in os.walk(self._config.BPZpath):
            if root.split(os.sep)[-1] in ("AB", "FILTER", "test", "output"):
                continue
            for f in files:
                if f.startswith(".") or f.endswith(".pyc"):
                    continue  # system or compiled files
                relpath = os.path.relpath(root, self._config.BPZpath)
                if relpath == ".":
                    relpath = ""
                source = os.path.join(root, f)
                dest = os.path.join(self.tempdir, relpath, f)
                if not os.path.exists(os.path.dirname(dest)):
                    os.mkdir(os.path.dirname(dest))
                shutil.copyfile(source, dest)
        for folder in [
                self._ab_path, self._filter_path,
                self._input_path, self._output_path]:
            os.mkdir(folder)

    def _check_prior_template(self):
        prior = self.config.prior["name"]
        if prior not in self.installed_priors:
            message = "unknown prior: {:} (options are: {:})"
            message = message.format(
                prior, ", ".join(self.installed_priors))
            raise ValueError(message)
        template = self.config.templates["name"]
        if template not in self.installed_templates:
            message = "unknown template list: {:} (options are: {:})"
            message = message.format(
                prior, ", ".join(self.installed_templates))
            raise ValueError(message)

    def _restore_environment(self):
        while len(self._restore_values) > 0:
            key, value = self._restore_values.popitem()
            # unset the variable if it did not exist before
            if value is None:
                del os.environ[key]
            # restore the original value if it did exist before
            else:
                os.environ[key] = value

    def _install_filters(self):
        self._installed_filters = {}
        for name, path in self.config.filters.items():
            if name in self._installed_filters:
                raise ValueError("filter already exists: {:}".format(name))
            dest = os.path.join(self._filter_path, "{:}.res".format(name))
            # copy the transmission file and mark it for later removal
            shutil.copyfile(path, dest)
            self._installed_filters[name] = dest

    def _write_columns_file(self):
        width = max(
            len(name) for name in self.filter_names)
        width = max(width, 8)
        # create the file header
        header = "{:<{width}}  columns  AB/Vega  zp_error  zp_offset\n"
        lines = header.format("# filter", width=width)
        line = "{:<{width}}  {:3d},{:3d}  {:>7}  {:8.4f}  {:9.4f}\n"
        # current implementation with fixed photometric zero-points
        zp_error, zp_offset = 0.01, 0.0
        system = "AB"
        # add a line for each filter (with column for magnitude and error)
        for idx, name in enumerate(self.filter_names):
            mag_col_idx = 2 * idx + 1
            err_col_idx = 2 * idx + 2
            lines += line.format(
                name, mag_col_idx, err_col_idx, self.config.system,
                zp_error, zp_offset, width=width)
        # add the prior magnitude column
        M_0_idx = self.filter_names.index(self.config.prior["filter"])
        lines += "{:<{width}}  {:7d}\n".format(
            "M_0", 2 * M_0_idx + 1, width=width)
        # write the file into the temporary directory
        with open(self._columns_file, "w") as f:
            f.write(lines)

    def _create_AB_files(self):
        dummy_args = [[20.0], [0.01]] * len(self.filter_names)
        self.execute(*dummy_args)

    @property
    def config(self):
        return self._config

    @property
    def path(self):
        return self._tempdir.name

    @property
    def tempdir(self):
        return self._tempdir.name

    @property
    def interpreter(self):
        if self.config.BPZenv == "python2":
            return self.config.BPZenv
        else:
            return os.path.join(self.config.BPZenv, "bin", "python2")

    @property
    def installed_templates(self):
        # find all files that match SED/*.list
        templates = {}
        dirname = os.path.join(self.path, "SED")
        for fname in os.listdir(dirname):
            if fname.endswith(".list"):
                key = fname.rstrip(".list")
                templates[key] = os.path.join(dirname, fname)
        return templates

    @property
    def installed_priors(self):
        # find all files that match prior_*.py
        priors = {}
        for fname in os.listdir(self.path):
            if fname.startswith("prior_") and fname.endswith(".py"):
                key = fname.lstrip("prior_").rstrip(".py")
                priors[key] = os.path.join(self.path, fname)
        return priors

    @property
    def installed_filters(self):
        return self._installed_filters

    @property
    def filter_names(self):
        return sorted(self._config.filters.keys())

    @property
    def colnames(self):
        return tuple(self._output_description.keys())

    @property
    def dtype(self):
        return self._output_dtype

    @property
    def descriptions(self):
        return self._output_description

    def _write_input(self, *mags_errs, threadID=None):
        n_filters = len(self.filter_names)
        if len(mags_errs) != 2 * n_filters:
            message = "expected {n:d} magnitude and {n:d} error columns"
            raise ValueError(message.format(n=n_filters))
        # format the lines
        line = "{:.5e} {:.5e} " * n_filters + "\n"
        lines = [line.format(*values) for values in zip(*mags_errs)]
        # write the lines
        threadID = "" if threadID is None else "_" + str(threadID)
        filename = self._input_template.format(threadID)
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    def _build_command(self, threadID, verbose=False):
        threadID = "" if threadID is None else "_" + str(threadID)
        # BPZ call
        command = [
            # select interpreter and executable
            self.interpreter, os.path.join(self.path, "bpz.py"),
            # in-/output
            os.path.join(
                self.tempdir, self._input_template.format(threadID)),
            "-COLUMNS", os.path.join(self.tempdir, self._columns_file),
            "-OUTPUT", os.path.join(
                self.tempdir, self._output_template.format(threadID)),
            # prior
            "-PRIOR", str(self.config.prior["name"]),
            # templates
            "-SPECTRA", "{:}.list".format(self.config.templates["name"]),
            "-INTERP", str(self.config.templates["interpolation"]),
            # likelihood
            "-ZMIN", str(self.config.likelihood["zmin"]),
            "-ZMAX", str(self.config.likelihood["zmax"]),
            "-DZ", str(self.config.likelihood["dz"]),
            "-ODDS", str(self.config.likelihood["odds"]),
            "-MIN_RMS", str(self.config.likelihood["min_rms"]),
            # disable extra stuff we don't need
            "-PROBS", "no", "-PROBS2", "no", "-PROBS_LITE", "no",
            "-NEW_AB", "no", "-CHECK", "no", "-INTERACTIVE", "no",
            "-VERBOSE", "yes" if verbose else "no"]
        return command

    def _run_BPZ(self, threadID, timeout=None, verbose=False):
        command = self._build_command(threadID, verbose)
        with subprocess.Popen(
                command, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE) as proc:
            try:
                stdout, stderr = proc.communicate(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            finally:
                if proc.returncode != 0 or verbose:
                    message = "########## BPZ start ##########\n"
                    message += "{:}\n\n{:}\n".format(
                        stdout.decode("utf-8").strip(),
                        stderr.decode("utf-8").strip())
                    message += "########### BPZ end ###########\n"
                    if proc.returncode == 0:
                        sys.stdout.write(message)
                    else:
                        sys.stderr.write(message)
                        raise subprocess.CalledProcessError(
                            proc.returncode, proc)

    def _read_output(self, threadID, get_ID, get_M_0):
        threadID = "" if threadID is None else "_" + str(threadID)
        outputfile = os.path.join(
            self.tempdir, self._output_template.format(threadID))
        data = np.loadtxt(outputfile, dtype=self._output_dtype)
        output_cols = []
        for col in self.colnames:
            if col == "ID" and not get_ID:
                continue
            if col == "M_0" and not get_M_0:
                continue
            output_cols.append(col)
        bpz_result = tuple(data[col] for col in output_cols)
        return bpz_result

    @Schedule.description("running BPZ")
    @Schedule.CPUbound
    def execute(
            self, *mags_errs, threadID=None, verbose=False,
            get_ID=False, get_M_0=True):
        self._write_input(*mags_errs, threadID=threadID)
        self._run_BPZ(threadID, verbose=verbose)
        result = self._read_output(threadID, get_ID, get_M_0)
        return result
