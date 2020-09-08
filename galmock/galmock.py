import functools
import inspect
import logging
import os
import sys
import warnings
from collections import OrderedDict

from galmock.core.bitmask import BitMaskManager
from galmock.core.config import TableParser
from galmock.core.datastore import DataStore, ModificationStamp
from galmock.core.readwrite import create_reader, create_writer
from galmock.core.utils import (ProgressBar, bytesize_with_prefix,
                                check_query_columns, sha1sum,
                                substitute_division_symbol)
from galmock.Flagship import find_central_galaxies, flux_to_magnitudes_wrapped
from galmock.matching import DataMatcher, MatcherParser
from galmock.MICE2 import evolution_correction_wrapped
from galmock.photometry import (PhotometryParser, apertures_wrapped,
                                find_percentile_wrapped,
                                magnification_correction_wrapped,
                                photometry_realisation_wrapped)
from galmock.photoz import BpzManager, BpzParser
from galmock.samples import DensitySampler, RedshiftSampler, SampleManager


def get_pseudo_sys_argv(func, args, kwargs):
    """inspect function name, parameters"""
    # get description of function parameters expected
    params = OrderedDict()
    argspec = inspect.getargspec(func)
    # go through each position based argument
    if argspec.args and type(argspec.args) is list:
        unnamed_idx = 0
        for i, arg in enumerate(args):
            try:
                params[argspec.args[i]] = arg
            except IndexError:
                if argspec.varargs:
                    key = "{:}[{:d}]".format(argspec.varargs, unnamed_idx)
                    params[key] = arg
                    unnamed_idx += 1
    # finally show the named varargs
    if kwargs:
        params.update(kwargs)
    # create a sys.argv style list
    try:
        classinst = params.pop("self")
        sys_argv = [".".join([classinst.__class__.__name__, func.__name__])]
    except KeyError:
        raise TypeError("function must be class method")
    for key, val in params.items():
        sys_argv.append("{:}={:}".format(key, str(val)))
    return sys_argv, classinst


def job(method):

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        sys_argv, classinst = get_pseudo_sys_argv(method, args, kwargs)
        #print(classinst)
        restore_logger = classinst.logger
        # reset the timestamp handler
        classinst.datastore._timestamp = ModificationStamp(sys_argv)
        # create a temorary logger for the class method
        classinst.logger = logging.getLogger(".".join([__name__, sys_argv[0]]))
        classinst.logger.setLevel(logging.DEBUG)
        classinst.logger.info("initialising job: {:}".format(
            sys_argv[0].split(".")[-1]))
        # calls original function
        res = method(*args, **kwargs)
        classinst.logger.info(
            "computation completed for {:,d} entries".format(len(classinst)))
        # add the check sums
        classinst.logger.debug("computing checksums and updating attributes")
        classinst.datastore._timestamp.add_checksums()
        classinst.datastore._timestamp.finalize()
        # restore the original logger
        classinst.logger = restore_logger
        return res

    return wrapper


class GalaxyMock(object):

    def __init__(self, datastore, readonly=True, threads=-1):
        self.datastore = DataStore.open(datastore, readonly=readonly)
        self.datastore.pool.max_threads = threads
        self.logger = logging.getLogger(
            ".".join([__name__, self.__class__.__name__]))
        self.logger.setLevel(logging.DEBUG)

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self.datastore.close()

    def __len__(self):
        return len(self.datastore)

    @property
    def filepath(self):
        return self.datastore.root

    @property
    def filesize(self):
        return self.datastore._filesize

    def show_metadata(self):
        print("==> META DATA")
        n_cols, n_rows = self.datastore.shape
        print("root:     {:}".format(self.datastore.root))
        print("size:     {:}".format(self.datastore.filesize))
        print("shape:    {:,d} rows x {:d} columns".format(n_rows, n_cols))

    def show_columns(self):
        header = "==> COLUMN NAME"
        width_cols = max(len(header), max(
            len(colname) for colname in self.datastore.colnames))
        print("\n{:}    {:}".format(header.ljust(width_cols), "TYPE"))
        for name in self.datastore.colnames:
            colname_padded = name.ljust(width_cols)
            line = "{:}    {:}".format(
                colname_padded, str(self.datastore[name].dtype))
            print(line)

    def show_attributes(self):
        print("\n==> ATTRIBUTES")
        for name in self.datastore.colnames:
            print()
            # print the column name indented and then a tree-like listing
            # of the attributes (drawing connecting lines for better
            # visibitilty)
            print("{:}".format(name))
            attrs = self.datastore[name].attr
            # all attributes from the pipeline should be dictionaries
            if type(attrs) is dict:
                i_last = len(attrs)
                width_key = max(len(key) + 2 for key in attrs)
                for i, key in enumerate(sorted(attrs), 1):
                    print_key = key + " :"
                    line = "{:}{:} {:}".format(
                        " └╴ " if i == i_last else " ├╴ ",
                        print_key.ljust(width_key), str(attrs[key]))
                    print(line)
            # fallback
            else:
                print("     └╴ {:}".format(str(attrs)))

    def show_history(self):
        print("\n==> HISTORY")
        date_width = 24
        for date, call in self.datastore.get_history().items():
            print("{:} : {:}".format(date.ljust(date_width), call))

    def show_logs(self):
        print("\n==> LOGS")
        logpath = self.datastore.root + ".log"
        if not os.path.exists(logpath):
            raise OSError("log file not found: {:}".format(logpath))
        with open(logpath) as f:
            for line in f.readlines():
                print(line.strip())

    @classmethod
    def create(
            cls, datastore, input, format=None, fits_ext=1, columns=None,
            purge=False, threads=-1):
        jobname = inspect.stack()[1][3]
        logger = logging.getLogger(".".join([__name__, jobname]))
        logger.setLevel(logging.DEBUG)
        logger.info("initialising job: {:}".format(jobname))
        # check the columns file
        if columns is not None:
            configuration = TableParser(columns)
            col_map_dict = configuration.column_map
        else:
            col_map_dict = None
        # read the data file and write it to the memmory mapped data store
        with create_reader(input, format, col_map_dict, fits_ext) as reader:
            with DataStore.create(datastore, len(reader), purge) as ds:
                ds._timestamp = ModificationStamp([
                    "GalaxyMock.create",
                    "datastore={:}".format(datastore),
                    "input={:}".format(input), "format={:}".format(format),
                    "fits_ext={:}".format(fits_ext),
                    "columns={:}".format(columns),
                    "datastore={:}".format(datastore)])
                # create the new data columns
                logger.debug(
                    "registering {:,d} new columns".format(len(col_map_dict)))
                for path, (colname, dtype) in col_map_dict.items():
                    try:
                        if dtype is None:
                            dtype = reader.dtype[colname].str
                        ds.add_column(
                            path, dtype=dtype, attr={
                                "source file": input, "source name": colname})
                    except KeyError as e:
                        logger.exception(str(e))
                        raise
                # copy the data
                logger.info("converting input data ...")
                pbar_kwargs = {
                    "leave": False, "unit_scale": True, "unit": "row",
                    "dynamic_ncols": True}
                if hasattr(reader, "_guess_length"):
                    pbar = ProgressBar()
                else:
                    pbar = ProgressBar(n_rows=len(reader))
                # read the data in chunks and copy them to the data store
                start = 0
                for chunk in reader:
                    end = reader.current_row  # index where reading continues
                    # map column by column onto the output table
                    for path, (colname, dtype) in col_map_dict.items():
                        # the CSV reader provides a conservative estimate of
                        # the required number of rows so we need to keep
                        # allocating more memory if necessary
                        if end > len(ds):
                            ds.expand(len(chunk))
                        ds[path][start:end] = chunk[colname]
                    # update the current row index
                    pbar.update(end - start)
                    start = end
                pbar.close()
                # if using the CSV reader: truncate any allocated, unused rows
                if len(ds) > end:
                    ds.resize(end)
                # print a preview of the table as quick check
                ds.show_preview()
                message = "finalized data store with {:,d} rows ({:})"
                logger.info(message.format(end, ds.filesize))
        # create and return the new GalaxyMock instance
        instance = cls(datastore, readonly=False, threads=threads)
        return instance

    def info(self, columns=False, attr=False, history=False, logs=False):
        self.show_metadata()
        if columns:
            self.show_columns()
        if attr:
            self.show_attributes()
        if history:
            self.show_history()
        if logs:
            self.show_logs()

    def verify(self):
        self.show_metadata()
        # verify the check sums
        header = "==> COLUMN NAME"
        width_cols = max(len(header), max(
            len(colname) for colname in self.datastore.colnames))
        print("\n{:}    {:}  {:}".format(
            header.ljust(width_cols), "STATUS ", "HASH"))
        # compute and verify the store checksums column by column
        n_good, n_warn, n_error = 0, 0, 0
        line = "{:<{width}s}    {:<7s}  {:s}"
        for name in self.datastore.colnames:
            column = self.datastore[name]
            try:
                checksum = column.attr["SHA-1 checksum"]
                assert(checksum == sha1sum(column.filename))
                n_good += 1
            except KeyError:
                print(line.format(
                    name, "WARNING", "no checksum provided", width=width_cols))
                n_warn += 1
            except AssertionError:
                print(line.format(
                    name, "ERROR", "checksums do not match", width=width_cols))
                n_error += 1
            else:
                print(line.format(name, "OK", checksum, width=width_cols))
        # do a final report
        if n_good == len(self.datastore.colnames):
            print("\nAll columns passed")
        else:
            print("\nPassed:   {:d}\nWarnings: {:d}\nErrors:   {:d}".format(
                n_good, n_warn, n_error))

    @job
    def query(
            self, output, query=None, verify=False, format=False, columns=None,
            compression=None, hdf5_shuffle=False, hdf5_checksum=False):
        to_stdout = output is None
        # parse the math expression
        selection_columns, expression = substitute_division_symbol(
            query, self.datastore)
        if selection_columns is not None:
            selection_table = self.datastore[sorted(selection_columns)]
        # check the requested columns
        request_table, dtype = check_query_columns(columns, self.datastore)
        # verify the data if requested
        if verify:
            self.logger.info("verifying SHA-1 checksums")
            if columns is None:
                verification_columns = self.datastore.colnames
            else:
                verification_columns = [col for col in columns]
                if selection_columns is not None:
                    verification_columns.extend(selection_columns)
            name_width = max(len(name) for name in verification_columns)
            for i, name in enumerate(verification_columns, 1):
                sys.stdout.write("checking {:3d} / {:3d}: {:}\r".format(
                    i, len(verification_columns), name.ljust(name_width)))
                sys.stdout.flush()
                self.datastore.verify_column(name)
        # select data and write them to the output file
        with create_writer(
                output, format, dtype, compression,
                hdf5_shuffle, hdf5_checksum) as writer:
            # query the table and write to the output fie
            if expression is not None:
                message = "filtering data and writing output file ..."
            else:
                message = "converting data and writing output file ..."
            self.logger.info(message)
            if not to_stdout:
                pbar = ProgressBar(len(self.datastore))
            n_select = 0
            buffersize = writer.buffersize // dtype.itemsize
            for start, end in self.datastore.row_iter(buffersize):
                # apply optional selection, read only minimal amount of data
                if expression is not None:
                    chunk = selection_table[start:end]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            mask = expression(chunk)
                    except KeyError as e:
                        message = "unknown column '{:}', check ".format(
                            e.args[0])
                        message += "the query expression for spelling "
                        message += "mistakes or undefined columns"
                        self.logger.exception(message)
                        raise KeyError(message)
                    selection = request_table[start:end][mask]
                # read all entries in the range without applying the selection
                else:
                    selection = request_table[start:end]
                # write to target
                writer.write_chunk(selection.to_records())
                n_select += len(selection)
                if not to_stdout:
                    pbar.update(end - start)
            if not to_stdout:
                pbar.close()
        message = "wrote {:,d} matching entries ({:})".format(
            n_select, bytesize_with_prefix(writer.filesize))
        self.logger.info(message)

    @job
    def prepare_MICE2(self, mag, evo):
        # apply the evolution correction to the model magnitudes
        self.datastore.pool.set_worker(evolution_correction_wrapped)
        # find redshift column
        z_path = "position/z/true"
        self.datastore.require_column(z_path, "true redshift")
        self.datastore.pool.add_argument_column(z_path)
        # find all magnitude columns
        try:
            model_mags, _ = self.datastore.load_photometry(mag)
        except KeyError as e:
            self.logger.exception(str(e))
            raise
        # create the output columns
        for key, mag_path in model_mags.items():
            evo_path = os.path.join(evo, key)
            # create new output columns
            self.datastore.add_column(
                evo_path, dtype=self.datastore[mag_path].dtype.str,
                attr={
                    "description":
                    "{:} with evolution correction applied".format(mag_path)},
                overwrite=True)
            # add columns to call signature
            self.datastore.pool.add_argument_column(mag_path)
            self.datastore.pool.add_result_column(evo_path)
        # compute and store the corrected magnitudes in parallel
        self.datastore.pool.execute()

    @job
    def prepare_Flagship(
            self, flux, mag, gal_idx=None, is_central=None):
        # convert model fluxes to model magnitudes
        self.datastore.pool.set_worker(flux_to_magnitudes_wrapped)
        # find all flux columns
        try:
            model_fluxes, _ = self.datastore.load_photometry(flux)
        except KeyError as e:
            self.logger.exception(str(e))
            raise
        # create the output columns
        for key, flux_path in model_fluxes.items():
            mag_path = os.path.join(mag, key)
            # create new output columns
            self.datastore.add_column(
                mag_path, dtype=self.datastore[flux_path].dtype.str,
                attr={
                    "description":
                    "{:} converted to AB magnitudes".format(flux_path)},
                overwrite=True)
            # add columns to call signature
            self.datastore.pool.add_argument_column(flux_path)
            self.datastore.pool.add_result_column(mag_path)
        # compute and store the corrected magnitudes in parallel
        self.datastore.pool.execute()
        # add the central galaxy flag
        self.datastore.pool.set_worker(find_central_galaxies)
        # find the input column
        self.datastore.require_column(gal_idx, "galaxy index")
        self.datastore.pool.add_argument_column(gal_idx)
        # create the output column
        self.datastore.add_column(
            is_central, dtype="bool", overwrite=True, attr={
                "description": "host central galaxy flag"})
        self.datastore.pool.add_result_column(is_central)
        # compute and store the corrected magnitudes in parallel
        self.datastore.pool.execute()

    @job
    def magnification(self, mag, lensed):
        # apply the magnification correction to the model magnitudes
        self.datastore.pool.set_worker(magnification_correction_wrapped)
        # find convergence column
        kappa_path = "lensing/kappa"
        self.datastore.require_column(kappa_path, "convergence")
        self.datastore.pool.add_argument_column(kappa_path)
        # find all magnitude columns
        try:
            input_mags, _ = self.datastore.load_photometry(mag)
        except KeyError as e:
            self.logger.exception(str(e))
            raise
        # create the output columns
        for key, mag_path in input_mags.items():
            lensed_path = os.path.join(lensed, key)
            # create new output columns
            self.datastore.add_column(
                lensed_path, dtype=self.datastore[mag_path].dtype.str,
                attr={
                    "description":
                    "{:} with magnification correction applied".format(
                        lensed_path)},
                overwrite=True)
            # add columns to call signature
            self.datastore.pool.add_argument_column(mag_path)
            self.datastore.pool.add_result_column(lensed_path)
        # compute and store the corrected magnitudes
        self.datastore.pool.execute()

    @job
    def effective_radius(self, config):
        # check the configuration file
        configuration = PhotometryParser(config)
        # apply the magnification correction to the model magnitudes
        self.datastore.pool.set_worker(find_percentile_wrapped)
        self.datastore.pool.add_argument_constant(
            configuration.intrinsic["flux_frac"])
        # find disk and bulge component columns
        input_columns = (
            ("disk size", "shape/disk/size"),
            ("bulge size", "shape/bulge/size"),
            ("bulge fraction", "shape/bulge/fraction"))
        for col_desc, path in input_columns:
            self.datastore.require_column(path, col_desc)
            self.datastore.pool.add_argument_column(path)
        # create the output column
        self.datastore.add_column(
            configuration.intrinsic["r_effective"], dtype="f4", attr={
                "description":
                "effective radius (emitting {:.1%} of the flux)".format(
                    configuration.intrinsic["flux_frac"])},
            overwrite=True)
        # add column to call signature
        self.datastore.pool.add_result_column(
            configuration.intrinsic["r_effective"])
        # compute and store the corrected magnitudes
        self.datastore.pool.execute()

    @job
    def apertures(self, config):
        # check the configuration file
        configuration = PhotometryParser(config)
        # apply the magnification correction to the model magnitudes
        # initialize the aperture computation
        self.datastore.pool.set_worker(apertures_wrapped)
        self.datastore.pool.add_argument_constant(configuration.method)
        self.datastore.pool.add_argument_constant(configuration)
        # find effective radius and b/a ratio columns
        input_columns = (
            ("effective radius", "shape/R_effective"),
            ("b/a ratio", "shape/axis_ratio"))
        for col_desc, path in input_columns:
            self.datastore.require_column(path, col_desc)
            self.datastore.pool.add_argument_column(path)
        output_columns = (  # for each filter three output columns are required
            ("apertures/{:}/major_axis/{:}",
                "{:} aperture major axis (PSF size: {:.2f}\")"),
            ("apertures/{:}/minor_axis/{:}",
                "{:} aperture minor axis (PSF size: {:.2f}\")"),
            ("apertures/{:}/snr_correction/{:}",
                "{:} aperture S/N correction (PSF size: {:.2f}\")"))
        # make the output columns for each filter
        for key in configuration.filter_names:
            for out_path, desc in output_columns:
                formatted_path = out_path.format(
                    configuration.aperture_name, key)
                self.datastore.add_column(
                    formatted_path, dtype="f4", overwrite=True, attr={
                        "description":
                        desc.format(
                            configuration.method, configuration.PSF[key])})
                # collect all new columns as output targets
                self.datastore.pool.add_result_column(formatted_path)
        # compute and store the apertures
        self.datastore.pool.execute()

    @job
    def photometry(self, mag, real, config, seed="sapling"):
        # check the configuration file
        configuration = PhotometryParser(config)
        # apply the magnification correction to the model magnitudes
        # find all magnitude columns
        try:
            input_mags, _ = self.datastore.load_photometry(mag)
        except KeyError as e:
            self.logger.exception(str(e))
            raise
        # select the required magnitude columns
        available = set(input_mags.keys())
        missing = set(configuration.filter_names) - available
        if len(missing) > 0:
            message = "requested filters not found: {:}".format(
                ", ".join(missing))
            self.logger.error(message)
            raise KeyError(message)
        # initialize the photometry generation
        self.datastore.pool.set_worker(photometry_realisation_wrapped)
        self.datastore.pool.add_argument_constant(configuration)
        # collect the filter-specific arguments
        for key in configuration.filter_names:
            # 1) magnitude column
            mag_path = input_mags[key]
            self.datastore.require_column(mag_path, "{:}-band".format(key))
            self.datastore.pool.add_argument_column(mag_path)
            # 2) magnitude limit
            self.datastore.pool.add_argument_constant(
                configuration.limits[key])
            # 3) S/N correction factors
            if configuration.photometry["apply_apertures"]:
                snr_path = "apertures/{:}/snr_correction/{:}".format(
                    configuration.aperture_name, key)
                self.datastore.require_column(
                    mag_path, "{:}-band S/N correction".format(key))
                self.datastore.pool.add_argument_column(snr_path)
            else:  # disable aperture correction
                self.datastore.pool.add_argument_constant(1.0)
        output_columns = (  # for each filter three output columns are required
            ("{:}/{:}",
            "{:} photometry realisation (from {:}, limit: {:.2f} mag)"),
            ("{:}/{:}_err",
            "{:} photometric error (from {:}, limit: {:.2f} mag)"))
        # make the output columns for each filter
        for key in configuration.filter_names:
            for out_path, desc in output_columns:
                self.datastore.add_column(
                    out_path.format(real, key),
                    dtype=self.datastore[mag_path].dtype.str, attr={
                        "description": desc.format(
                            configuration.method, mag_path,
                            configuration.limits[key])},
                    overwrite=True)
                self.datastore.pool.add_result_column(
                    out_path.format(real, key))
        # compute and store the corrected magnitudes
        self.datastore.pool.execute(seed=seed)

    @job
    def match_data(self, config):
        # check the configuration file
        configuration = MatcherParser(config)
        # apply the magnification correction to the model magnitudes
        with DataMatcher(configuration) as matcher:
            self.datastore.pool.set_worker(matcher.apply)
            # increase the default chunksize, larger chunks will be marginally
            # faster but the progress update will be infrequent
            self.datastore.pool.chunksize = \
                self.datastore.pool.max_threads * self.datastore.pool.chunksize
            # select the required feature columns
            for feature_path in configuration.features.values():
                self.datastore.require_column(feature_path, "feature")
                self.datastore.pool.add_argument_column(
                    feature_path, keyword=feature_path)
            # check that the data types are compatible
            try:
                matcher.check_features(self.datastore)
            except Exception as e:
                self.logger.exception(str(e))
                raise
            # make the output columns for each observable
            for output_path, dtype, attr in matcher.observable_information():
                self.datastore.add_column(
                    output_path, dtype=dtype, attr=attr, overwrite=True)
                self.datastore.pool.add_result_column(output_path)
            matcher.build_tree()
            # compute and store the corrected magnitudes
            self.datastore.pool.execute()  # cKDTree releases GIL

    @job
    def BPZ(self, mag, zphot, config):
        # check the configuration file
        configuration = BpzParser(config)
        # run BPZ on the selected magnitudes
        # find all magnitude columns
        try:
            input_mags, input_errs = self.datastore.load_photometry(mag)
        except KeyError as e:
            self.logger.exception(str(e))
            raise
        # launch the BPZ manager
        with BpzManager(configuration) as bpz:
            self.datastore.pool.set_worker(bpz.execute)
            # add the magnitude and error columns to call signature
            for key in bpz.filter_names:
                try:
                    mag_key, err_key = input_mags[key], input_errs[key]
                    self.datastore.pool.add_argument_column(mag_key)
                    self.datastore.pool.add_argument_column(err_key)
                except KeyError as e:
                    self.logger.exception(str(e))
                    raise
            # create the output columns
            for key, desc in bpz.descriptions.items():
                zphot_path = os.path.join(zphot, key)
                # create new output columns
                self.datastore.add_column(
                    zphot_path, dtype=bpz.dtype[key].str, overwrite=True,
                    attr={"description": desc})
                # add columns to call signature
                self.datastore.pool.add_result_column(zphot_path)
            self.datastore.pool.parse_thread_id = True
            self.datastore.pool.execute()
            self.datastore.pool.parse_thread_id = False

    @job
    def select_sample(
            self, config, sample, area, type="reference", seed="sapling"):
        # check the configuration file
        Parser = SampleManager.get_parser(type, sample)
        configuration = Parser(config)
        # apply the magnification correction to the model magnitudes
        self.logger.info("apply selection funcion: {:}".format(sample))
        # allow the worker threads to modify the bitmask column directly
        self.datastore.pool.allow_modify = True
        # make the output column
        BMM = BitMaskManager(sample)
        bitmask = self.datastore.add_column(
            configuration.bitmask, dtype=BMM.dtype, overwrite=True)
        # initialize the selection bit (bit 1) to true, all subsequent
        # selections will be joined with AND
        self.logger.debug("initializing bit mask")
        bitmask[:] = 1
        # start with photometric selection
        selector_class = SampleManager.get_selector(type, sample)
        selector = selector_class(BMM)
        self.datastore.pool.set_worker(selector.apply)
        # select the columns needed for the selection function
        self.datastore.pool.add_argument_column(configuration.bitmask)
        for name, path in configuration.selection.items():
            self.datastore.require_column(path)
            self.datastore.pool.add_argument_column(path, keyword=name)
        # apply selection
        self.datastore.pool.execute(seed=seed, prefix="photometric selection")
        # optional density sampling
        sampler_class = SampleManager.get_sampler(type, sample)
        if sampler_class is not NotImplemented:
            try:
                # surface density
                if issubclass(sampler_class, DensitySampler):
                    self.logger.info("estimating surface density ...")
                    sampler = sampler_class(BMM, area, bitmask)
                # redshift weighted density
                else:
                    self.logger.info("estimating redshift density ...")
                    sampler = sampler_class(
                        BMM, area, bitmask,
                        self.datastore[configuration.selection["redshift"]])
                self.datastore.pool.set_worker(sampler.apply)
                # select the columns needed for the selection function
                self.datastore.pool.add_argument_column(configuration.bitmask)
                # redshift weighted density requires a mock n(z) estimate
                if isinstance(sampler, RedshiftSampler):
                    self.datastore.pool.add_argument_column(
                        configuration.selection["redshift"],
                        keyword="redshift")
                # apply selection
                self.datastore.pool.execute(
                    seed=seed, prefix="density sampling")
            except ValueError as e:
                if str(e).startswith("sample density must be"):
                    message = "skipping sampling due to low mock density"
                    self.logger.warning(message)
                else:
                    raise e
        # add the description attribute to the output column
        bitmask.attr = {"description": BMM.description}
        # show final statistics
        N_mock = DensitySampler.count_selected(bitmask)
        density = "{:.3f} / arcmin²".format(N_mock / area / 3600.0)
        self.logger.info("density of selected objects: " + density)
