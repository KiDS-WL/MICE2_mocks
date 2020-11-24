#
# This is the main module of galmock that implements the mock data processing
# methods. These are exposed as methods of the GalaxyMock class and by stand-
# alone scripts, which manage the internal mock data storage system.s
#
# Each method should be wrapped with the @job decorator that manages the
# internal logging system, time-stamps and data check-sums.
#

import functools
import inspect
import logging
import os
import sys
import warnings
from collections import OrderedDict

import numpy as np
import toml

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
    """
    Inspects the parameters a called class method to create an sys.argv style
    listing of parameters and values.

    Parameters:
    -----------
    func : object
        Callable to inspect.
    args : list
        List of function parameters used to call func.
    kwargs : dict
        Named variable function parameters used to call func.

    Returns:
    --------
    sys_argv : list
        List containing the class and method names in "." syntax and a list of
        the parameter values and keyword argument - value pairs.
    classinst : object
        The class instance which owns the called method.
    """
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
    # add the named varargs
    if kwargs:
        params.update(kwargs)
    # fill up with arguments that have default values
    if argspec.defaults:
        n_defaults = len(argspec.defaults)
        for arg, default in zip(argspec.args[-n_defaults:], argspec.defaults):
            if arg not in params:
                params[arg] = default
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
    """
    Decorator for galmock pipeline processing steps implemented in GalaxyMock.
    Manages the ModificationStamp instance of the data store to add time
    stamps, information of the method call and data check sums.
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        sys_argv, classinst = get_pseudo_sys_argv(method, args, kwargs)
        restore_logger = classinst.logger
        # reset the timestamp handler
        classinst.datastore._timestamp = ModificationStamp(sys_argv)
        # create a temorary logger for the class method
        classinst.logger = logging.getLogger(
            ".".join([__name__.split(".")[0], sys_argv[0]]))
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
    """
    Wrapper and managaer for the galmock.core.datastore.DataStore that
    implements all tasks of the pipeline and hiding the complexity of the data
    management. Implements additional methods to display information about the
    data store status, such as (file-)size, columns names and attributes, the
    history of class methods applied to the data and a summary of the log data
    produced by each processing step.

    To create a new, empty data store, see GalaxyMock.create().

    Parameters:
    -----------
    datastore : str
        Path to the data store to open. The data store is a directory that
        contains nothing but a collection column data, for details see
        mmaptable.MmapTable.
    readonly : bool
        Whether to allow write acces to the data store.
    threads : int
        The maximum number of threads or parallel processes to run in parallel
        on the data. All by default.
    """

    def __init__(self, datastore, readonly=True, threads=-1):
        self.datastore = DataStore.open(datastore, readonly=readonly)
        self.datastore.pool.max_threads = threads
        self.logger = logging.getLogger(
            ".".join([__name__.split(".")[0], self.__class__.__name__]))
        self.logger.setLevel(logging.DEBUG)

    @classmethod
    def create(
            cls, datastore, input, format=None, fits_ext=1, columns=None,
            index=None, purge=False, threads=-1):
        """
        Create a new galaxy mock instance and data store from input simulation
        data (such as MICE2, Flagship, ...).

        Parameters:
        -----------
        datastore : str
            Path at which the data store directory is created.
        input : str
            Path of the input file containing the raw simulation data.
        format : str
            Format descibing string, see galmock.core.readwrite for all
            supported file formats.
        fits_ext : int
            If the input is in FITS format, read data from this table
            extension.
        columns : str
            A column mapping configuration file in TOML format, see
            galmock.core.config.TableParser that maps the column name of the
            input file to paths in the data store (optional).
        index : str
            Path at which a range index is created (default: none added),
            useful as unique identifer when creating subsets.
        purge : bool
            Handle with care! Erases the data store directory if it exists.
        threads : int
            The maximum number of threads or parallel processes to run in
            parallel on the data. All by default.
        """
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
                # add the range index
                if index is not None:
                    idx_col = ds.add_column(
                        index, dtype="i8", attr={"description": "range index"})
                    for start, end in ds.row_iter():
                        idx_col[start:end] = np.arange(start, end, dtype="i8")
                # print a preview of the table as quick check
                ds.show_preview()
                message = "finalized data store with {:,d} rows ({:})"
                logger.info(message.format(end, ds.filesize))
        # create and return the new GalaxyMock instance
        instance = cls(datastore, readonly=False, threads=threads)
        return instance

    def close(self):
        """
        Close the data store and the underlying file pointers.
        """
        self.datastore.close()

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def __len__(self):
        return len(self.datastore)

    def __contains__(self, key):
        return key in self.datastore

    @property
    def filepath(self):
        """
        Return the root file path of the data store.
        """
        return self.datastore.root

    @property
    def filesize(self):
        """
        Return the total size of the data on disk (excluding the attributes).
        """
        return self.datastore._filesize

    def show_metadata(self):
        """
        Print the meta data of the table: file path, size on disk and number of
        rows and columns.
        """
        print("==> META DATA")
        n_cols, n_rows = self.datastore.shape
        print("root:     {:}".format(self.datastore.root))
        print("size:     {:}".format(self.datastore.filesize))
        print("shape:    {:,d} rows x {:d} columns".format(n_rows, n_cols))

    def show_columns(self):
        """
        Generate and print a listing of all columns with their respective data
        type.
        """
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
        """
        Generate and print a listing of all columns with their dictonary-style
        attributes listed (such as creation time stamp and check sums).
        """
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
        """
        Generate and print a list of the of class methods that were applied to
        the data store, ordered by time. Only shows the latest call if a method
        was called multiple times.
        """
        print("\n==> HISTORY")
        date_width = 24
        for date, call in self.datastore.get_history().items():
            print("{:} : {:}".format(date.ljust(date_width), call))

    def show_logs(self):
        """
        Print the content of the data stores log file.
        """
        print("\n==> LOGS")
        logpath = self.datastore.root + ".log"
        if not os.path.exists(logpath):
            raise OSError("log file not found: {:}".format(logpath))
        with open(logpath) as f:
            for line in f.readlines():
                print(line.strip())

    @job
    def ingest_column(
            self, input, format=None, fits_ext=1, column=None,
            path=None, description=None, overwrite=True):
        """
        Ingest a single data column from an external data source. The length
        of the data must match the number of entries in the data store and is
        not checked in advance.

        Parameters:
        -----------
        input : str
            Path of the input file containing the data column.
        format : str
            Format descibing string, see galmock.core.readwrite for all
            supported file formats.
        fits_ext : int
            If the input is in FITS format, read data from this table
            extension.
        column : str
            Name of the data column in the input file. Usefull if the file
            contains more than one column.
        path : str
            Path to store the new column at in the data store. If none is
            provided the column name is used.
        description : str
            Descriptive text to store as column attribute. By default stores
            the input file path and column name.
        overwrite : bool
            Whether to overwrite the target column if it exists.
        """
        # read the data file and write it to the memmory mapped data store
        with create_reader(input, format, None, fits_ext) as reader:
            # check if the input column
            if column is None:
                if len(reader.colnames) != 1:
                    message = "input file contains multiple columns and "
                    message += "column is not specified"
                    self.logger.error(message)
                    raise ValueError(message)
                else:
                    column = reader.colnames[0]
            elif column not in reader.colnames:
                message = "requested column not found in input "
                message += "file: {:}".format(column)
                self.logger.error(message)
                raise KeyError(message)
            # create the new data columns
            if path is None:
                path = column
            self.logger.info("adding new column: {:}".format(path))
            if description is None:
                attr = {"source file": input, "source name": column}
            else:
                attr = {"description": description}
            try:
                dtype = reader.dtype[column].str
                self.datastore.add_column(
                    path, dtype=dtype, attr=attr, overwrite=overwrite)
                # copy the data
                self.logger.info("converting input data ...")
                pbar_kwargs = {
                    "leave": False, "unit_scale": True, "unit": "row",
                    "dynamic_ncols": True}
                if hasattr(reader, "_guess_length"):
                    pbar = ProgressBar()
                else:
                    pbar = ProgressBar(n_rows=len(reader))
                # read the data in chunks and copy it to the column
                start = 0
                for chunk in reader:
                    end = reader.current_row  # index where reading continues
                    if end > len(self):
                        message = "input data exceeds column length"
                        self.logger.error(message)
                        raise ValueError(message)
                    # insert into data store column
                    self.datastore[path][start:end] = chunk[column]
                    # update the current row index
                    pbar.update(end - start)
                    start = end
                if end != len(self):
                    message = "ran out of input data while filling column"
                    raise ValueError(message)
                pbar.close()
                # print a preview of the ingested column as quick check
                self.datastore.show_preview(columns=[path])
            except Exception as e:
                self.logger.exception(str(e).strip("\""))
                # delete the new column if the data was not copied successfully
                try:
                    self.datastore.drop_column(path)
                except Exception:
                    pass
                raise

    @job
    def add_column(
            self, path, dtype="f8", description=None, fill_value=None,
            overwrite=False):
        """
        Create a new column and initialize it with an optional fill value.

        Parameters:
        -----------
        path : str
            Path of the new column in the data store.
        dtyle : str
            String describing a numpy data type, e.g. 'f8' -> 64bit float, 'i4'
            -> 32bit int, 'bool' -> True/False boolean.
        description : str
            Descriptive text that is stored in the column attributes.
        fill_value : any
            Optional fill value to initialize the column, must parse to the
            specified data type. Can be NaN or Inf for floating point types.
        overwrite : bool
            Whether to overwrite the target column if it exists.
        """
        if path in self.datastore and not overwrite:
            message = "column already exists: {:}".format(path)
            self.logger.error(message)
            raise KeyError(message)
        # establish the data type and check the fill value
        try:
            dtype = np.dtype(dtype)
            message = "initializing as {:}".format(dtype.__str__().upper())
            if fill_value is not None:
                fill_value = dtype.type(fill_value)
                message += " with value '{:}'".format(fill_value)
        except TypeError as e:
            self.logger.exception(str(e))
            raise
        except ValueError as e:
            self.logger.exception(str(e))
            raise
        # create the column
        self.logger.debug(message)
        column = self.datastore.add_column(
            path, dtype=dtype, attr={"description": description},
            overwrite=overwrite)
        if fill_value is not None:
            column[:] = fill_value

    def drop_column(self, columns):
        """
        Delete a set of columns from the data store.

        Parameters:
        -----------
        columns : list of str
            Paths of the columns to delete from the data store.
        """
        for column in columns:
            self.datastore.drop_column(column)

    def info(self, columns=False, attr=False, history=False, logs=False):
        """
        Print some information about the data store, such as meta data, column
        names, attributes, history and the log file content.

        Parameters:
        -----------
        columns : bool
            Whether to print the listing of column names and data types.
        attr : bool
            Whether to list the column attributes.
        history : bool
            Whether to print the processing history.
        logs : bool
            Whether to print the log file content.
        """
        self.show_metadata()
        if columns:
            self.show_columns()
        if attr:
            self.show_attributes()
        if history:
            self.show_history()
        if logs:
            self.show_logs()

    def verify(self, recalc=False):
        """
        Compute and verify the SHA-1 check sums for each data column with the
        reference value stored in the column attributes.

        Parameters:
        -----------
        recalc : bool
            Recalculate all check sums
        """
        self.show_metadata()
        # verify the check sums
        if recalc:
            print("\nRecomputing all check sums")
        header = "==> COLUMN NAME"
        width_cols = max(len(header), max(
            len(colname) for colname in self.datastore.colnames))
        # compute and verify the store checksums column by column
        n_good, n_warn, n_error = 0, 0, 0
        if recalc:
            line = "{:<{width}s}    {:s}"
            print("\n{:}    {:}".format(
                header.ljust(width_cols), "HASH"))
        else:
            line = "{:<{width}s}    {:<7s}  {:s}"
            print("\n{:}    {:}  {:}".format(
                header.ljust(width_cols), "STATUS ", "HASH"))
        for name in self.datastore.colnames:
            column = self.datastore[name]
            if recalc:
                checksum = sha1sum(column.filename)
                attr = column.attr
                attr["SHA-1 checksum"] = checksum
                column.attr = attr
                print(line.format(name, checksum, width=width_cols))
            else:
                try:
                    checksum = column.attr["SHA-1 checksum"]
                    assert(checksum == sha1sum(column.filename))
                    n_good += 1
                except KeyError:
                    print(line.format(
                        name, "WARNING", "no checksum provided",
                        width=width_cols))
                    n_warn += 1
                except AssertionError:
                    print(line.format(
                        name, "ERROR", "checksums do not match",
                        width=width_cols))
                    n_error += 1
                else:
                    print(line.format(name, "OK", checksum, width=width_cols))
        # do a final report
        if not recalc:
            if n_good == len(self.datastore.colnames):
                print("\nAll columns passed")
            else:
                print(
                    "\nPassed:   {:d}\nWarnings: {:d}\nErrors:   {:d}".format(
                        n_good, n_warn, n_error))

    @job
    def merge(
            self, lindex, input, rindex, format=None, fits_ext=1, columns=None,
            overwrite=True):
        """
        Merge data from an external file onto the data store using a unique
        identifier. This identifier must be sorted numerically increasing.

        Parameters:
        -----------
        lindex : str
            Path to the unique identifier column in the data store.
        input : str
            Path of the input file containing the data.
        rindex : str
            Name of the unique identifier column in the input file.
        format : str
            Format descibing string, see galmock.core.readwrite for all
            supported file formats.
        fits_ext : int
            If the input is in FITS format, read data from this table
            extension.
        columns : str
            A plain TOML file in which each line maps a column in the data
            store to a column in the data file (optional), e.g.:
            "position/ra/obs" = "RA"
        overwrite : bool
            Whether to overwrite target columns if they exists.
        """
        # read the column mapping
        if columns is not None:
            try:
                with open(columns) as f:
                    col_map_dict = toml.load(f)
            except OSError as e:
                self.logger.exception("configuration file not found")
                raise
            except Exception as e:
                self.logger.exception("malformed configuration file")
                raise
            for col in col_map_dict.keys():
                if col not in self.datastore:
                    message = "data store does not contain column: {:}"
                    self.logger.error(message.format(col))
                    raise KeyError(message.format(col))
        else:
            col_map_dict = None
        # read the input file and collect the indices
        self.logger.debug("reading external index")
        ext_idx = []
        with create_reader(
                input, format, {"": rindex}, fits_ext=fits_ext) as reader:
            if hasattr(reader, "_guess_length"):
                pbar = ProgressBar()
            else:
                pbar = ProgressBar(n_rows=len(reader))
            for chunk in reader:
                ext_idx.extend(chunk[rindex])
                pbar.update(len(chunk))
            pbar.close()
        self.logger.info("sorting indices")
        ext_idx.sort()
        # for each index in external data find the mathing index in data store
        self.logger.info("mapping indices")
        idx_map = {}  # look up the index in the data store later
        row_ext = 0
        pbar = ProgressBar(len(self))
        pbar_step = 1000
        # optimize the search by assuming ext_idx and datastore[index] are
        # sorted by value
        for row_all, index_all in enumerate(self.datastore[lindex]):
            index_ext = ext_idx[row_ext]
            if index_all == index_ext:
                idx_map[index_ext] = row_all
                row_ext += 1  # search for the next item
                if row_ext == len(ext_idx):
                    break
            if row_all % pbar_step == 0:
                pbar.update(pbar_step)
        pbar.close()
        if row_ext != len(ext_idx):
            message = "some indices in the input file cannot be matched"
            raise ValueError(message)
        # map the external data onto the data store
        with create_reader(
                input, format, col_map_dict, fits_ext=fits_ext) as reader:
            self.logger.info("mapping values")
            if col_map_dict is None:  # by default map names one-to-one
                col_map_dict = {col: col for col in reader.colnames}
            if hasattr(reader, "_guess_length"):
                pbar = ProgressBar()
            else:
                pbar = ProgressBar(n_rows=len(reader))
            for chunk in reader:
                idx_target = [idx_map[i] for i in chunk[rindex]]
                for path, colname in col_map_dict.items():
                    self.datastore[path][idx_target] = chunk[colname]
                pbar.update(len(chunk))
            pbar.close()

    @job
    def query(
            self, output, query=None, verify=False, format=False, columns=None,
            compression=None, hdf5_shuffle=False, hdf5_checksum=False):
        """
        Query the data store and write matching results to an output file. The
        queries are given as strings and allow computation with table columns
        or constant data using brackets and the following operators:
            arithmetic operators:
                ** (power), - (negation), * (multiplication), / division,
                % (modular division), + (addition), - (subtraction)
            bitwise operators:
                ~ (not), & (and), | (or), ^ (xor),
            comparators:
                == (equal), != (not equal), < (less), > (greater),
                <= (less or equal), >= (greater or equal)
            logial operators:
                NOT, AND, OR, XOR
        The usual order of operators applies, from arithmetic operators
        (hightest) to logical operators (lowest priority), with brackets taking
        precedence.

        Example: position/ra/true > 30.0 AND position/ra/true < 60
        selects all objects with right ascension (named "position/ra/true" in
        the data store) 30 < RA < 60.

        Parameters:
        -----------
        output : str
            Path where the output file is written to. If no output is specified
            and the output format is CSV, the data is written to stdout.
        query : str
            Query to perform on the data, see notes above.
        verify : bool
            Verify the column check sums before querying the data.
        format : str
            Format descibing string, see galmock.core.readwrite for all
            supported file formats.
        columns : list
            List of column names to include in the output file.
        compression : str
            Compression algorithm to use, if supported by the output format,
            see galmock.core.readwrite.
        hdf5_shuffle : bool
            Whether to apply the shuffle filter in HDF5 files.
        hdf5_checksum : bool
            Whether to add Fletcher32 check sums to the HDF5 data sets.
        """
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
            # add attributes if format supports it
            attrs = {
                col: request_table[col].attr for col in request_table.colnames}
            writer.store_attributes(attrs)
        if not to_stdout:
            message = "wrote {:,d} matching entries ({:})".format(
                n_select, bytesize_with_prefix(writer.filesize))
            self.logger.info(message)

    @job
    def prepare_MICE2(self, mag, evo):
        """
        Apply the MICE2 evolution correction.

        Parameters:
        -----------
        mag : str
            Directory in the data store that contains the raw MICE2 input
            magnitudes.
        evo : str
            Directory in the data store in which the evolution corrected
            magnitudes are store, using the same filter key name.
        """
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
        """
        Convert the Flagship fluxes to AB magnitudes and determine if a galaxy
        is the central halo galaxy (if their halo ID == 0).

        Parameters:
        -----------
        flux : str
            Directory in the data store that contains the raw Flagship input
            fluxes.
        evo : str
            Directory in the data store in which the converted AB magnitudes
            are store, using the same filter key name.
        gal_idx : str
            Path of the galaxy index column in the data store.
        is_central : str
            Path of the central galaxy flag column in the data store.
        """
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
        """
        Compute and add the effect of magnification to a set of input
        magnitudes.

        Parameters:
        -----------
        mag : str
            Directory in the data store that contains the input magnitudes.
        evo : str
            Directory in the data store in which the magnification corrected
            magnitudes are store, using the same filter key name.
        """
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
        """
        Compute a proxy for the intrinsic galaxy size, projected on-sky, in
        arcseconds. When configured with flux_frac=0.5, this corresponds to
        the effective radius (emitting 50% of the total flux).

        Multiple intrinsic galaxy size can exist simultenously if named
        accordingly.

        Parameters:
        -----------
        config : str
            Path to a TOML photometry configuration file, see
            galmock.photometry.PhotometryParser.
        """
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
        """
        Add an aperture realisation for each galaxy, based on its intrinsic
        size and average observing conditions (PSF FWHM).

        Multiple aperture realistaions can exist simultenously if named
        accordingly.

        Parameters:
        -----------
        config : str
            Path to a TOML photometry configuration file, see
            galmock.photometry.PhotometryParser.
        """
        # check the configuration file
        configuration = PhotometryParser(config)
        # apply the magnification correction to the model magnitudes
        # initialize the aperture computation
        self.datastore.pool.set_worker(apertures_wrapped)
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
        """
        Add an photometry realisation for each galaxy, based on its aperture
        size and limiting magnitude.

        Multiple photometry realistaions can exist simultenously if named
        accordingly.

        Parameters:
        -----------
        mag : str
            Directory in the data store that contains the noiseless input
            magnitudes.
        real : str
            Directory in the data store where the magnitude realisations and
            their errors are stored.
        config : str
            Path to a TOML photometry configuration file, see
            galmock.photometry.PhotometryParser.
        seed : str
            Used to seed the random number generator for the photometric error.
            Results are only reproducible, if the number of parallel processes
            remains the same.
        """
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
        """
        Derive an observable from an external data set based on nearest
        neighbour matching in a feature space (for example galaxy magntiudes).
        
        The external data must be a valid galmock.core.datastore which can be
        created using Galmock.create, leaving the columns argument blank.

        Parameters:
        -----------
        config : str
            Path to a TOML sample matching configuration file, see
            galmock.matching.MatcherParser.
        """
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
        """
        Add photometric redshifts computed with BPZ for a magnitude
        realisation (magnitude errors are required).

        Multiple photometric redshift estimates can exist simultenously if
        named accordingly.

        Note that BPZ requires a python2 environment, see the
        requirements_BPZ_py2.txt in the root directory.

        Parameters:
        -----------
        mag : str
            Directory in the data store that contains the input magnitude
            realistion and errors.
        zphot : str
            Directory in the data store where the photo-z and additional BPZ
            output is stored.
        config : str
            Path to a TOML BPZ configuration file, see
            galmock.photoz.BpzParser.
        """
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
        """
        Apply a sample selection function to the simulation data. Each sample
        must be defined in galmock.samples and requires a configuration file
        that specifies the required input columns in the data store. Adds a
        bit mask column that indicates each (sub-)selection step performed.

        Each sample has its own selection bit mask.

        Parameters:
        -----------
        config : str
            Path to a TOML sample configuration file, see the sample specific
            parsers in galmock.samples.reference.
        sample : str
            Name used to identify the selection function implemented in
            galmock.samples.
        area : float
            On-sky area covered by the input simulation data.
        type : str
            If a simulation data specific selection function should be used
            (e.g. MICE2), defaults to the reference implementation (closest to
            the literature selection).
        seed : str
            Used to seed the random number generator for the photometric error.
            Results are only reproducible, if the number of parallel processes
            remains the same.
        """
        # check the configuration file
        Parser = SampleManager.get_parser(type, sample)
        configuration = Parser(config)
        # apply the magnification correction to the model magnitudes
        message = "apply selection funcion: {:} (type: {:})"
        self.logger.info(message.format(sample, type))
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
        self.datastore.pool.allow_modify = False
        # add the description attribute to the output column
        bitmask.attr = {"description": BMM.description}
        # show final statistics
        N_mock = DensitySampler.count_selected(bitmask)
        density = "{:.3f} / arcmin²".format(N_mock / area / 3600.0)
        self.logger.info("density of selected objects: " + density)
