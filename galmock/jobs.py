import inspect
import os
import sys
import warnings

import numpy as np
from mmaptable.mathexpression import MathTerm

import galmock as mocks


def datastore_create(
        datastore,
        input,
        format=None,
        fits_ext=1,
        columns=None,
        purge=False):
    from galmock.core.config import TableParser
    from galmock.core.readwrite import guess_format, SUPPORTED_READERS
    from galmock.core.utils import ProgressBar

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore, append=False)

    # check the columns file
    if columns is not None:
        config = TableParser(columns, logger)
        col_map_dict = config.column_map

    # automatically determine the input file format
    if format is None:
        try:
            format = guess_format(input)
        except NotImplementedError as e:
            logger.handleException(e)
    message = "opening input as {:}: {:}".format(format.upper(), input)
    logger.info(message)

    # create a standardized input reader
    reader_class = SUPPORTED_READERS[format]
    try:
        kwargs = {"ext": fits_ext}
        if col_map_dict is not None:
            kwargs["datasets"] = set(col_map_dict.values())
        reader = reader_class(input, **kwargs)
        # create a dummy for col_map_dict
        if col_map_dict is None:
            col_map_dict = {name: name for name in reader.colnames}
    except Exception as e:
        logger.handleException(e)
    message = "buffer size: {:.2f} MB ({:,d} rows)".format(
        reader.buffersize / 1024**2, reader.bufferlength)
    logger.debug(message)

    # read the data file and write it to the memmory mapped data store
    with reader:

        with mocks.DataStore.create(
                datastore, len(reader), purge, logger=logger) as ds:

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
                    logger.handleException(e)

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
                end = reader.current_row  # index where reading will continue
                # map column by column onto the output table
                for path, (colname, dtype) in col_map_dict.items():
                    # the CSV reader provides a conservative estimate of the
                    # required number of rows so we need to keep allocating
                    # more memory if necessary
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
            try:
                preview = str(ds)
                sys.stdout.write("\n" + preview + "\n\n")
            except Exception:
                logger.warn("table preview failed")

            message = "finalized data store with {:,d} rows ({:})"
            logger.info(message.format(end, ds.filesize))


def datastore_verify(datastore):
    from galmock.core.utils import sha1sum

    with mocks.DataStore.open(datastore) as ds:

        # display the table meta data
        print("==> META DATA")
        n_cols, n_rows = ds.shape
        print("root:     {:}".format(ds.root))
        print("size:     {:}".format(ds.filesize))
        print("shape:    {:,d} rows x {:d} columns".format(n_rows, n_cols))

        # verify the check sums
        header = "==> COLUMN NAME"
        width_cols = max(len(header), max(
            len(colname) for colname in ds.colnames))
        print("\n{:}    {:}  {:}".format(
            header.ljust(width_cols), "STATUS ", "HASH"))
        # compute and verify the store checksums column by column
        n_good, n_warn, n_error = 0, 0, 0
        line = "{:<{width}s}    {:<7s}  {:s}"
        for name in ds.colnames:
            column = ds[name]
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
        if n_good == len(ds.colnames):
            print("\nAll columns passed")
        else:
            print("\nPassed:   {:d}\nWarnings: {:d}\nErrors:   {:d}".format(
                n_good, n_warn, n_error))


def datastore_info(
        datastore,
        columns=False,
        attr=False,
        history=False,
        logs=False,
        **kwargs):

    with mocks.DataStore.open(datastore) as ds:

        # display the table meta data
        print("==> META DATA")
        n_cols, n_rows = ds.shape
        print("root:     {:}".format(ds.root))
        print("size:     {:}".format(ds.filesize))
        print("shape:    {:,d} rows x {:d} columns".format(n_rows, n_cols))

        if columns:
            # list all the column names and data types
            header = "==> COLUMN NAME"
            width_cols = max(len(header), max(
                len(colname) for colname in ds.colnames))
            print("\n{:}    {:}".format(header.ljust(width_cols), "TYPE"))
            for name in ds.colnames:
                colname_padded = name.ljust(width_cols)
                line = "{:}    {:}".format(
                    colname_padded, str(ds[name].dtype))
                print(line)

        if attr:
            # for each column print a summary of the attached attributes
            print("\n==> ATTRIBUTES")
            for name in ds.colnames:
                print()
                # print the column name indented and then a tree-like listing
                # of the attributes (drawing connecting lines for better
                # visibitilty)
                print("{:}".format(name))
                attrs = ds[name].attr
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

        if history:
            # list all pipeline script calls ordered by time
            print("\n==> HISTORY")
            date_width = 24
            for date, call in ds.get_history().items():
                print("{:} : {:}".format(date.ljust(date_width), call))

        if logs:
            # show the log file
            print("\n==> LOGS")
            logpath = ds.root + ".log"
            if not os.path.exists(logpath):
                raise OSError("log file not found: {:}".format(logpath))
            with open(logpath) as f:
                for line in f.readlines():
                    print(line.strip())


def datastore_query(
        datastore,
        output,
        query=None,
        verify=False,
        format=False,
        columns=None,
        compression=None,
        hdf5_shuffle=False,
        hdf5_checksum=False):
    from galmock.core.readwrite import (BUFFERSIZE, guess_format,
                                        SUPPORTED_WRITERS)
    from galmock.core.utils import ProgressBar, bytesize_with_prefix

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)
    # determine if output goes to stdout in CSV format (allows redirecting)
    to_stdout = output is None
    if to_stdout:  # only print in case of error
        logger.setTermLevel("error")

    # read the input table and write the selected entries to the output file
    with mocks.DataStore.open(datastore, logger=logger) as ds:

        # parse the math expression
        if query is not None:
            selection_columns = set()
            # Since the symbol / can be used as column name and division
            # symbol, we need to temporarily substitute the symbol before
            # parsing the expression.
            substitute = "#"
            try:
                # apply the substitutions to all valid column names apperaing
                # in the math expression to avoid substitute intended divisions
                for colname in ds.colnames:
                    substitue = colname.replace("/", substitute)
                    while colname in query:
                        query = query.replace(colname, substitue)
                        selection_columns.add(colname)
                expression = MathTerm.from_string(query)
                # recursively undo the substitution
                expression._substitute_characters(substitute, "/")
                # display the interpreted expression
                message = "apply selection: {:}".format(expression.expression)
                logger.info(message)
            except SyntaxError as e:
                message = e.args[0].replace(substitute, "/")
                logger.handleException(SyntaxError(message))
            except Exception as e:
                logger.handleException(e)
            # create a sub-table with the data needed for the selection
            selection_table = ds[sorted(selection_columns)]
        else:
            expression = None
            selection_columns = None

        # check the requested columns
        if columns is not None:
            # check for duplicates
            requested_columns = set()
            for colname in columns:
                if colname in requested_columns:
                    message = "duplicate column: {:}".format(colname)
                    logger.handleException(KeyError(message))
                requested_columns.add(colname)
            # find requested columns that do not exist in the table
            missing_cols = requested_columns - set(ds.colnames)
            if len(missing_cols) > 0:
                message = "column {:} not found: {:}".format(
                    "name" if len(missing_cols) == 1 else "names",
                    ", ".join(sorted(missing_cols)))
                logger.handleException(KeyError(message))
            # establish the requried data type
            n_cols = len(requested_columns)
            message = "select a subset of {:d} column".format(n_cols)
            if n_cols > 1:
                message += "s"
            logger.info(message)
            dtype = np.dtype([
                (colname, ds.dtype[colname]) for colname in columns])
            # create a table view with only the requested columns
            request_table = ds[columns]
        else:
            dtype = ds.dtype
            request_table = ds
        
        # verify the data if requested
        if verify:
            logger.info("verifying SHA-1 checksums")
            if columns is None:
                verification_columns = ds.colnames
            else:
                verification_columns = [col for col in columns]
                if selection_columns is not None:
                    verification_columns.extend(selection_columns)
            name_width = max(len(name) for name in verification_columns)
            for i, name in enumerate(verification_columns, 1):
                sys.stdout.write("checking {:3d} / {:3d}: {:}\r".format(
                    i, len(verification_columns), name.ljust(name_width)))
                sys.stdout.flush()
                ds.verify_column(name)

        # automatically determine the format file format
        if output is None:  # write to stdout in csv format
            format = "csv"
        if format is None:
            try:
                format = guess_format(output)
            except NotImplementedError as e:
                logger.handleException(e)
        message = "writing output as {:}: {:}".format(format.upper(), output)
        logger.info(message)

        # create a standardized output writer
        writer_class = SUPPORTED_WRITERS[format]
        try:
            writer = writer_class(
                dtype, output, overwrite=True,
                # format specific parameters
                compression=compression,
                hdf5_shuffle=hdf5_shuffle,
                hdf5_checksum=hdf5_checksum)
        except Exception as e:
            logger.handleException(e)

        with writer:

            # determine an automatic buffer/chunk size
            chunksize = BUFFERSIZE // ds.itemsize
            message = "buffer size: {:} ({:,d} rows)".format(
                bytesize_with_prefix(writer.buffersize), writer.bufferlength)
            logger.debug(message)

            # query the table and write to the output fie
            if expression is not None:
                message = "filtering data and writing output file ..."
            else:
                message = "converting data and writing output file ..."
            logger.info(message)
            if not to_stdout:
                pbar = ProgressBar(len(ds))
            n_select = 0
            for start, end in ds.row_iter(chunksize):

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
                        logger.handleException(KeyError(message))
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

        logger.info("computation completed for {:,d} entries".format(len(ds)))
        message = "wrote {:,d} matching entries ({:})".format(
            n_select, bytesize_with_prefix(writer.filesize))
        logger.info(message)


def init_pipeline(
        datastore,
        input,
        columns,
        format=None,
        fits_ext=1,
        purge=False,
        **kwargs):
    datastore_create(
        datastore, input, format=format, fits_ext=fits_ext,
        columns=columns, purge=purge)


def convert_input(
        datastore,
        input,
        format=None,
        fits_ext=1,
        purge=False,
        **kwargs):
    datastore_create(
        datastore, input, format=format, fits_ext=fits_ext,
        columns=None, purge=purge)


def prepare_MICE2(
        datastore,
        mag,
        evo,
        threads=-1,
        **kwargs):
    from galmock.MICE2 import evolution_correction_wrapped

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # apply the evolution correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        ds.pool.set_worker(evolution_correction_wrapped)

        # find redshift column
        z_path = "position/z/true"
        ds.require_column(z_path, "true redshift")
        ds.pool.add_argument_column(z_path)

        # find all magnitude columns
        try:
            model_mags, _ = ds.load_photometry(mag)
        except KeyError as e:
            logger.handleException(e)

        # create the output columns
        for key, mag_path in model_mags.items():
            evo_path = os.path.join(evo, key)
            # create new output columns
            ds.add_column(
                evo_path, dtype=ds[mag_path].dtype.str, overwrite=True,
                attr={
                    "description":
                    "{:} with evolution correction applied".format(mag_path)})
            # add columns to call signature
            ds.pool.add_argument_column(mag_path)
            ds.pool.add_result_column(evo_path)

        # compute and store the corrected magnitudes in parallel
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def prepare_Flagship(
        datastore,
        flux,
        mag,
        gal_idx=None,
        is_central=None,
        threads=-1,
        **kwargs):
    from galmock.Flagship import (find_central_galaxies,
                                  flux_to_magnitudes_wrapped)

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # convert model fluxes to model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        ds.pool.set_worker(flux_to_magnitudes_wrapped)

        # find all flux columns
        try:
            model_fluxes, _ = ds.load_photometry(flux)
        except KeyError as e:
            logger.handleException(e)

        # create the output columns
        for key, flux_path in model_fluxes.items():
            mag_path = os.path.join(mag, key)
            # create new output columns
            ds.add_column(
                mag_path, dtype=ds[flux_path].dtype.str, overwrite=True,
                attr={
                    "description":
                    "{:} converted to AB magnitudes".format(flux_path)})
            # add columns to call signature
            ds.pool.add_argument_column(flux_path)
            ds.pool.add_result_column(mag_path)

        # compute and store the corrected magnitudes in parallel
        ds.pool.execute()

        # add the central galaxy flag
        ds.pool.set_worker(find_central_galaxies)

        # find the input column
        ds.require_column(gal_idx, "galaxy index")
        ds.pool.add_argument_column(gal_idx)

        # create the output column
        ds.add_column(
            is_central, dtype="bool", overwrite=True, attr={
                "description": "host central galaxy flag"})
        ds.pool.add_result_column(is_central)

        # compute and store the corrected magnitudes in parallel
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def magnification(
        datastore,
        mag,
        lensed,
        threads=-1,
        **kwargs):
    from galmock.photometry import magnification_correction_wrapped

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        ds.pool.set_worker(magnification_correction_wrapped)

        # find convergence column
        kappa_path = "lensing/kappa"
        ds.require_column(kappa_path, "convergence")
        ds.pool.add_argument_column(kappa_path)

        # find all magnitude columns
        try:
            input_mags, _ = ds.load_photometry(mag)
        except KeyError as e:
            logger.handleException(e)

        # create the output columns
        for key, mag_path in input_mags.items():
            lensed_path = os.path.join(lensed, key)
            # create new output columns
            ds.add_column(
                lensed_path, dtype=ds[mag_path].dtype.str, overwrite=True,
                attr={
                    "description":
                    "{:} with magnification correction applied".format(
                        lensed_path)})
            # add columns to call signature
            ds.pool.add_argument_column(mag_path)
            ds.pool.add_result_column(lensed_path)

        # compute and store the corrected magnitudes
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def effective_radius(
        datastore,
        config,
        threads=-1,
        **kwargs):
    from galmock.photometry import PhotometryParser, find_percentile_wrapped

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    config = PhotometryParser(config, logger)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        ds.pool.set_worker(find_percentile_wrapped)

        ds.pool.add_argument_constant(config.intrinsic["flux_frac"])

        # find disk and bulge component columns
        input_columns = (
            ("disk size", "shape/disk/size"),
            ("bulge size", "shape/bulge/size"),
            ("bulge fraction", "shape/bulge/fraction"))
        for col_desc, path in input_columns:
            ds.require_column(path, col_desc)
            ds.pool.add_argument_column(path)

        # create the output column
        ds.add_column(
            config.intrinsic["r_effective"], dtype="f4", overwrite=True, attr={
                "description":
                "effective radius (emitting {:.1%} of the flux)".format(
                    config.intrinsic["flux_frac"])})
        # add column to call signature
        ds.pool.add_result_column(config.intrinsic["r_effective"])

        # compute and store the corrected magnitudes
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def apertures(
        datastore,
        config,
        threads=-1,
        **kwargs):
    from galmock.photometry import PhotometryParser, apertures_wrapped

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    config = PhotometryParser(config, logger)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        # initialize the aperture computation
        ds.pool.set_worker(apertures_wrapped)
        ds.pool.add_argument_constant(config.method)
        ds.pool.add_argument_constant(config)

        # find effective radius and b/a ratio columns
        input_columns = (
            ("effective radius", "shape/R_effective"),
            ("b/a ratio", "shape/axis_ratio"))
        for col_desc, path in input_columns:
            ds.require_column(path, col_desc)
            ds.pool.add_argument_column(path)

        output_columns = (  # for each filter three output columns are required
            ("apertures/{:}/major_axis/{:}",
                "{:} aperture major axis (PSF size: {:.2f}\")"),
            ("apertures/{:}/minor_axis/{:}",
                "{:} aperture minor axis (PSF size: {:.2f}\")"),
            ("apertures/{:}/snr_correction/{:}",
                "{:} aperture S/N correction (PSF size: {:.2f}\")"))
        # make the output columns for each filter
        for key in config.filter_names:
            for out_path, desc in output_columns:
                formatted_path = out_path.format(config.aperture_name, key)
                ds.add_column(
                    formatted_path, dtype="f4", overwrite=True, attr={
                        "description":
                        desc.format(config.method, config.PSF[key])})
                # collect all new columns as output targets
                ds.pool.add_result_column(formatted_path)

        # compute and store the apertures
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def photometry(
        datastore,
        mag,
        real,
        config,
        seed="sapling",
        threads=-1,
        **kwargs):
    from galmock.photometry import (PhotometryParser,
                                    photometry_realisation_wrapped)

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    config = PhotometryParser(config, logger)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        # find all magnitude columns
        try:
            input_mags, _ = ds.load_photometry(mag)
        except KeyError as e:
            logger.handleException(e)

        # select the required magnitude columns
        available = set(input_mags.keys())
        missing = set(config.filter_names) - available
        if len(missing) > 0:
            message = "requested filters not found: {:}".format(
                ", ".join(missing))
            logger.handleException(KeyError(message))
        
        # initialize the photometry generation
        ds.pool.set_worker(photometry_realisation_wrapped)
        ds.pool.add_argument_constant(config)

        # collect the filter-specific arguments
        for key in config.filter_names:
            # 1) magnitude column
            mag_path = input_mags[key]
            ds.require_column(mag_path, "{:}-band".format(key))
            ds.pool.add_argument_column(mag_path)
            # 2) magnitude limit
            ds.pool.add_argument_constant(config.limits[key])
            # 3) S/N correction factors
            if config.photometry["apply_apertures"]:
                snr_path = "apertures/{:}/snr_correction/{:}".format(
                    config.aperture_name, key)
                ds.require_column(
                    mag_path, "{:}-band S/N correction".format(key))
                ds.pool.add_argument_column(snr_path)
            else:  # disable aperture correction
                ds.pool.add_argument_constant(1.0)

        output_columns = (  # for each filter three output columns are required
            ("{:}/{:}",
             "{:} photometry realisation (from {:}, limit: {:.2f} mag)"),
            ("{:}/{:}_err",
             "{:} photometric error (from {:}, limit: {:.2f} mag)"))
        # make the output columns for each filter
        for key in config.filter_names:
            for out_path, desc in output_columns:
                ds.add_column(
                    out_path.format(real, key),
                    dtype=ds[mag_path].dtype.str, overwrite=True, attr={
                        "description": desc.format(
                            config.method, mag_path, config.limits[key])})
                ds.pool.add_result_column(out_path.format(real, key))

        # compute and store the corrected magnitudes
        ds.pool.execute(seed=seed)
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def match_data(
        datastore,
        config,
        threads=-1,
        **kwargs):
    from galmock.matching import DataMatcher, MatcherParser

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    config = MatcherParser(config, logger)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        with DataMatcher(config, logger) as matcher:

            ds.pool.set_worker(matcher.apply)
            # increase the default chunksize, larger chunks will be marginally
            # faster but the progress update will be infrequent
            ds.pool.chunksize = threads * ds.pool.chunksize

            # select the required feature columns
            for feature_path in config.features.values():
                ds.require_column(feature_path, "feature")
                ds.pool.add_argument_column(feature_path, keyword=feature_path)
            # check that the data types are compatible
            try:
                matcher.check_features(ds)
            except Exception as e:
                logger.handleException(e)

            # make the output columns for each observable
            for output_path, dtype, attr in matcher.observable_information():
                ds.add_column(
                    output_path, dtype=dtype, attr=attr, overwrite=True)
                ds.pool.add_result_column(output_path)

            logger.info(
                "building feature space tree, this may take a while ...")
            matcher.build_tree()

            # compute and store the corrected magnitudes
            ds.pool.execute()  # cKDTree implementation releases GIL
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def BPZ(
        datastore,
        mag,
        zphot,
        config,
        threads=-1,
        **kwargs):
    from galmock.photoz import BpzManager, BpzParser

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    config = BpzParser(config, logger)

    # run BPZ on the selected magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        # find all magnitude columns
        try:
            input_mags, input_errs = ds.load_photometry(mag)
        except KeyError as e:
            logger.handleException(e)

        # launch the BPZ manager
        logger.info("initializing BPZ")
        with BpzManager(config, logger) as bpz:

            ds.pool.set_worker(bpz.execute)
            ds.pool.parse_thread_id = True

            # add the magnitude and error columns to call signature
            for key in bpz.filter_names:
                try:
                    mag_key, err_key = input_mags[key], input_errs[key]
                    ds.pool.add_argument_column(mag_key)
                    ds.pool.add_argument_column(err_key)
                except KeyError as e:
                    logger.handleException(e)

            # create the output columns
            for key, desc in bpz.descriptions.items():
                zphot_path = os.path.join(zphot, key)
                # create new output columns
                ds.add_column(
                    zphot_path, dtype=bpz.dtype[key].str, overwrite=True,
                    attr={"description": desc})
                # add columns to call signature
                ds.pool.add_result_column(zphot_path)

            ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def select_sample(
        datastore,
        config,
        sample,
        area,
        type="reference",
        seed="sapling",
        threads=-1,
        **kwargs):
    from galmock.core.bitmask import BitMaskManager
    from galmock.matching import DistributionEstimator
    from galmock.samples import (DensitySampler, DumpConfig, RedshiftSampler,
                                 SampleManager)

    fname = inspect.currentframe().f_code.co_name
    logger = mocks.PipeLogger(fname, datastore)

    # check the configuration file
    Parser = SampleManager.get_parser(type, sample)
    config = Parser(config, logger)

    # apply the magnification correction to the model magnitudes
    with mocks.DataStore.open(datastore, False, logger=logger) as ds:
        ds.pool.max_threads = threads

        logger.info("apply selection funcion: {:}".format(sample))

        # allow the worker threads to modify the bitmask column directly
        ds.pool.allow_modify = True

        # make the output column
        BMM = BitMaskManager(sample)
        bitmask = ds.add_column(
            config.bitmask, dtype=BMM.dtype, overwrite=True)
        # initialize the selection bit (bit 1) to true, all subsequent
        # selections will be joined with AND
        logger.debug("initializing bit mask")
        bitmask[:] = 1

        # start with photometric selection
        selector_class = SampleManager.get_selector(type, sample)
        selector = selector_class(BMM)

        ds.pool.set_worker(selector.apply)
        # select the columns needed for the selection function
        ds.pool.add_argument_column(config.bitmask)
        for name, path in config.selection.items():
            ds.require_column(path)
            ds.pool.add_argument_column(path, keyword=name)

        # apply selection
        ds.pool.execute(seed=seed, prefix="photometric selection")

        # optional density sampling
        sampler_class = SampleManager.get_sampler(type, sample)
        if sampler_class is not NotImplemented:
            try:
                # surface density
                if issubclass(sampler_class, DensitySampler):
                    logger.info("estimating surface density ...")
                    sampler = sampler_class(BMM, area, bitmask)
                # redshift weighted density
                else:
                    logger.info("estimating redshift density ...")
                    sampler = sampler_class(
                        BMM, area, bitmask, ds[config.selection["redshift"]])

                ds.pool.set_worker(sampler.apply)
                # select the columns needed for the selection function
                ds.pool.add_argument_column(config.bitmask)
                # redshift weighted density requires a mock n(z) estimate
                if isinstance(sampler, RedshiftSampler):
                    ds.pool.add_argument_column(
                        config.selection["redshift"], keyword="redshift")

                # apply selection
                    ds.pool.execute(seed=seed, prefix="density sampling")
            except ValueError as e:
                if str(e).startswith("sample density must be"):
                    logger.warning("skipping sampling due to low mock density")
                else:
                    raise e

        # add the description attribute to the output column
        bitmask.attr = {"description": BMM.description}

        # show final statistics
        N_mock = DensitySampler.count_selected(bitmask)
        density = "{:.3f} / arcmin²".format(N_mock / area / 3600.0)
        logger.info("density of selected objects: " + density)
        logger.info("computation completed for {:,d} entries".format(len(ds)))
