import inspect
import logging
import os
import sys
import warnings
from collections import OrderedDict

import numpy as np

import galmock
from galmock.core.bitmask import BitMaskManager
from galmock.core.config import TableParser
from galmock.core.datastore import ModificationStamp
from galmock.core.readwrite import create_reader, create_writer
from galmock.core.utils import (ProgressBar, bytesize_with_prefix,
                                check_query_columns, sha1sum,
                                substitute_division_symbol)
from galmock.Flagship import find_central_galaxies, flux_to_magnitudes_wrapped
from galmock.matching import DataMatcher, DistributionEstimator, MatcherParser
from galmock.MICE2 import evolution_correction_wrapped
from galmock.photometry import (PhotometryParser, apertures_wrapped,
                                find_percentile_wrapped,
                                magnification_correction_wrapped,
                                photometry_realisation_wrapped)
from galmock.photoz import BpzManager, BpzParser
from galmock.samples import (DensitySampler, DumpConfig, RedshiftSampler,
                             SampleManager)


def _create_job_logger():
    jobname = inspect.stack()[1][3]
    logger = logging.getLogger(".".join([__name__, jobname]))
    logger.setLevel(logging.DEBUG)
    logger.info("initialising job: {:}".format(jobname))
    return logger


def _get_pseudo_sys_argv():
    caller_frame = inspect.stack()[1][0]
    # list the positional, variable and named arguments
    params = OrderedDict()
    arginfo = inspect.getargvalues(caller_frame)
    # positional
    for key in arginfo.args:
        params[key] = arginfo.locals[key]
    # varaible
    if arginfo.varargs is not None:
        for i, arg in enumerate(arginfo.locals[arginfo.varargs]):
            key = "{:}[{:d}]".format(arginfo.varargs, i)
            params[key] = arg
    # keyword
    if arginfo.keywords is not None:
        for key, arg in arginfo.locals[arginfo.keywords].items():
            params[key] = arg
    # create a sys.argv style list
    sys_argv = [caller_frame.f_code.co_name]
    for key, val in params.items():
        sys_argv.append("{:}={:}".format(key, str(val)))
    return sys_argv


def datastore_create(
        datastore,
        input,
        format=None,
        fits_ext=1,
        columns=None,
        purge=False):
    logger = _create_job_logger()

    # check the columns file
    if columns is not None:
        configuration = TableParser(columns)
        col_map_dict = configuration.column_map
    else:
        col_map_dict = None

    # read the data file and write it to the memmory mapped data store
    with create_reader(input, format, col_map_dict, fits_ext) as reader:

        with galmock.DataStore.create(datastore, len(reader), purge) as ds:
            ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())

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
            ds.show_preview()

            message = "finalized data store with {:,d} rows ({:})"
            logger.info(message.format(end, ds.filesize))


def datastore_verify(
        datastore,
        **kwargs):
    with galmock.DataStore.open(datastore) as ds:
        ds.show_metadata()
        ds.verify()


def datastore_info(
        datastore,
        columns=False,
        attr=False,
        history=False,
        logs=False,
        **kwargs):
    with galmock.DataStore.open(datastore) as ds:
        ds.show_metadata()
        if columns:
            ds.show_columns()
        if attr:
            ds.show_attributes()
        if history:
            ds.show_history()
        if logs:
            ds.show_logs()


def datastore_query(
        datastore,
        output,
        query=None,
        verify=False,
        format=False,
        columns=None,
        compression=None,
        hdf5_shuffle=False,
        hdf5_checksum=False,
        **kwargs):
    logger = _create_job_logger()
    to_stdout = output is None

    # read the input table and write the selected entries to the output file
    with galmock.DataStore.open(datastore) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())

        # parse the math expression
        selection_columns, expression = substitute_division_symbol(query, ds)
        if selection_columns is not None:
            selection_table = ds[sorted(selection_columns)]
        # check the requested columns
        request_table, dtype = check_query_columns(columns, ds)
        
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

        with create_writer(
                output, format, dtype, compression,
                hdf5_shuffle, hdf5_checksum) as writer:

            # query the table and write to the output fie
            if expression is not None:
                message = "filtering data and writing output file ..."
            else:
                message = "converting data and writing output file ..."
            logger.info(message)
            if not to_stdout:
                pbar = ProgressBar(len(ds))
            n_select = 0
            for start, end in ds.row_iter(writer.buffersize // dtype.itemsize):

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
                        logger.exception(message)
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
    logger = _create_job_logger()

    # apply the evolution correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
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
            logger.exception(str(e))
            raise

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
    logger = _create_job_logger()

    # convert model fluxes to model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        ds.pool.set_worker(flux_to_magnitudes_wrapped)

        # find all flux columns
        try:
            model_fluxes, _ = ds.load_photometry(flux)
        except KeyError as e:
            logger.exception(str(e))
            raise

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
    logger = _create_job_logger()

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
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
            logger.exception(str(e))
            raise

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
    logger = _create_job_logger()

    # check the configuration file
    configuration = PhotometryParser(config)

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        ds.pool.set_worker(find_percentile_wrapped)

        ds.pool.add_argument_constant(configuration.intrinsic["flux_frac"])

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
            configuration.intrinsic["r_effective"], dtype="f4", attr={
                "description":
                "effective radius (emitting {:.1%} of the flux)".format(
                    configuration.intrinsic["flux_frac"])},
            overwrite=True)
        # add column to call signature
        ds.pool.add_result_column(configuration.intrinsic["r_effective"])

        # compute and store the corrected magnitudes
        ds.pool.execute()
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def apertures(
        datastore,
        config,
        threads=-1,
        **kwargs):
    logger = _create_job_logger()

    # check the configuration file
    configuration = PhotometryParser(config)

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        # initialize the aperture computation
        ds.pool.set_worker(apertures_wrapped)
        ds.pool.add_argument_constant(configuration.method)
        ds.pool.add_argument_constant(configuration)

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
        for key in configuration.filter_names:
            for out_path, desc in output_columns:
                formatted_path = out_path.format(
                    configuration.aperture_name, key)
                ds.add_column(
                    formatted_path, dtype="f4", overwrite=True, attr={
                        "description":
                        desc.format(
                            configuration.method, configuration.PSF[key])})
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
    logger = _create_job_logger()

    # check the configuration file
    configuration = PhotometryParser(config)

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        # find all magnitude columns
        try:
            input_mags, _ = ds.load_photometry(mag)
        except KeyError as e:
            logger.exception(str(e))
            raise

        # select the required magnitude columns
        available = set(input_mags.keys())
        missing = set(configuration.filter_names) - available
        if len(missing) > 0:
            message = "requested filters not found: {:}".format(
                ", ".join(missing))
            logger.error(message)
            raise KeyError(message)
        
        # initialize the photometry generation
        ds.pool.set_worker(photometry_realisation_wrapped)
        ds.pool.add_argument_constant(configuration)

        # collect the filter-specific arguments
        for key in configuration.filter_names:
            # 1) magnitude column
            mag_path = input_mags[key]
            ds.require_column(mag_path, "{:}-band".format(key))
            ds.pool.add_argument_column(mag_path)
            # 2) magnitude limit
            ds.pool.add_argument_constant(configuration.limits[key])
            # 3) S/N correction factors
            if configuration.photometry["apply_apertures"]:
                snr_path = "apertures/{:}/snr_correction/{:}".format(
                    configuration.aperture_name, key)
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
        for key in configuration.filter_names:
            for out_path, desc in output_columns:
                ds.add_column(
                    out_path.format(real, key),
                    dtype=ds[mag_path].dtype.str, overwrite=True, attr={
                        "description": desc.format(
                            configuration.method, mag_path,
                            configuration.limits[key])})
                ds.pool.add_result_column(out_path.format(real, key))

        # compute and store the corrected magnitudes
        ds.pool.execute(seed=seed)
        logger.info("computation completed for {:,d} entries".format(len(ds)))


def match_data(
        datastore,
        config,
        threads=-1,
        **kwargs):
    logger = _create_job_logger()

    # check the configuration file
    configuration = MatcherParser(config)

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        with DataMatcher(configuration) as matcher:

            ds.pool.set_worker(matcher.apply)
            # increase the default chunksize, larger chunks will be marginally
            # faster but the progress update will be infrequent
            ds.pool.chunksize = threads * ds.pool.chunksize

            # select the required feature columns
            for feature_path in configuration.features.values():
                ds.require_column(feature_path, "feature")
                ds.pool.add_argument_column(feature_path, keyword=feature_path)
            # check that the data types are compatible
            try:
                matcher.check_features(ds)
            except Exception as e:
                logger.exception(str(e))
                raise

            # make the output columns for each observable
            for output_path, dtype, attr in matcher.observable_information():
                ds.add_column(
                    output_path, dtype=dtype, attr=attr, overwrite=True)
                ds.pool.add_result_column(output_path)

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
    logger = _create_job_logger()

    # check the configuration file
    configuration = BpzParser(config)

    # run BPZ on the selected magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        # find all magnitude columns
        try:
            input_mags, input_errs = ds.load_photometry(mag)
        except KeyError as e:
            logger.exception(str(e))
            raise

        # launch the BPZ manager
        with BpzManager(configuration) as bpz:

            ds.pool.set_worker(bpz.execute)
            ds.pool.parse_thread_id = True

            # add the magnitude and error columns to call signature
            for key in bpz.filter_names:
                try:
                    mag_key, err_key = input_mags[key], input_errs[key]
                    ds.pool.add_argument_column(mag_key)
                    ds.pool.add_argument_column(err_key)
                except KeyError as e:
                    logger.exception(str(e))
                    raise

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
    logger = _create_job_logger()

    # check the configuration file
    Parser = SampleManager.get_parser(type, sample)
    configuration = Parser(config)

    # apply the magnification correction to the model magnitudes
    with galmock.DataStore.open(datastore, False) as ds:
        ds._timestamp = ModificationStamp(_get_pseudo_sys_argv())
        ds.pool.max_threads = threads

        logger.info("apply selection funcion: {:}".format(sample))

        # allow the worker threads to modify the bitmask column directly
        ds.pool.allow_modify = True

        # make the output column
        BMM = BitMaskManager(sample)
        bitmask = ds.add_column(
            configuration.bitmask, dtype=BMM.dtype, overwrite=True)
        # initialize the selection bit (bit 1) to true, all subsequent
        # selections will be joined with AND
        logger.debug("initializing bit mask")
        bitmask[:] = 1

        # start with photometric selection
        selector_class = SampleManager.get_selector(type, sample)
        selector = selector_class(BMM)

        ds.pool.set_worker(selector.apply)
        # select the columns needed for the selection function
        ds.pool.add_argument_column(configuration.bitmask)
        for name, path in configuration.selection.items():
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
                        BMM, area, bitmask,
                        ds[configuration.selection["redshift"]])

                ds.pool.set_worker(sampler.apply)
                # select the columns needed for the selection function
                ds.pool.add_argument_column(configuration.bitmask)
                # redshift weighted density requires a mock n(z) estimate
                if isinstance(sampler, RedshiftSampler):
                    ds.pool.add_argument_column(
                        configuration.selection["redshift"],
                        keyword="redshift")

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
