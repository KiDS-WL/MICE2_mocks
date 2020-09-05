import logging
import multiprocessing
import os
import sys
from collections import OrderedDict
from time import asctime, strptime

from mmaptable import MmapTable

from galmock.core.version import __version__
from galmock.core.utils import bytesize_with_prefix, expand_path, sha1sum
from galmock.core.parallel import ParallelTable


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def preview(datastore):
    preview_lines = []
    for line in str(datastore).split("\n"):
        if line.strip():
            preview_lines.append(line)
    linelength = max(len(l) for l in preview_lines)
    header = "{ preview }".center(linelength, "-")
    footer = "-" * linelength
    preview = "\n".join(preview_lines[1:-2])
    sys.stdout.write("{:}\n{:}\n{:}\n".format(header, preview, footer))


class ModificationStamp(object):
    """
    Write attributes which indicate by which pipeline scipt (including command
    line arguments and pipline version) and when column have been modified
    last in the data store.

    Parameters:
    -----------
    sys_argv : sys.argv
        Commandline arguments including script name.
    """

    def __init__(self, sys_argv):
        self._columns = OrderedDict()
        # store the script call that created/modified this column
        call_basename = os.path.basename(sys_argv[0])
        call_arguments = sys_argv[1:]
        # create a dictionary with all shared entries
        self._attrs = {}
        self._attrs["created by"] = " ".join([call_basename, *call_arguments])
        self._attrs["pipeline version"] = __version__
    
    def __len__(self):
        return len(self._columns)

    def _update_attribute(self, name, attributes):
        """
        Update the attribute dictionary of a given column.

        Parameters:
        -----------
        name : str
            Name of the column to update.
        attributes : dict
            Key-value pairs to add to the existing column attributes.
        """
        attrs = self._columns[name].attr  # read
        if attrs is None:
            attrs = {}
        attrs.update(attributes)
        self._columns[name].attr = attrs  # write

    def register(self, column, name):
        """
        Register a column for which the attributes should be updated.

        Parameters:
        -----------
        column : mmaptable.MmapColumn
            Column instance from the data store to update.
        name : string
            Name of the column (path within the data store).
        """
        if not hasattr(column, "attr"):
            raise TypeError("column must have an attribute 'attr'")
        self._columns[name] = column

    def add_checksums(self):
        # get an ordered list of file names for the columns
        filenames = [column.filename for column in self._columns.values()]
        # compute the check sums in parallel processes
        with multiprocessing.Pool(4) as pool:
            checksums = pool.map(sha1sum, filenames)
        # store the attributes
        for name, checksum in zip(self._columns, checksums):
            self._update_attribute(name, {"SHA-1 checksum": checksum})

    def finalize(self, timestamp=None):
        """
        Take the current local time and format and write the attributes to the
        registered columns.

        Parameters:
        -----------
        timestamp : str
            Text encoded time stamp.
        checksum : bool
            Whether to add a SHA-1 checksum attribute.
        """
        # store the current time if none is provided
        if timestamp is None:
            self._attrs["created at"] = asctime()
        else:
            self._attrs["created at"] = timestamp
        # update the columns attributes
        for name in self._columns:
            self._update_attribute(name, self._attrs)


class DataStore(MmapTable):
    """
    Wrapper to open an existing MmapTable on disk.

    Parameters:
    -----------
    path : str
        Path to MmapTable storage (must be a directory).
    readonly : bool
        Whether the storage is opened as read-only.
    """

    def __init__(self, path, nrows=None, mode="r"):
        # open the data store
        try:
            super().__init__(path, nrows, mode)
        except Exception as e:
            logger.exception(str(e))
            raise
        self._filesize = bytesize_with_prefix(self.nbytes)
        self._timestamp = ModificationStamp(sys.argv)
        self.pool = ParallelTable(self)

    @classmethod
    def create(cls, path, nrows, overwrite=False):
        full_path = expand_path(path)
        exists = os.path.exists(full_path)
        if not overwrite and exists:
            # this is a safety feature since any existing output directory is
            # erased completely
            message = "ouput path exists but overwriting is not permitted: {:}"
            message = message.format(full_path)
            logger.error(message)
            raise OSError(message)
        logger.info("creating data store: {:}".format(full_path))
        instance = cls(full_path, nrows, mode="w+")
        logger.debug("allocating {:,d} table rows".format(len(instance)))
        return instance

    @classmethod
    def open(cls, path, readonly=True):
        logger.info("opening data store: {:}".format(path))
        mode = "r" if readonly else "r+"
        instance = cls(expand_path(path), mode=mode)
        return instance

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        exit_with_error = args[0] is not None
        self.close(add_checksum=not exit_with_error)

    def close(self, add_checksum=True):
        if len(self._timestamp) > 0:
            message = "updating attributes"
            if add_checksum:
                message = "computing checksums and " + message
            logger.debug(message)
            if add_checksum:
                self._timestamp.add_checksums()
            self._timestamp.finalize()
        super().close()
        logger.info("data store closed")

    @property
    def filesize(self):
        return self._filesize

    def expand(self, nrows):
        super().resize(len(self) + nrows)
        logger.debug("allocated {:,d} additional rows".format(nrows))

    def require_column(self, path, description=""):
        """
        Test whether the table contains a requested column. If not, a KeyError
        is raised and logged.

        Parameters:
        -----------
        path : str
            Column name (path relative to the table root).
        description : str
            Descriptive name of the column that may be used to format the error
            message.
        """
        if path not in self:
            if len(description) > 0:
                description += " "
            message = "{:}column not found: {:}".format(description, path)
            logger.error(message)
            raise KeyError(message)

    def add_column(self, path, *args, **kwargs):
        """
        Wrapper to create a new column in an existing MmapTable.

        Parameters:
        -----------
        path : str
            Column name (path relative to the table root).

        Further arguments are passed to MmapTable.add_column() and specify
        the data type, attributes and overwrite permissions.

        Returns:
        --------
        column : mmaptable.MmapColumn
            Newly created table column instance.
        """
        if path in self:
            logger.warn("overwriting column: {:}".format(path))
        else:
            logger.debug("creating column: {:}".format(path))
        column = super().add_column(path, *args, **kwargs)
        self._timestamp.register(column, path)
        self._filesize = bytesize_with_prefix(self.nbytes)
        return column

    def verify_column(self, path):
        """
        Wrapper to create a new column in an existing MmapTable.

        Parameters:
        -----------
        path : str
            Name of column to check (path relative to the table root).
        """
        column = self[path]
        try:
            checksum = column.attr["SHA-1 checksum"]
            assert(checksum == sha1sum(column.filename))
        except KeyError:
            logger.warn("no checksum provided: {:}".format(path))
        except AssertionError:
            message = "checksums do not match: {:}".format(path)
            logger.exception(message)
            raise

    def get_history(self):
        """
        Read the creation labels from all columns attributes of the data store
        and keep a unique listing of the called scripts. The time resolution is
        1 second.

        Returns:
        --------
        history : OrderedDict
            Mapping of timestamp -> script comands, ordered by time.
        """
        calls = {}
        for column in self.colnames:
            attrs = self[column].attr
            try:
                timestamp = strptime(attrs["created at"])
                calls[timestamp] = attrs["created by"]
            except KeyError:
                message = "column has no creation time stamp: {:}"
                logger.warn(message.format(column))
            except TypeError:
                continue  # no attribute exists
        # return history ordered time and convert time stamps back to strings
        history = OrderedDict()
        for key in sorted(calls):
            history[asctime(key)] = calls[key]
        return history

    def load_photometry(self, photometry_path, filter_selection=None):
        """
        Collect all columns belonging to a photometry (realization) by using a
        subpath (e.g. /mags/model).

        Parameters:
        -----------
        photometry_path : str
            Path within the data table that contains photometric data (labeled
            with filter names, e.g. /mags/model/r, /mags/model/i).
        filter_selection : array_like
            filter keys to exclude from the selection.

        Returns:
        --------
        photometry_columns : dict
            Dictionary of column names with photometric data, labeled with the
            filter name.
        error_columns : dict
            Dictionary of column names with photometric errors, labeled with
            the filter name.
        """
        if filter_selection is None:
            filter_selection = []
        photometry_columns = {}
        error_columns = {}
        for column in self.colnames:
            # match the root name of each column against the photometry path
            root, key_str = os.path.split(column)
            if key_str.endswith("_err"):
                key = key_str[:-4]
            else:
                key = key_str
            if root == photometry_path:
                # check if the matching filter is excluded
                if key in filter_selection:
                    continue
                # check that this filter does not exist yet, which could happen
                # if a path is selected that contains multiple photometries
                if key_str.endswith("_err"):
                    error_columns[key] = column
                elif key in photometry_columns:
                    message = "found multiple matches for filter: {:}"
                    message = message.format(key)
                    logger.error(message)
                    ValueError(message)
                else:
                    photometry_columns[key] = column
        if len(photometry_columns) == 0:
            message = "photometry not found: {:}".format(photometry_path)
            logger.error(message)
            raise KeyError(message)
        return photometry_columns, error_columns
