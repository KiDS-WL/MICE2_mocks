import os
import sys
from collections import OrderedDict
from hashlib import sha1
from time import asctime, strptime

from tqdm import tqdm

from memmap_table import MemmapTable
from ._version import __version__


def expand_path(path):
    """
    Normalises a path (e.g. from the command line) and substitutes environment
    variables and the user (e.g. ~/ or ~user/).

    Parameters:
    -----------
    path : str
        Input raw path.

    Returns:
    --------
    path : str
        Normalized absolute path with substitutions applied.
    """
    # check for tilde
    if path.startswith("~" + os.sep):
        path = os.path.expanduser(path)
    elif path.startswith("~"):  # like ~user/path/file
        home_root = os.path.dirname(os.path.expanduser("~"))
        path = os.path.join(home_root, path[1:])
    path = os.path.expandvars(path)
    path = os.path.normpath(path)
    path = os.path.abspath(path)
    return path


def open_datastore(path, logger, readonly=True):
    """
    Wrapper to open an existing MemmapTable on disk.

    Parameters:
    -----------
    path : str
        Path to MemmapTable storage (must be a directory).
    logger : python logger instance
        Logger instance that logs events.
    readonly : bool
        Whether the storage is opened as read-only.

    Returns:
    --------
    table : memmap_table.MemmapTable
        Opened data storage interface.
    """
    # open the data store
    logger.info("opening data store: {:}".format(path))
    mode = "r" if readonly else "r+"
    try:
        table = MemmapTable(path, mode=mode)
    except Exception as e:
        logger.handleException(e)
    return table


def require_column(table, logger, path, col_desc=None):
    """
    Convenience functions to check whether a column exists in a given table. If
    not, a KeyError is raised and logged.

    Parameters:
    -----------
    table : memmap_table.MemmapTable
        Opened data storage interface.
    logger : python logger instance
        Logger instance that logs events.
    path : str
        Column name (path relative to the table root).
    col_desc : str
        Descriptive name of the column used to format the error message.
    """
    if path not in table:
        if col_desc is None:
            col_desc = ""
        else:
            col_desc = col_desc + " "
        message = "{:}column not found: {:}".format(col_desc, path)
        logger.handleException(KeyError(message))


def create_column(table, logger, path, *args, **kwargs):
    """
    Wrapper to create a new column in an existing MemmapTable.

    Parameters:
    -----------
    table : memmap_table.MemmapTable
        Opened data storage interface with write permissions.
    logger : python logger instance
        Logger instance that logs events.
    path : str
        Column name (path relative to the table root).

    Further arguments are passed to MemmapTable.add_column() and specify the
    data type, attributes and overwrite permissions.

    Returns:
    --------
    column : memmap_table.Column
        Newly created table column instance.
    """
    if path in table:
        message = "overwriting column: {:}"
        logger.warn(message.format(path))
    else:
        message = "creating column: {:}"
        logger.debug(message.format(path))
    column = table.add_column(path, *args, **kwargs)
    return column


def build_history(table, logger=None):
    """
    Read the creation labels from all columns attributes of the data store and
    keep a unique listing of the called scripts. The time resolution is
    1 second.

    Parameters:
    -----------
    table : memmap_table.MemmapTable
        Data storage interface processed with this pipeline.
    logger : python logger instance
        Logger instance that logs events (optional).

    Returns:
    --------
    history : OrderedDict
        Mapping of timestamp -> script comands, ordered by time.
    """
    calls = {}
    for column in table.colnames:
        attrs = table[column].attr
        try:
            timestamp = strptime(attrs["created at"])
            calls[timestamp] = attrs["created by"]
        except KeyError:
            message = "column has no creation time stamp: {:}".format(column)
            if logger is None:
                print("WARNING: " + message)
            else:
                logger.warn(message)
        except TypeError:
            continue  # no attribute exists
    # return history ordered time and convert time stamps back to strings
    history = OrderedDict()
    for key in sorted(calls):
        history[asctime(key)] = calls[key]
    return history


def bytesize_with_prefix(nbytes, precision=2):
    """
    Convert a data size in bytes to a printable string with metric prefix (e.g.
    314,215,650 Bytes = 299.66 MB).

    Parameters:
    -----------
    nbytes : int
        Number of bytes to convert
    precision : int
        Number of significant digits to included in converted string.
    
    Returns:
    --------
    string : str
        Byte size in with metric prefix.
    """
    # future proof prefix list
    units = ["YB", "ZB", "EB", "PB", "TB", "GB", "MB", "kB", "Bytes"]
    # divide size by 1024 and increase the prefix until the number is < 1000 
    value = float(nbytes)
    unit = units.pop()
    while value > 1000.0 and len(units) > 0:
        value /= 1024.0
        unit = units.pop()
    string = "{:.{p}f} {:}".format(value, unit, p=precision)
    return string


class ProgressBar(tqdm):
    """
    tqdm progress bar with standardized configuration and optimized prediction
    smoothing scale.

    Parameters:
    -----------
    n_rows : int
        The total number of rows to expect. If None, only the number of
        processed rows and the current rate are displayed.
    prefix : str
        Prefix for the progressbar (optional).
    """

    def __init__(self, n_rows=None, prefix=None):
        super().__init__(
            total=n_rows, leave=False, unit_scale=True, unit=" rows",
            dynamic_ncols=True, desc=prefix)
        self.smoothing = 0.05


def sha1sum(path):
    hasher = sha1() 
    with open(path, "rb") as f:
        while True:
            buffer = f.read(1048576)
            if not buffer:
                break
            hasher.update(buffer)
    checksum = hasher.hexdigest()
    return checksum


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
        self._columns = []
        # store the script call that created/modified this column
        call_basename = os.path.basename(sys_argv[0])
        call_arguments = sys_argv[1:]
        # create a dictionary with all shared entries
        self._attrs = {}
        self._attrs["created by"] = " ".join([call_basename, *call_arguments])
        self._attrs["version"] = __version__
    
    def register(self, column):
        """
        Register a column for which the attributes should be updated.

        Parameters:
        -----------
        column : memmap_table.Column
            Column instance from the data store to update.
        """
        if not hasattr(column, "attr"):
            raise TypeError("column must have an attribute 'attr'")
        self._columns.append(column)

    def finalize(self, timestamp=None, checksum=True):
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
        # update all columns
        for column in self._columns:
            attrs = column.attr  # read
            attrs.update(self._attrs)
            if checksum:
                attrs.update({"SHA-1 checksum": sha1sum(column.filename)})
            column.attr = attrs  # write
