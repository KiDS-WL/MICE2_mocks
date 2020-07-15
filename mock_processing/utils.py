import os
import sys
from collections import OrderedDict
from time import asctime, strptime

import toml

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
        Normalized path with substitutions applied.
    """
    # check for tilde
    if path.startswith("~" + os.sep):
        path = os.path.expanduser(path)
    elif path.startswith("~"):  # like ~user/path/file
        home_root = os.path.dirname(os.path.expanduser("~"))
        path = os.path.join(home_root, path[1:])
    path = os.path.expandvars(path)
    path = os.path.normpath(path)
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
        message = "overwriting output column: {:}"
        logger.warn(message.format(path))
    else:
        message = "creating output column: {:}"
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


class ParseColumnMap(object):
    """
    Parse a column mapping file used to convert the column names of the input
    files into paths within the data store.

    Parameters:
    -----------
    path : str
        Column map file path
    """

    def __init__(self, path):
        # unpack the potentially nested dictionary by converting nested keys
        # into file system paths
        self._column_map = dict()
        # parse the TOML file and parse it as dictionary
        with open(path) as f:
            self._traverse_dict(toml.load(f))

    @property
    def get(self):
        """
        Get the column mapping data store path -> input file column.
        """
        return self._column_map

    def _traverse_dict(self, subdict, path=""):
        """
        Iterate the directory and insert the values in the column map by
        concatenating nested keywords like file system paths.

        Parameters:
        -----------
        subdict : dict
            Dictionary that maps the input file column names.
        path : str
            path under which the dictionary items are registered in the global
            column map.
        """
        for key, value in subdict.items():
            if type(key) is not str:
                message = "invalid type {:} for set name"
                raise TypeError(message.format(str(type(value))))
            if type(value) is dict:
                self._traverse_dict(value, os.path.join(path, key))
            else:
                if type(value) is list:
                    dtype_tuple = tuple(value)
                else:
                    dtype_tuple = (value, None)
                self._column_map[os.path.join(path, key)] = dtype_tuple


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

    def finalize(self, timestamp=None):
        """
        Take the current local time and format and write the attributes to the
        registered columns.

        Parameters:
        -----------
        timestamp : str
            Text encoded time stamp.
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
            column.attr = attrs  # write
