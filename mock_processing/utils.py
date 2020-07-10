import os
from collections import OrderedDict
from time import asctime, strptime

from memmap_table import MemmapTable
from ._version import __version__


def expand_path(path):
    """
    Normalises a path (e.g. from the command line) and substitutes environment
    variables and the user (e.g. ~/ or ~user/).
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
    # open the data store
    logger.info("opening data store: {:}".format(path))
    mode = "r" if readonly else "r+"
    try:
        return MemmapTable(path, mode=mode)
    except Exception as e:
        logger.handleException(e)


def build_history(table):
    # read the creation labels from all columns and keep a unique listing
    # assuming that no two successful calls occured within one second
    calls = {}
    for column in table.colnames:
        attrs = table[column].attr
        try:
            timestamp = strptime(attrs["created at"])
            calls[timestamp] = attrs["created by"]
        except KeyError:
            message = "WARNING: column has no creation time stamp: {:}"
            print(message.format(column))
    # return history ordered time and convert time stamps back to strings
    history = OrderedDict()
    for key in sorted(calls):
        history[asctime(key)] = calls[key]
    return history


class ColumnDictTranslator(object):

    def __init__(self, col_dict):
        self._col_dict = col_dict
        self.column_map = dict()
        self._traverse_dict(self._col_dict)

    def _traverse_dict(self, subdict, path=""):
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
                self.column_map[os.path.join(path, key)] = dtype_tuple


class ModificationStamp(object):

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
        if not hasattr(column, "attr"):
            raise TypeError("column must have an attribute 'attr'")
        self._columns.append(column)

    def finalize(self, timestamp=None):
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
