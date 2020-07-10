import os


# compute an automatic buffer size for the system
_megabyte = 1048576  # bytes
try:
    import psutil
    # between 50 MB < 5% < 500 MB
    auto_size = int(0.05 * psutil.virtual_memory().total)
    BUFFERSIZE = max(auto_size, 50 * _megabyte)
    BUFFERSIZE = min(BUFFERSIZE, 500 * _megabyte)
except Exception:
    BUFFERSIZE = 50 * _megabyte


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
