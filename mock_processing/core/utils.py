import os
from hashlib import sha1

from tqdm import tqdm


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


def sha1sum(path):
    """
    Compute a SHA-1 checksum for a given path.

    Parameters:
    -----------
    path : str
        Path to file for which the checksum is computed.

    Returns:
    --------
    checksum : str
        Checksum encoded as 40-digit hex-string.
    """
    hasher = sha1()
    with open(path, "rb") as f:
        while True:
            buffer = f.read(1048576)
            if not buffer:
                break
            hasher.update(buffer)
    checksum = hasher.hexdigest()
    return checksum


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
