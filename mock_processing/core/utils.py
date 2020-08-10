import os
from hashlib import sha1

import numpy as np
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


def footprint_area(RAmin, RAmax, DECmin, DECmax):
    """
    Calculate the area within a RA/DEC bound.

    Parameters
    ----------
    RAmin : array_like or float
        Minimum right ascension of the bounds.
    RAmax : array_like or float
        Maximum right ascension of the bounds.
    DECmin : array_like or float
        Minimum declination of the bounds.
    DECmax : array_like or float
        Maximum declination of the bounds.

    Returns
    -------
    area : array_like or float
        Area within the bounds in square degrees.
    """
    # np.radians and np.degrees avoids rounding errors
    sin_DEC_min, sin_DEC_max = np.sin(np.radians([DECmin, DECmax]))
    dRA = RAmax - RAmin
    if RAmin > RAmax:
        dRA += 360.0
    area = dRA * np.degrees(sin_DEC_max - sin_DEC_min)
    return area
