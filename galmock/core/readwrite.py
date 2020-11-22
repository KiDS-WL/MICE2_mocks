#
# This module implements unified reader and writer classes for CSV, FITS, HDF5
# and parquet files. The latter three formats are available only, if the
# necessary packages are installed.
#
# Data is always read or written in chunks to prevent memory overflow. This
# may be a performance bottle neck for smaller files.
#

import csv
import logging
import os
import sys
import warnings
from sys import stdout
from collections import OrderedDict

import numpy as np

from galmock.core.utils import bytesize_with_prefix, expand_path


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_mega_byte = 1024 * 1024
BUFFERSIZE = 100 * _mega_byte

EXTENSION_ALIAS = {}
SUPPORTED_READERS = {}
SUPPORTED_WRITERS = {}


def register(name, extensions):
    def decorator_register(iohandler):
        """
        Register Reader or Writer objects.
        """
        if issubclass(iohandler, Reader):
            SUPPORTED_READERS[name] = iohandler
        elif issubclass(iohandler, Writer):
            SUPPORTED_WRITERS[name] = iohandler
        else:
            raise TypeError("Object must be subclass of 'Reader' or 'Writer'")
        EXTENSION_ALIAS[name] = extensions
        return iohandler
    return decorator_register


def guess_format(fpath) -> str:
    """
    Guess the file type based on its extension.

    Parameters:
    -----------
    fpath : str
        File path from which the format is guessed.

    Returns:
    --------
    format_key : str
        File format identifier to find the appropriate reader of writer class.
    """
    _, ext = os.path.splitext(fpath.strip())
    ext = ext.lower().lstrip(".")
    # compare with all supported file types
    for format_key, aliases in EXTENSION_ALIAS.items():
        if ext in aliases:
            return format_key
    raise NotImplementedError(
        "unsupported file format with extension: {:}".format(ext))


class ConvertByteOrder(object):
    """
    Automatically convert byte order of numpy arrays for a fixed data type to
    the native order of the architecture. Usefull to convert a series input
    arrays.

    Parameters:
    -----------
    dtype : numpy.dtype or str
        Expected input data type. If the byte order does not match the
        architecture, it will be converted to the native order.
    """

    def __init__(self, dtype):
        # get the numpy data type of the expected input
        if type(dtype) is str:
            dtype = np.dtype(str)
        else:
            assert(isinstance(dtype, np.dtype))
        # check if the byte order matches the native order, identified by the
        # numpy dtype string representation: little endian = "<" and
        # big endian = ">"
        if dtype.str.startswith(("<", ">")):
            if sys.byteorder == "little":
                self._dtype = np.dtype("<" + dtype.base.str.strip("><"))
            elif sys.byteorder == "big":
                self._dtype = np.dtype(">" + dtype.base.str.strip("><"))
        # types that do not have come with a byte order
        else:
            self._dtype = dtype

    def __call__(self, data) -> np.array:
        """
        Reverse the byte order of the input array in place if necessaray. Must
        be of the same type as during initialization.

        Parameters:
        -----------
        data : numpy.array
            Input array to correct, must be of the same type.

        Returns:
        --------
        corrected : numpy.array
            Input array, but with converted byte order in place.
        """
        # do not copy data and only accept data of the same type
        corrected = data.astype(self._dtype, casting="equiv", copy=False)
        return corrected

    @property
    def dtype(self) -> np.dtype:
        """
        Return the output data type.
        """
        return self._dtype


################################  base classes  ###############################


class DataBase(object):

    _dtype = None
    _len = 0

    def __len__(self):
        return self._len

    @property
    def colnames(self) -> tuple:
        """
        Tuple of the column names.
        """
        return tuple(self._dtype.names)

    @property
    def dtype(self) -> np.dtype:
        """
        Numpy data type of the data columns.
        """
        return self._dtype

    @property
    def itemsize(self) -> int:
        """
        Total number of bytes a row of the data occupies.
        """
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        """
        Total number of the bytes the data occupies.
        """
        return self.size * self.itemsize

    @property
    def size(self) -> int:
        """
        Total number of elements in the data.
        """
        return np.multiply(*self.shape)

    @property
    def shape(self) -> tuple:
        """
        The number of columns and rows/entries of the data.
        """
        return (len(self._dtype), len(self),)


class FileInterface(object):

    _path = None
    _file = None

    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()

    @property
    def filesize(self) -> int:
        """
        Input file size in bytes.
        """
        return os.path.getsize(self._path)

    def close(self):
        return NotImplemented


class BufferQueue(DataBase):
    """
    Queue to collect data and return it in chunks of fixed size.

    Parameters:
    -----------
    dtype : numpy.dtype
        Data type of the allocated buffer structure with named fields.
    """

    _buffersize = BUFFERSIZE

    def __init__(self, dtype):
        self._dtype = dtype
        self._buffer = np.empty(0, self._dtype)

    def __len__(self):
        return len(self._buffer)

    @property
    def bufferlength(self) -> int:
        """
        Number of rows the buffer can hold.
        """
        return max(1, self._buffersize // self.itemsize)

    @property
    def buffersize(self) -> int:
        """
        The size of the read buffer in bytes.
        """
        return self._buffersize

    @buffersize.setter
    def buffersize(self, nbytes):
        """
        Set the size of the read buffer in bytes.
        """
        self._buffersize = int(nbytes)

    @property
    def full(self) -> bool:
        """
        Whether there is at least [buffersize] of data left in the queue.
        """
        return len(self) >= self.bufferlength
    
    @property
    def empty(self) -> bool:
        """
        Whether the queue is completely depleted.
        """
        return len(self) == 0

    def push(self, data):
        """
        Add an arbitrary number of data rows to the end of the queue.

        Parameters:
        -----------
        data : numpy.array
            Data to append to the queued.
        """
        if data.dtype != self.dtype:
            message = "input data types do not match the buffer data types"
            raise TypeError(message)
        self._buffer = np.append(self._buffer, data)
    
    def pop(self) -> np.array:
        """
        Collect [buffersize] of data from the beginning of the queue. Raises a
        ValueError if there is not enough data.

        Returns:
        --------
        data : numpy.array
            Queued data of [buffersize].
        """
        if self.full:
            data = self._buffer[:self.bufferlength].copy()
            # shrink the queue
            self._buffer = self._buffer[self.bufferlength:]
        else:
            message = "insufficient data in the queue to fill the buffer"
            raise ValueError(message)
        return data
    
    def pop_all(self) -> iter:
        """
        Iterator that yields chunks of [buffersize] from the beginning from the
        queue.

        Yields:
        --------
        data : numpy.array
            Queued data of [buffersize].
        """
        while self.full:
            yield self.pop()

    def clear(self) -> np.array:
        """
        Empty the queue and return any remaning data.

        Returns:
        --------
        data : numpy.array
            Queued data.
        """
        data = self._buffer.copy()
        self._buffer = np.empty(0, self._dtype)
        return data


class Reader(DataBase, FileInterface):

    _current_row = 0

    def __init__(self, dtype):
        self._buffer = BufferQueue(dtype)

    def __iter__(self):
        while True:
            try:
                yield self.read_chunk()
            except EOFError:
                break

    def reopen(self):
        # rest the _current_row counter and move back to beginning of the file
        return NotImplemented

    @property
    def bufferlength(self) -> int:
        """
        Number of rows the buffer can hold.
        """
        return max(1, self.buffersize // self.itemsize)

    @property
    def buffersize(self) -> int:
        """
        The size of the read buffer in bytes.
        """
        return self._buffer.buffersize

    @buffersize.setter
    def buffersize(self, nbytes):
        """
        Set the size of the read buffer in bytes.
        """
        self._buffer.buffersize = nbytes

    @property
    def current_row(self) -> int:
        """
        Index of next row to read.
        """
        return self._current_row

    def read_chunk(self):
        # read until the buffer is full and return the data a record array
        return NotImplemented


class Writer(DataBase, FileInterface):

    def __init__(self, dtype):
        self._buffer = BufferQueue(dtype)

    @staticmethod
    def _check_overwrite(fpath, overwrite):
        if os.path.exists(fpath):
            if overwrite:
                os.remove(fpath)
            else:
                raise OSError("output file '{:}' exists".format(fpath))

    @property
    def bufferlength(self) -> int:
        """
        Number of rows the buffer can hold.
        """
        return max(1, self.buffersize // self.itemsize)

    @property
    def buffersize(self) -> int:
        """
        The size of the read buffer in bytes.
        """
        return self._buffer.buffersize

    @buffersize.setter
    def buffersize(self, nbytes):
        """
        Set the size of the read buffer in bytes.
        """
        self._buffer.buffersize = nbytes

    def _write(self, data):
        # write some amount of data to the end of the file
        return NotImplemented

    def write_chunk(self, table):
        # push to the buffer queue and write if necessary
        self._buffer.push(table)
        for data in self._buffer.pop_all():
            self._write(data)

    def flush(self):
        """
        Write any queued data to the end of the file.
        """
        remainder = self._buffer.clear()
        if len(remainder) > 0:
            self._write(remainder)


###################  shipped with all python installations  ###################


@register("csv", ("csv",))
class CSVreader(Reader):
    """
    Read a CSV file into numpy arrays using a buffer.

    Parameters:
    -----------
    fpath : str
        Path to the CSV file.
    dtypes : dict
        Provide a mapping of {column name: data type} to overwrite the
        automatically derived data types (e.g. reduced bit-size). Recommended
        for text columns which are inferred as fixed width which may be lead to
        truncation.
    """

    def __init__(self, fpath, dtypes=None, **kwargs):
        self._path = expand_path(fpath)
        self._file = open(fpath)
        try:
            self._guess_dialect()
            self._guess_dtype(dtypes)
            # allocate the buffer (which we don't really utilize)
            self._buffer = BufferQueue(self.dtype)
            self._guess_length()
        except Exception as e:
            self._file.close()
            logger.exception(str(e))
            raise

    def _guess_dialect(self):
        """
        Read the first 10 kB of the input file to determine the CSV dialect.
        """
        text_sample = self._file.read(10240)  # read 10kB
        # identify the CSV dialect and whether the file has a header
        sniffer = csv.Sniffer()
        self._dialect = sniffer.sniff(text_sample)
        self._has_header = sniffer.has_header(text_sample)
        self.reopen(skip_header=False)

    def _dtype_from_string(self, colname, entry):
        """
        Infer data type automatically: numeric type without decimal point must
        be integers, non-numeric types are kept as fixed length strings.

        Parameters:
        -----------
        colname : str
            Column name of processed element.
        entry : str
            String from which the data type is inferred.

        Returns:
        --------
        dtype : tuple
            Pair of (colname, dtype), where dtype is a numpy.dtype instance.
        """
        try:
            int(entry)
            dtype = (colname, np.dtype(np.int).str)
            return dtype
        except ValueError:
            pass
        # other numeric types should be floats
        try:
            float(entry)
            dtype = (colname, np.dtype(np.float).str)
            return dtype
        # fall back to string type of fixed width
        except ValueError:
            dtype = (colname, "|S{:d}".format(len(entry)))
            # issue a warning that long data might be truncated
            message = "column '{:}' is interpreted as length "
            message += "{:d} string, longer entries will be truncated"
            warnings.warn(message.format(colname, len(entry)))
        return dtype

    def _guess_dtype(self, dtype_hints):
        """
        Take a sample row from the input file (already split into elements) and
        try to infer the data type of each column.

        Parameters:
        -----------
        dtype_hints : dict
            Provide a mapping of {column name: data type} to overwrite the
            automatically derived data types (e.g. reduced bit-size).
            Recommended for text columns which are inferred as fixed width
            which may be lead to truncation.
        """
        if dtype_hints is None:
            dtype_hints = {}
        else:
            if type(dtype_hints) is not dict:
                raise TypeError("data type hints must be a dictionary")
        # read the first data row to determine the data type from it
        if self._has_header:
            colnames = next(self._reader)
            row = next(self._reader)
        else:
            row = next(self._reader)
            colnames = ["column{:d}".format(i) for i in range(len(row))]
        self.reopen()
        # infer the data type from each row element
        dtypes = []
        for colname, entry in zip(colnames, row):
            try:
                # check for user-suggested data type
                hint = dtype_hints.pop(colname)
                if type(hint) is str:
                    dtype = (colname, np.dtype(hint).str)
                else:
                    dtype = (colname, hint.str)
            except KeyError:
                dtypes.append(self._dtype_from_string(colname, entry))
            except AttributeError:
                message = "data type hints must be numpy type strings or "
                message += "dtype instances"
                logger.exception(message)
                raise TypeError(message)
        # check if there are hints which do not match any of the file's columns
        if len(dtype_hints) > 0:
            message = "There are {:d} data type ".format(len(dtype_hints))
            message += "hints which did not match any existing column"
            raise ValueError(message)
        self._dtype = np.dtype(dtypes)

    def _guess_length(self):
        """
        Guess the number of rows from extrapolating from the first chunk of
        data with an additional 5% safety margin.
        """
        i = 0
        # read the first chunk and assume it is representative
        line_bytes = []
        while i < self.bufferlength:
            try:
                row = next(self._reader)
                # estimate the row length, assuming no quoting is used by
                # taking element delimiters and line terminator into account
                line_bytes.append(
                    sum(len(entry) for entry in row) +
                    len(self._dialect.delimiter) * (len(row) - 1) +
                    len(self._dialect.lineterminator))
                i += 1
            except StopIteration:  # if the file is very short
                break
        # compute the average number of bytes per row and estimate the number
        # of rows by comparing to the file size
        mean_line_bytes = sum(line_bytes) / len(line_bytes)
        self._len = int(self.filesize / mean_line_bytes * 1.05)
        self.reopen()

    def read_chunk(self) -> np.array:
        """
        Read a chunk from the input file until the buffer is full or the end of
        the file is reached.

        Returns:
        -----------
        buffer : numpy.array
            Data read from disk, packed in an array with fields named like the
            data columns.
        """
        if self._file.closed:
            raise ValueError("I/O operation on closed file.")
        # read and place row by row in the buffer
        i = 0
        buffer = np.empty(self.bufferlength, self.dtype)
        try:
            while i < self.bufferlength:
                row = next(self._reader)
                buffer[i] = tuple(row)
                i += 1
        except StopIteration:
            # If the end of the file is reached, return the buffer with
            # emtpy buffer rows truncated. If the method is called the next
            # time, there are no rows left and it aborts with an error.
            if i > 0:
                buffer = buffer[:i]  # truncate the buffer
            else:
                raise EOFError("reached end of file")
        except ValueError:
            # CSV is not safe against on evolution of the table schema
            message = "row {:d} with length {:d} does not match header "
            message += "schema with length {:d}"
            message = message.format(i, len(row), len(self.dtype))
            logger.exception(message)
            raise ValueError(message)
        self._current_row += i
        return buffer

    def reopen(self, skip_header=True):
        """
        Rewind to the first row with data.
        """
        self._file.seek(0)
        self._current_row = 0
        self._reader = csv.reader(self._file, self._dialect)
        # skip the header row if it exists
        if self._has_header and skip_header:
            next(self._reader)

    def close(self):
        """
        Close the underlying file pointer.
        """
        self._file.close()


@register("csv", ("csv",))
class CSVwriter(Writer):
    """
    Write table data to a CSV file or stdout using a buffer.

    Parameters:
    -----------
    dtype : numpy.dtype
        Describes the expected column names and data types.
    fpath : str
        Path to the CSV file. If None, the data will be written to stdout.
    overwrite : bool
        Whether the output file is overwritten if it exists.
    """

    _header_written = False

    def __init__(self, dtype, fpath=None, overwrite=False, **kwargs):
        if fpath is None:
            self._path = None
        else:
            self._path = expand_path(fpath)
        self._dtype = dtype
        # initialize the file writer and the buffer
        self._buffer = BufferQueue(dtype)
        self._init_file(overwrite)

    def _init_file(self, overwrite):
        """
        Open the correct target (file or stdout) an create a CSV writer.

        Parameters:
        -----------
        overwrite : bool
            Whether the output file is overwritten if it exists
        """
        # initialize redirection to stdout if no file path is provided
        if self._path is None:
            self._file = stdout
            self._buffer.buffersize = 1024  # one kByte
        else:
            self._check_overwrite(self._path, overwrite)
            self._file = open(self._path, "w")
        self._writer = csv.writer(self._file)
        # write the header
        self._writer.writerow(self.dtype.names)

    def _write(self, data):
        """
        Write rows to the end of the output file

        Parameters:
        -----------
        data : numpy.array
            Data table with column names and correct data types.
        """
        for i in range(len(data)):
            self._writer.writerow(data[i])
        # update the current length
        self._len += len(data)

    def close(self):
        """
        Flush the buffer and close the file.
        """
        if self._file is stdout:
            stdout.flush()
        else:
            self.flush()
            self._file.close()


########################  optionally supported formats  #######################

try:
    from fitsio import FITS

    @register("fits", ("fit", "fits"))
    class FITSreader(Reader):
        """
        Read from single extension of a FITS file into numpy arrays using a
        buffer.

        Parameters:
        -----------
        fpath : str
            Path to the FITS file.
        ext : int
            Index of the FITS extension to read from
        """

        def __init__(self, fpath, ext=1, **kwargs):
            self._path = expand_path(fpath)
            self._ext = ext
            self._file = FITS(fpath)
            # select the file extension and verify that it contains binary data
            self._reader = self._file[ext]
            if self._reader.get_exttype() != "BINARY_TBL":
                message = "Fits extension {:d} is not a binary table"
                raise TypeError(message.format(ext))
            self._get_dtype()
            self._len = self._reader.get_nrows()
            # allocate the buffer (which we don't really utilize)
            self._buffer = BufferQueue(self.dtype)

        def _get_dtype(self):
            """
            Read the data types directly off the FITS header. Change the byte
            order to the architecture native (FITS is big endian).
            """
            dtypes = self._reader.get_rec_dtype()[0]  # data types from header
            # use the converter class to get the types with correct byte order
            native = "<" if sys.byteorder == "little" else ">"
            dtype_list = []
            for name in dtypes.names:
                dtype = dtypes[name]
                # apparently, fitsio silently converts byte strings to unicode
                # so the type casting will fail if we do not expect unicode
                if "S" in dtype.str:
                    endian, count = dtype.str.split("S")
                    dtype = np.dtype("{:s}U{:s}".format(native, count))
                dtype_list.append((name, ConvertByteOrder(dtype).dtype))
            self._dtype = np.dtype(dtype_list)

        def read_chunk(self) -> np.array:
            """
            Read a chunk from the input file until the buffer is full or the
            end of the file is reached.

            Returns:
            --------
            buffer : numpy.array
                Data read from disk, packed in an array with fields named like
                the data columns.
            """
            if self._file.mode is None:
                raise ValueError("I/O operation on closed file.")
            # check if the end of the file has been reached
            if self._current_row == self._len:
                raise EOFError("reached end of file")
            # determine the index range and read the next block of data
            start = self._current_row
            end = min(self._len, start + self.bufferlength)
            buffer = self._reader[start:end]
            self._current_row = end
            # correct the byte order directly (without the converters), since
            # the raw data is already contiguous
            return buffer.astype(self._dtype, casting="same_kind", copy=False)

        def reopen(self):
            """
            Reset the row counter and reopen the file.
            """
            self._current_row = 0
            self.close()
            self._file = FITS(self._path)
            self._reader = self._file[self._ext]

        def close(self):
            """
            Close the underlying file pointer.
            """
            if self._file.mode is not None:
                self._file.reopen()
                self._file.close()


    @register("fits", ("fit", "fits"))
    class FITSwriter(Writer):
        """
        Write table data to the first extension of a FITS file using a buffer.

        Parameters:
        -----------
        dtype : numpy.dtype
            Describes the expected column names and data types.
        fpath : str
            Path to the FITS file.
        overwrite : bool
            Whether the output file is overwritten if it exists.
        """

        def __init__(self, dtype, fpath, overwrite=False, **kwargs):
            self._path = expand_path(fpath)
            self._check_overwrite(fpath, overwrite)
            self._dtype = dtype
            # initialize the file writer and the buffer
            self._buffer = BufferQueue(dtype)
            self._init_file()

        def _init_file(self):
            """
            Open the correct target file an create a FITS table writer.
            """
            self._file = FITS(self._path, mode="rw")

        def _write(self, data):
            """
            Write rows to the end of the output file

            Parameters:
            -----------
            data : numpy.array
                Data table with column names and correct data types.
            """
            if self._len == 0:
                self._file.write(data)
            else:
                self._file[1].append(data)
            # update the current length
            self._len += len(data)

        def write_history(self, line):
            """
            Write a history stemp to the FITS header.
            """
            raise NotImplementedError
            self._file[1].write_history(line)
            self._file.reopen()  # flush data

        def close(self):
            """
            Flush the buffer and close the file.
            """
            self.flush()  # flush buffer
            #self._file.reopen()  # flush low level buffers
            self._file.close()

except ImportError:
    pass


try:
    import h5py


    @register("hdf5", ("h5", "hdf", "hdf5"))
    class HDF5reader(Reader):
        """
        Read a collection of equal length data sets from a HDF5 file into numpy
        arrays using a buffer.

        Parameters:
        -----------
        fpath : str
            Path to the FITS file.
        datasets : array like
            List of a subset of (equal length) data sets in the HDF5 file to
            read. If not provided, all data sets are used.
        """

        def __init__(self, fpath, datasets=None, **kwargs):
            self._path = expand_path(fpath)
            self._file = h5py.File(fpath, mode="r")
            self._closed = False
            self._init_datasets(datasets)
            self._check_dataset_lengths()
            self._get_dtype()
            # allocate the buffer (which we don't really utilize)
            self._buffer = BufferQueue(self.dtype)

        def _init_datasets(self, dataset_list):
            """
            Automatically find data sets in the HDF5 file or check the provided
            input list for groups.

            Parameters:
            -----------
            dataset_list : array like
                List of a subset of (equal length) data sets in the HDF5 file
                to read.
            """
            if dataset_list is None:
                # find all data sets in the file
                dataset_list = []
                self._file.visit(dataset_list.append)
                # discard all groups
                self._datasets = {}
                for path in dataset_list:
                    entry = self._file[path]
                    if type(entry) is not h5py.Group:
                        self._datasets[path] = entry
            else:
                # check if the provided data sets actually exist
                self._datasets = {}
                for path in dataset_list:
                    if type(path) is not str:
                        path = path[0]  # remove the optional type hint
                    entry = self._file[path]
                    # check that no groups entered the selection
                    if type(entry) is h5py.Group:
                        message = "input path is a group: {:}".format(path)
                        raise TypeError(message)
                    self._datasets[path] = entry

        def _check_dataset_lengths(self):
            """
            Get the length of the data sets and verify that they all have the
            same length.
            """
            self._len = None
            for path, dset in self._datasets.items():
                # check that all data sets have the same length
                if self._len is None:
                    self._len = len(dset)
                elif len(dset) != self._len:
                    message = "length of data set'{:}' does not match "
                    message += "common length {:d}"
                    raise ValueError(message.format(path, self._len))
            # in case all sets are empty
            if self._len is None:
                self._len = 0

        def _get_dtype(self):
            """
            Get the data types from the data sets and change the byte order to
            the architecture native.
            """
            self._converters = OrderedDict(
                (path, ConvertByteOrder(dset.dtype))
                for path, dset in self._datasets.items())
            # get the native data types form the converter instances
            self._dtype = np.dtype([
                (path, converter.dtype)
                for path, converter in self._converters.items()])

        def read_chunk(self) -> np.array:
            """
            Read a chunk from the input file until the buffer is full or the
            end of the file is reached.

            Returns:
            --------
            buffer : numpy.array
                Data read from disk, packed in an array with fields named like
                the data columns.
            """
            if self._closed:
                raise ValueError("I/O operation on closed file.")
            # check if the end of the file has been reached
            if self._current_row == self._len:
                raise EOFError("reached end of file")
            # determine the index range for the next block of data
            start = self._current_row
            end = min(self._len, start + self.bufferlength)
            n_data = end - start
            buffer = np.empty(n_data, dtype=self.dtype)
            # read from all data sets into the buffer and correct byte order
            for path, converter in self._converters.items():
                raw_data = self._file[path][start:end]
                buffer[path] = converter(raw_data)
            self._current_row = end
            return buffer

        def reopen(self):
            """
            Reset the row counter and reopen
            """
            self._current_row = 0
            self._file.close()
            self._file = h5py.File(self._path, mode="r")
            self._closed = False

        def close(self):
            """
            Close the underlying file pointer.
            """
            self._file.close()
            self._closed = True


    @register("hdf5", ("h5", "hdf", "hdf5"))
    class HDF5writer(Writer):
        """
        Write table data to a HDF5 file using a buffer. Each table column is
        represented by a data set in the output file.

        Parameters:
        -----------
        dtype : numpy.dtype
            Describes the expected column names and data types.
        fpath : str
            Path to the HDF5 file.
        overwrite : bool
            Whether the output file is overwritten if it exists.
        compression : str
            Compression filter to use for all data sets, must be either of
            "none" (disabled), "lzf" (default) or "gzip".
        hdf5_shuffle : bool
            Apply the shuffle filter prior to compression.
        hdf5_checksum : bool
            Write fletcher32 check sums to data blocks.
        """

        def __init__(
                self, dtype, fpath, overwrite=False, compression=None,
                hdf5_shuffle=True, hdf5_checksum=True, **kwargs):
            self._path = expand_path(fpath)
            self._check_overwrite(fpath, overwrite)
            self._dtype = dtype
            # initialize the file writer and the buffer
            self._buffer = BufferQueue(dtype)
            self._init_file(compression, hdf5_shuffle, hdf5_checksum)

        def _init_file(self, compression, hdf5_shuffle, hdf5_checksum):
            """
            Open the ouput file and create a data set for each column.
            """
            self._file = h5py.File(self._path, mode="w-")
            # initialize the required datasets
            compression = "lzf" if compression is None else compression
            for name in self.dtype.names:
                self._file.create_dataset(
                    name, dtype=self.dtype[name].str, shape=(0,),
                    chunks=True, maxshape=(None,), shuffle=hdf5_shuffle,
                    compression=compression, fletcher32=hdf5_checksum)

        def _write(self, data):
            """
            Write rows to the end of the output file

            Parameters:
            -----------
            data : numpy.array
                Data table with column names and correct data types.
            """
            # write the chunk column by column
            start = self._len
            end = start + len(data)
            for name in data.dtype.names:
                dset = self._file[name]
                # grow the data sets
                dset.resize((end,))
                dset[start:end] = data[name]
            # update the current length
            self._len = end

        def close(self):
            """
            Flush the buffer and close the file.
            """
            self.flush()
            self._file.close()

except ImportError:
    pass


try:
    import pyarrow as pa
    from pyarrow import parquet as pq


    @register("parquet", ("parquet", "pqt"))
    class PARQUETreader(Reader):
        """
        Read a parquet file into numpy arrays using a buffer.

        Parameters:
        -----------
        fpath : str
            Path to the CSV file.
        """

        _row_group_idx = 0

        def __init__(self, fpath, **kwargs):
            self._path = expand_path(fpath)
            self._file = pq.ParquetFile(fpath)
            self._closed = False
            self._get_dtype()
            self._len = self._file.metadata.num_rows
            # data is organized and read in groups of arbitrary size 
            self._n_row_groups = self._file.metadata.num_row_groups
            self._buffer = BufferQueue(self.dtype)

        def _get_dtype(self):
            """
            Construct the data type from the (py)arrow table schema.
            """
            schema = self._file.schema_arrow
            dtypes = []
            for name, pyarrow_dtype in zip(schema.names, schema.types):
                # get the equivalent data type representation in numpy
                dtypes.append((name, pyarrow_dtype.to_pandas_dtype()))
            self._dtype = np.dtype(dtypes)

        def _reached_EOF(self) -> bool:
            """
            Whether the last row group has been read.
            """
            return self._row_group_idx == self._n_row_groups

        def read_chunk(self, chunksize=None) -> np.array:
            """
            Read a data from the buffer until it runs empty. Dynamically load
            more data from disk until the end of the file is reached.

            Returns:
            --------
            buffer : numpy.array
                Data read from disk, packed in an array with fields named like
                the data columns.
            """
            if self._closed:
                raise ValueError("I/O operation on closed file.")
            # there is enough data in the buffer
            if self._buffer.full:
                buffer = self._buffer.pop()
            # at EOF: return remaining data in buffer or stop iteration
            elif self._reached_EOF():
                if not self._buffer.empty:
                    buffer = self._buffer.clear()
                else:
                    raise EOFError("reached end of file")
            # read row groups until the buffer is full or EOF is reached
            else:
                while not self._buffer.full and not self._reached_EOF():
                    raw_data = self._file.read_row_group(self._row_group_idx)
                    self._row_group_idx += 1
                    # convert to numpy record array
                    row_group = np.empty(len(raw_data), dtype=self.dtype)
                    for name in self._dtype.names:
                        row_group[name] = raw_data[name]
                    self._buffer.push(row_group)
                # collect the data to return
                if self._reached_EOF() and not self._buffer.full:
                    buffer = self._buffer.clear()
                else:
                    buffer = self._buffer.pop()
            self._current_row += len(buffer)
            return buffer

        def reopen(self):
            """
            Reset the row counter.
            """
            self._buffer.clear()
            self._current_row = 0
            self._row_group_idx = 0
            self._closed = False

        def close(self):
            """
            no close() required/implemented in ParquetFile
            """
            self._closed = True


    @register("parquet", ("parquet", "pqt"))
    class PARQUETwriter(Writer):
        """
        Write table data to a parquet file using a buffer.

        Parameters:
        -----------
        dtype : numpy.dtype
            Describes the expected column names and data types.
        fpath : str
            Path to the parquet file.
        overwrite : bool
            Whether the output file is overwritten if it exists.
        compression : str
            Compression filter to use for all data sets, must be either of
            "none" (disabled), "snappy" (default), "gzip", "lzo", "brotli",
            "lz4" or "zstd" (list may differ on systems).
        """

        def __init__(
                self, dtype, fpath, overwrite=False,
                compression=None, **kwargs):
            self._path = expand_path(fpath)
            self._check_overwrite(fpath, overwrite)
            self._dtype = dtype
            # we cannot open the file yet, create a dummy instead
            self._file = open(fpath, "wb")
            # initialize the file writer and the buffer
            self._buffer = BufferQueue(dtype)
            self.buffersize = 512 * _mega_byte  # recommended value
            self._init_file(compression)

        def _schema_from_dtype(self, dtype):
            """
            Convert a numpy dtype object to an equivalent pyarrow schema.

            Parameters:
            -----------
            dtype : numpy.dtype
                Describes the data column names and data types.
            """
            dummy_table = pa.Table.from_pydict({
                name: np.array([], dtype[name]) for name in dtype.names})
            self._schema = dummy_table.schema

        def _init_file(self, compression):
            """
            Open the ouput file and create the table meta data.
            """
            self._schema_from_dtype(self.dtype)
            # create the parquet writer
            fpath = self._file.name
            self._file.close()
            compr_mode = "NONE" if compression is None else compression.upper()
            self._file = pq.ParquetWriter(
                fpath, self._schema, compression=compr_mode)

        def _write(self, data):
            """
            Write rows to the end of the output file

            Parameters:
            -----------
            data : numpy.array
                Data table with column names and correct data types.
            """
            # convert to pyarrow table
            array_list = [pa.array(data[name]) for name in data.dtype.names]
            pyarrow_table = pa.Table.from_arrays(
                array_list, schema=self._schema)
            # write the chunk
            self._file.write_table(pyarrow_table)
            # update the current length
            self._len += len(data)

        def close(self):
            """
            Flush the buffer and close the file.
            """
            self.flush()
            self._file.close()

except ImportError:
    pass


############################  convenience functions  ##########################

def create_reader(input, format, col_map_dict, fits_ext):
    # automatically determine the input file format
    if format is None:
        try:
            format = guess_format(input)
        except NotImplementedError as e:
            logger.exception(str(e))
            raise
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
        logger.exception(str(e))
        raise
    # notify about buffer size
    message = "buffer size: {:.2f} MB ({:,d} rows)".format(
        reader.buffersize / 1024**2, reader.bufferlength)
    logger.debug(message)
    return reader


def create_writer(
        output, format, dtype, compression, hdf5_shuffle, hdf5_checksum):
    # automatically determine the output file format
    if output is None:  # write to stdout in csv format
        format = "csv"
    if format is None:
        try:
            format = guess_format(output)
        except NotImplementedError as e:
            logger.exception(str(e))
            raise
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
        logger.exception(str(e))
        raise
    # determine an automatic buffer/chunk size
    message = "buffer size: {:} ({:,d} rows)".format(
        bytesize_with_prefix(writer.buffersize), writer.bufferlength)
    logger.debug(message)
    return writer
