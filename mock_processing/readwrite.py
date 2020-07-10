import csv
import os
import warnings
from sys import stdout

import numpy as np


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


def guess_format(path):
    path, ext = os.path.splitext(path.strip())
    ext = ext.lower().lstrip(".")
    # compare with all supported file types
    for format_key, aliases in extension_alias.items():
        if ext in aliases:
            return format_key
    raise NotImplementedError(
        "unsupported file format with extension: {:}".format(ext))



################################  base classes  ###############################

class Reader(object):

    _file = None
    _dtype = None
    _len = 0
    _current_row = 0
    _buffersize = BUFFERSIZE

    def __len__(self):
        return self._len

    def __iter__(self):
        while True:
            try:
                yield self.read_chunk()
            except EOFError:
                break

    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()

    @property
    def dtype(self):
        return self._dtype

    @property
    def current_row(self):
        return self._current_row

    @property
    def buffersize(self):
        return self._buffersize
    
    @buffersize.setter
    def buffersize(self, nbytes):
        self._buffersize = nbytes

    @property
    def filesize(self):
        try:
            return os.path.getsize(self._file.name)
        except AttributeError:
            return os.path.getsize(self._file.filename)

    def _get_buffer_length(self):
        bytes_per_row = self._dtype.itemsize
        return max(1, self._buffersize // bytes_per_row)

    def read_chunk(self):
        return NotImplemented

    def close(self):
        return NotImplemented


class Writer(object):

    _file = None
    _dtype = None
    _len = 0

    def __len__(self):
        return self._len

    def __enter__(self, *args, **kwargs):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()

    @property
    def dtype(self):
        return self._dtype

    @property
    def filesize(self):
        return os.path.getsize(self._file.name)

    def write_chunk(self):
        return NotImplemented

    def close(self):
        return NotImplemented


class CSVreader(Reader):

    def __init__(self, fpath, **kwargs):
        self._file = open(fpath)
        try:
            # read the first 10 kB to analyse the file
            text_sample = self._file.read(10240)
            # identify the CSV dialect and whether it has a header
            sniffer = csv.Sniffer()
            self._dialect = sniffer.sniff(text_sample)
            self._has_header = sniffer.has_header(text_sample)
            # read the first data row to determine its data type
            self._file.seek(0)
            tmp_reader = csv.reader(self._file, self._dialect)
            colnames = next(tmp_reader)
            row = next(tmp_reader)
            if not self._has_header:
                colnames = ["column{:d}".format(i) for i in range(len(row))]
            # try to identify a numpy data type for each column
            dtypes = []
            for colname, entry in zip(colnames, row):
                try:
                    int(entry)
                    dtypes.append((colname, np.dtype(np.int).str))
                    continue
                except ValueError:
                    pass
                try:
                    float(entry)
                    dtypes.append((colname, np.dtype(np.float).str))
                    continue
                except ValueError:
                    dtypes.append((colname, "|S{:d}".format(len(entry))))
                    # issue a warning that long data might be truncated
                    message = "column '{:}' is interpreted as length "
                    message += "{:d} string, longer entries will be truncated"
                    warnings.warn(message.format(colname, len(entry)))
            self._dtype = np.dtype(dtypes)
            # guess number of rows with a 5% safety margin
            self._reopen()
            i = 0
            # read the first chunk and assume it is representative
            line_bytes = []
            while i < self._get_buffer_length():
                row = next(self._reader)
                # make a guess for the row length, assuming no quoting is used
                line_bytes.append(
                    sum(len(entry) for entry in row) +
                    len(self._dialect.delimiter) * (len(row) - 1) +
                    len(self._dialect.lineterminator))
                i += 1
            mean_line_bytes = sum(line_bytes) / len(line_bytes)
            self._len = int(self.filesize / mean_line_bytes * 1.05)
            # open the final reader object
            self._reopen()
        except Exception as e:
            self._file.close()
            raise e

    def _reopen(self):
        self._file.seek(0)
        self._reader = csv.reader(self._file, self._dialect)
        if self._has_header:
            next(self._reader)

    def read_chunk(self, chunksize=None):
        if chunksize is None:
            chunksize = self._get_buffer_length()
        chunk = np.empty(chunksize, dtype=self.dtype)
        i = 0
        while i < chunksize:
            try:
                row = next(self._reader)
                chunk[i] = tuple(row)
                i += 1
            except StopIteration:
                if i > 0:
                    break  # return the remaining rows
                else:
                    raise EOFError("reached the end of the file")
            except ValueError:
                message = "row {:d} with length {:d} does not match header "
                message += "schema with length {:d}"
                raise ValueError(message.format(i, len(row), len(self.dtype)))
        self._current_row += i
        return chunk[:i]  # truncate if EOF is reached

    def close(self):
        self._file.close()


#########################  CSV from standard library  #########################

class CSVwriter(Writer):

    _header_written = False

    def __init__(self, fpath=None, overwrite=False):
        if fpath is None:
            self._file = stdout
        else:
            if os.path.exists(fpath):
                if overwrite:
                    os.remove(fpath)
                else:
                    raise OSError("output file '{:}' exists".format(fpath))
            self._file = open(fpath, "a")
        self._writer = csv.writer(self._file)

    def write_chunk(self, table):
        # store/check the input data type and write the header
        if not self._header_written:
            self._dtype = table.dtype
            # write the header
            self._writer.writerow(self.dtype.names)
            self._header_written = True
        else:
            if self._dtype != table.dtype:
                raise TypeError(
                    "input data type does not match previous records")
        # write the chunk
        for i in range(len(table)):
            self._writer.writerow(table[i])
        # update the current length
        self._len += len(table)

    def close(self):
        stdout.flush()
        if self._file is not stdout:
            self._file.close()


extension_alias = {"csv": ("csv",)}
supported_readers = {"csv": CSVreader}
supported_writers = {"csv": CSVwriter}


########################  optionally supported formats  #######################

try:
    from fitsio import FITS


    class FITSreader(Reader):

        def __init__(self, fpath, ext=1, **kwargs):
            self._file = FITS(fpath)
            self._file.filename = fpath
            # figure out the data format and check the resources
            self._reader = self._file[ext]
            if self._reader.get_exttype() != "BINARY_TBL":
                message = "Fits extension {:d} is not a binary table"
                raise TypeError(message.format(ext))
            self._dtype = self._reader.get_rec_dtype()[0]
            # keep track of the current position
            self._len = self._reader.get_nrows()

        def read_chunk(self, chunksize=None):
            # check if the end of the file has been reached
            if self._current_row == self._len:
                raise EOFError("reached the end of the file")
            # determine the next data slice
            if chunksize is None:
                chunksize = self._get_buffer_length()
            start = self._current_row
            end = min(self._len, start + chunksize)
            self._current_row = end
            return self._reader[start:end]

        def close(self):
            self._file.close()


    class FITSwriter(Writer):

        def __init__(self, fpath, overwrite=False):
            if os.path.exists(fpath):
                if overwrite:
                    os.remove(fpath)
                else:
                    raise OSError("output file '{:}' exists".format(fpath))
            self._file = FITS(fpath, mode="rw")

        def write_chunk(self, table):
            if len(table) == 0:
                return
            # store/check the input data type
            if self._len == 0:
                self._dtype = table.dtype
            else:
                if self._dtype != table.dtype:
                    raise TypeError(
                        "input data type does not match previous records")
            # write the chunk
            if self._len == 0:
                self._file.write(table)
            else:
                self._file[1].append(table)
            self._file.reopen()  # flush data
            # update the current length
            self._len += len(table)

        def write_history(self, line):
            self._file[1].write_history(line)
            self._file.reopen()  # flush data

        def close(self):
            self._file.close()


    extension_alias["fits"] = ("fit", "fits")
    supported_readers["fits"] = FITSreader
    supported_writers["fits"] = FITSwriter

except ImportError:
    pass


try:
    import h5py


    class HDF5reader(Reader):

        def __init__(self, fpath, **kwargs):
            self._file = h5py.File(fpath, mode="r")
            # discover all data sets
            dsets = []
            self._file.visit(dsets.append)
            # determine the table data type
            dtypes = []
            self._len = None
            for colname in dsets:
                entry = self._file[colname]
                if type(entry) is not h5py.Group:
                    dtypes.append((colname, entry.dtype.str))
                    # check that all data sets have the same length
                    if self._len is None:
                        self._len = len(entry)
                    elif len(entry) != self._len:
                        message = "length of data set'{:}' does not match common "
                        message += "length {:d}"
                        raise ValueError(message.format(colname, self._len))
            self._dtype = np.dtype(dtypes)

        def read_chunk(self, chunksize=None):
            # check if the end of the file has been reached
            if self._current_row == self._len:
                raise EOFError("reached the end of the file")
            # determine the next data slice
            if chunksize is None:
                chunksize = self._get_buffer_length()
            start = self._current_row
            end = min(self._len, start + chunksize)
            # read from all data sets
            chunk = np.empty(end - start, dtype=self.dtype)
            for name in self._dtype.names:
                chunk[name] = self._file[name][start:end]
            self._current_row = end
            return chunk

        def close(self):
            self._file.close()


    class HDF5writer(Writer):

        def __init__(self, fpath=None, overwrite=False):
            if os.path.exists(fpath):
                if overwrite:
                    os.remove(fpath)
                else:
                    raise OSError("output file '{:}' exists".format(fpath))
            self._file = h5py.File(fpath, mode="w-")

        def write_chunk(self, table):
            if len(table) == 0:
                return
            # store/check the input data type
            if self._len == 0:
                self._dtype = table.dtype
                # if the file is uninitialized, create the required datasets
                for name in self._dtype.fields:
                    self._file.create_dataset(
                        name, dtype=self._dtype[name].str, shape=(len(table),),
                        chunks=True, maxshape=(None,), shuffle=True,
                        compression="lzf", fletcher32=True)
            else:
                if self._dtype != table.dtype:
                    raise TypeError(
                        "input data type does not match previous records")
            # write the chunk field by field
            start = self._len
            end = start + len(table)
            for name in table.dtype.names:
                dset = self._file[name]
                # grow the data sets
                dset.resize((end,))
                dset[start:end] = table[name]
            # update the current length
            self._len = end

        def close(self):
            self._file.close()


    extension_alias["hdf5"] = ("h5", "hdf", "hdf5")
    supported_readers["hdf5"] = HDF5reader
    supported_writers["hdf5"] = HDF5writer

except ImportError:
    pass
