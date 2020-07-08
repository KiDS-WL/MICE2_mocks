import csv
import os
import warnings
from collections import OrderedDict

import numpy as np
from fitsio import FITS


class reader(object):

    _file = None
    _dtype = None
    _len = 0
    _current_row = 0
    _buffersize = 104857600  # default of 100 MB

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
        return os.path.getsize(self._file.name)

    def _get_buffer_length(self):
        bytes_per_row = self._dtype.itemsize
        return max(1, self._buffersize // bytes_per_row)

    def read_chunk(self):
        return NotImplemented

    def close(self):
        return NotImplemented


class CSVreader(reader):

    def __init__(self, fpath, *args):
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
        chunk = np.zeros(chunksize, dtype=self.dtype)
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


class FITSreader(reader):

    def __init__(self, fpath, ext=1, *args):
        self._file = FITS(fpath)
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


if __name__ == "__main__":
    
    basepath = "../../MICE2_256th_uBgVrRciIcYJHKs_shapes_halos_WL"

    with CSVreader(basepath + ".csv") as reader:
        print(reader)
        l = 0
        for chunk in reader:
            l += len(chunk)
        print("predicted length:", len(reader))
        print("true length:     ", l)
        if l == len(reader):
            print("margin:          ", "exact")
        else:
            print("margin:          ", "{:.1%}".format((len(reader) - l) / l))

    with FITSreader(basepath + ".fits") as reader:
        print(reader)
        l = 0
        for chunk in reader:
            l += len(chunk)
        print("predicted length:", len(reader))
        print("true length:     ", l)
        if l == len(reader):
            print("margin:          ", "exact")
        else:
            print("margin:          ", "{:.1%}".format((len(reader) - l) / l))
