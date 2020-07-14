import multiprocessing as mp
import os
import sys
from itertools import repeat

from memmap_table.column import MemmapColumn
from memmap_table.table import MemmapTable


# initialize a global counter that keeps track of the number of rows processed
_ROW_PROGRESS = None


class TableColumn(object):
    """
    Container for a table column name used to distinguish from ordinary string
    arguments in ParallelTable.
    """

    def __init__(self, tablepath, colname):
        self._path = os.path.join(tablepath, colname)
        self._name = colname
    
    def __repr__(self):
        return "table[{:}]".format(self._name)

    @property
    def name(self):
        """
        Get the name of the column
        """
        return self._name

    @property
    def path(self):
        """
        Get the file path of the column
        """
        return self._path


class ParallelIterator(object):
    """
    Iterator over a chunk of a data table for parallel processing. Progress is
    synchronized between threads and reported

    Parameters:
    -----------
    start : int
        Row index at which the thread starts.
    end : int
        Row index where the next thread takes over
    step : int
        Load and process chunks of length step.
    total : int
        Total number of rows in the table.
    """

    def __init__(self, start, end, step, total):
        self.start = start
        self.end = end
        self.step = step
        self.total = total

    def __len__(self):
        """
        Number of rows processed by this iterator
        """
        return self.end - self.start

    def __iter__(self):
        """
        Iterate a given range [start, end] in chunks of fixed step size. The
        progress in reported and updated an screen synchronized between all
        concurrent threads.

        Yields:
        -------
        start : int
            First index of the current chunk.
        end : int
            Index past the last index of the current chunk.
        """
        line_message = "progress: {:6.1%}\r"
        # provide an index reach for the next chunk to process
        for start in range(self.start, self.end, self.step):
            end = min(self.end, start + self.step)
            yield start, end
            # update the global progress counter and refresh the command line
            with _ROW_PROGRESS.get_lock():
                _ROW_PROGRESS.value += end - start
                sys.stdout.write(
                    line_message.format(_ROW_PROGRESS.value / self.total))
                sys.stdout.flush()


class ParallelTable(object):
    """
    Wrapper to apply a function to the columns of a MemmapTable with concurrent
    threads. For each thread the table rows are divided into equal sized chunks
    to avoid current I/O operations on the same memory.

    Parameters:
    -----------
    table : memmap_table.MemmapTable
        Data table object to process.
    logger : python logger instance
        Enables optional event logging.
    """

    _chunksize = 16384  # default in current MemmapTable implementation
    _worker_function = None
    _call_args = None
    _call_kwargs = None
    _return_map = None

    def __init__(self, table, logger=None):
        # keep the table reference for later use
        self._table = table
        self._logger = logger

    @staticmethod
    def _n_threads(n_threads=None):
        """
        Check and normalize the provided number of threads.

        Parameters:
        -----------
        n_threads : int or None:
            The suggested number of threads to use, if None, use all available.

        Returns:
        --------
        threads : int
            The number of threads to use.
        """
        max_threads = mp.cpu_count()
        if n_threads is None:
            threads = max_threads
        elif 1 <= int(n_threads):
            threads = min(int(n_threads), max_threads)
        else:
            message = "number of threads must be in range 1-{:d}"
            raise ValueError(message.format(max_threads))
        return threads

    def _thread_iterator(self, n_threads):
        """
        Generate a list of iterators that walk through non-overlapping row
        groups of the data table.

        Parameters:
        -----------
        n_threads : int
            Number of threads on which to distribute the load.

        Returns:
        --------
        idx_range : list
            The list of the .
        """
        # average range of row assigned to a thread
        stepsize = float(len(self._table)) / n_threads
        idx_range = []
        # compute the fractional limit for each range and round to the next
        # lower index
        idx = stepsize
        while int(idx) <= len(self._table):
            iterator = ParallelIterator(
                int(idx - stepsize), int(idx),
                self.chunksize, len(self._table))
            idx_range.append(iterator)
            idx += stepsize
        # correct rounding errors
        idx_range[-1].end = len(self._table)
        return idx_range

    @property
    def chunksize(self) -> int:
        """
        Current number of rows processed at once between I/O operations.
        """
        return self._chunksize

    @chunksize.setter
    def chunksize(self, chunksize):
        """
        Set the number of rows processed at once between I/O operations. A
        larger chunksize will increase the memory requirement but also increase
        the performance by fewer concurrent I/O operations.
        """
        self._chunksize = int(chunksize)

    def set_worker(self, function):
        """
        Set a worker function that is applied in threads to chunks of the. The
        signature must be provided, i.e. the arguments must be assigned column
        names or constant parameters and the return values must map back to
        column names as well.

        Parameters:
        -----------
        function : object
            Worker function, which must be callable and support pickeling.
        """
        if not callable(function):
            raise TypeError("worker function is not callable")
        self._worker_function = function

    def _add_argument(self, arg, keyword):
        """
        Register a new argument for the worker function.

        Parameters:
        -----------
        arg : object
            Value that is provided when calling the function.
        keyword : str or None
            If keyword is a string, add the argument value as keyword argument.
        """
        # initialize the argument collection
        if self._call_args is None:
            self._call_args = []
        if self._call_kwargs is None:
            self._call_kwargs = {}
        # assign as positional or keyword argument
        if keyword is None:
            self._call_args.append(arg)
        else:
            self._call_kwargs[keyword] = arg

    def add_argument_column(self, colname, keyword=None):
        """
        Add an input argument for the worker function which is provided by a
        table column. All arguments must be registered in order or provide a
        keyword name.

        Parameters:
        -----------
        colname : str
            Name of the column which provides the values for this argument.
        keyword : str
            Keyword name of the argument in the function signature (optional).
        """
        # verify that the table contains the input column
        if type(colname) is not str:
            raise TypeError("column name must be string")
        elif colname not in self._table.colnames:
            message = "table does not contain column: {:}"
            raise KeyError(message.format(colname))
        self._add_argument(TableColumn(self._table.root, colname), keyword)

    def add_argument_constant(self, value, keyword=None):
        """
        Add a constant input argument for the worker function. All arguments
        must be registered in order or provide a keyword name.

        Parameters:
        -----------
        value : object
            Value provided to the worker function. Must support pickeling.
        keyword : str
            Keyword name of the argument in the function signature (optional).
        """
        self._add_argument(value, keyword)

    def add_result_column(self, colname):
        """
        Add an output table column that receives results from the worker
        function. All arguments must be registered in order or provide a
        keyword name.

        Parameters:
        -----------
        colname : str
            Name of the column which provides the values for this argument.
        """
        if self._return_map is None:
            self._return_map = []
        # check the input data type and whether the table contains all output
        # columns
        if type(colname) is not str:
            raise TypeError("column name must be string")
        elif colname not in self._table.colnames:
            message = "table does not contain column: {:}"
            raise KeyError(message.format(colname))
        self._return_map.append(TableColumn(self._table.root, colname))

    @property
    def signature(self):
        """
        String representing the call signature:
            worker_function(*args, **kwargs) -> results
        """
        function = self._worker_function
        # collect the function arguments
        args = [str(arg) for arg in self._call_args]
        args.extend(
            "{:}={:}".format(val, str(key))
            for val, key in self._call_kwargs.items())
        args = ", ".join(args)
        # collect the return values
        result = ", ".join(str(res) for res in self._return_map)
        # build the signature string
        sign = "{:}({:}) -> {:}".format(function.__name__, args, result)
        return sign

    def execute(self, n_threads=None):
        """
        Apply the worker function on the input table using a given number of
        threads and writing the results to the output columns.

        Parameters:
        -----------
        n_threads : int
            Number of parallel threads to use (all by default).
        """
        threads = self._n_threads(n_threads)
        # initialize the global row progress counter
        global _ROW_PROGRESS
        _ROW_PROGRESS = mp.Value('i', 0)
        # assign row subsets to the threads
        n_rows = len(self._table)
        index_ranges = self._thread_iterator(
            # there cannot be more threads than table rows
            n_rows if threads > n_rows else threads)
        threads = len(index_ranges)
        # collect the call arguments for the worker
        worker_args = list(zip(
            index_ranges, repeat(self._worker_function),
            repeat(self._call_args), repeat(self._call_kwargs),
            repeat(self._return_map)))
        # notify begin of processing
        if self._logger is not None:
            if threads > 1:
                message = "processing input stream using {:d} threads ..."
                message = message.format(threads)
            else:
                message = "processing input stream ..."
            self._logger.info(message)
        # create the worker pool
        with mp.Pool(threads) as pool:
            sys.stdout.write("progress: {:6.1%}\r".format(0.0))
            sys.stdout.flush()
            pool.map(_thread_worker, worker_args)
        # reset row counter for future use
        _ROW_PROGRESS = None


def _thread_worker(wrapper_args):
    """
    Wrapper around the worker function that is running in each thread spawned
    by ParallelTable. Based on the signature registered in ParallelTable, data
    is loaded from the table columns, processed and written to the appropriate
    output column(s).

    Parameters:
    -----------
    wrapper_args : list
        ParallelIterator : manages the chunkwise loading of data
        callable : the worker function
        list : the positional function arguments
        dict : the function keyword arguments
        list : the column(s) where the function return values are stored
    """
    # unpack all input arguments
    iterator, function, args, kwargs, results = wrapper_args
    # open the data sets from the table are needed
    # process the positional arguments
    args_expanded = []
    for arg in args:
        if type(arg) is TableColumn:
            column = MemmapColumn(arg.path, mode="r")
            args_expanded.append(column)
        else:
            args_expanded.append(arg)
    # process the keyword arguements
    kwargs_expanded = {}
    for key, arg in kwargs.items():
        # this does not load any data yet
        if type(arg) is TableColumn:
            column = MemmapColumn(arg.path, mode="r")
            kwargs_expanded[key] = column
        else:
            kwargs_expanded[key] = arg
    # process the results
    results_expanded = []
    for result in results:
        column = MemmapColumn(result.path, mode="r+")
        results_expanded.append(column)

    # apply the worker function
    for start, end in iterator:
        # load the actual values
        call_args = [
            arg[start:end] if type(arg) is MemmapColumn else arg
            for arg in args_expanded]
        call_kwargs = {
            key: arg[start:end] if type(arg) is MemmapColumn else arg
            for key, arg in kwargs_expanded.items()}
        # execute worker
        return_values = function(*call_args, **call_kwargs)
        # map back and write results
        if len(results_expanded) == 1:
            results_expanded[0][start:end] = return_values
        else:
            for result, values in zip(results_expanded, return_values):
                result[start:end] = values
