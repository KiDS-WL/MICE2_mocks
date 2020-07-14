import multiprocessing as mp


class ParallelTable(object):

    def __init__(self, table, threads=None):
        max_threads = mp.cpu_count()
        if threads is None:
            self._threads = max_threads
        elif 1 <= int(threads) <= max_threads:
            self._threads = int(threads)
        else:
            message = "number of threads must be in range 1-{:d}"
            raise ValueError(message.format(max_threads))
