import logging
import os


class PipeLogger(object):
    """
    Wrapper for logging.Logger with predefined message format. Logs are by
    default written to stdout (level: debug) but can additionally be copied to
    a log file (level: info). Implements a few extensions to the default
    logging.Logger().

    Parameters:
    -----------
    caller_path : str
        Path to the python script in which the logger initialized.
    data_path : str
        File path of the pipeline data store to construct the path of the log
        file, i.e. /path/to/data yields a log file name /path/to/data.log. If
        no path is provided, no logfile is written.
    append : str
        Whether to append events to an existing log file or overwriting it.
    """

    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S")

    def __init__(self, caller_path, data_path=None, append=True):
        # create logger for script at caller_path
        self.logger = logging.getLogger(
            os.path.basename(caller_path))
        self.logger.setLevel(logging.DEBUG)
        # create terminal handler
        self._term_handler = logging.StreamHandler()
        self._term_handler.setLevel(logging.DEBUG)
        self._term_handler.setFormatter(self._formatter)
        self.logger.addHandler(self._term_handler)
        # add an optional file handler
        if data_path is not None:
            logpath = os.path.splitext(data_path)[0] + ".log"
            if append:
                if not os.path.exists(data_path):
                    message = "data store not found: {:}".format(data_path)
                    self.handleException(OSError(message))
                filemode = "a"
                self.info("appending to logfile: {:}".format(logpath))
            else:
                filemode = "w"
                self.info("creating new logfile: {:}".format(logpath))
            self._file_handler = logging.FileHandler(logpath, filemode)
            self._file_handler.setLevel(logging.INFO)
            self._file_handler.setFormatter(self._formatter)
            self.logger.addHandler(self._file_handler)
        else:
            self._file_handler = None

    def __getattr__(self, attr):
        return getattr(self.logger, attr)
    
    def handleException(self, exception, message=None):
        """
        Take an exception, log a critical event with the exceptions message and
        finaly raise the exception as normal.

        Parameters:
        -----------
        exception : Exception
            Exception to log and raise.
        message : str
            Replace the description message of the exception by this text.
        """
        if message is None:
            # get the error message from the exception
            message = ""
            for arg in exception.args:
                if type(arg) is str:
                    message = arg
                    break
        self.logger.critical(message)
        raise exception

    def setTermLevel(self, levelstr):
        """
        Set the logging level for the terminal (usually stdout), see
        logging.Logger.setLevel().

        Parameters:
        -----------
        levelstr : str
            Filter all events with a lower priority than this one (can be
            either of: error, warning, info, debug).
        """
        if levelstr not in ("error", "warning", "info", "debug"):
            raise ValueError("invalid level: {:}".format(levelstr))
        self._term_handler.setLevel(getattr(logging, levelstr.upper()))

    def setFileLevel(self, levelstr):
        """
        Set the logging level for the log file (usually on disk), see
        logging.Logger.setLevel().

        Parameters:
        -----------
        levelstr : str
            Filter all events with a lower priority than this one (can be
            either of: error, warning, info, debug).
        """
        if levelstr not in ("error", "warning", "info", "debug"):
            raise ValueError("invalid level: {:}".format(levelstr))
        if self._file_handler is not None:
            self._file_handler.setLevel(getattr(logging, levelstr.upper()))
        else:
            self.warn("logging file not set, cannot change logger level")
