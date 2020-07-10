import logging
from os.path import basename, splitext


class PipeLogger(object):

    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S")

    def __init__(self, caller_path, data_path=None, append=True):
        # create logger for script at caller_path
        self.logger = logging.getLogger(basename(caller_path))
        self.logger.setLevel(logging.DEBUG)
        # create terminal handler
        self._term_handler = logging.StreamHandler()
        self._term_handler.setLevel(logging.DEBUG)
        self._term_handler.setFormatter(self._formatter)
        self.logger.addHandler(self._term_handler)
        # add an optional file handler
        if data_path is not None:
            logpath = splitext(data_path)[0] + ".log"
            filemode = "a" if append else "w"
            if filemode == "w":
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
        if levelstr not in ("error", "warning", "info", "debug"):
            raise ValueError("invalid level: {:}".format(levelstr))
        self._term_handler.setLevel(getattr(logging, levelstr.upper()))

    def setFileLevel(self, levelstr):
        if levelstr not in ("error", "warning", "info", "debug"):
            raise ValueError("invalid level: {:}".format(levelstr))
        if self._file_handler is not None:
            self._file_handler.setLevel(getattr(logging, levelstr.upper()))
        else:
            self.warn("logging file not set, cannot change logger level")
