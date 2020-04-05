import logging
from os.path import basename


def pipe_logger(caller_path):
    # create logger with 'spam_application'
    logger = logging.getLogger(basename(caller_path))
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    # add the handlers to the pipe_logger
    logger.addHandler(ch)
    return logger
