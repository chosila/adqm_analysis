"""
Taken from HiggsDNA project: https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA
Original author: Massimiliano Galli
"""

import logging
from rich.logging import RichHandler
from rich.console import Console

LOGGER_NAME = "autodqm_ml"

def setup_logger(level="INFO", logfile=None):
    """Setup a logger that uses RichHandler to write the same message both in stdout
    and in a log file called logfile. Level of information can be customized and
    dumping a logfile is optional.

    :param level: level of information
    :type level: str, optional
    :param logfile: file where information are stored
    :type logfile: str
    """
    logger = logging.getLogger(LOGGER_NAME)

    # Set up level of information
    possible_levels = ["INFO", "DEBUG"]
    if level not in possible_levels:
        raise ValueError("Passed wrong level for the logger. Allowed levels are: {}".format(
            ', '.join(possible_levels)))
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter("%(message)s")

    # Set up stream handler (for stdout)
    stream_handler = RichHandler(show_time=False, rich_tracebacks=True)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Set up file handler (for logfile)
    if logfile:
        file_handler = RichHandler(
                show_time=False,
                rich_tracebacks=True,
                console=Console(file=open(logfile, "wt"))
                )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
