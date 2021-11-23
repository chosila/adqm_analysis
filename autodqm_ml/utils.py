"""
Taken from HiggsDNA project: https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA
Original author: Massimiliano Galli
"""

import os
import copy
import subprocess

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


def expand_path(relative_path):
    """
    Convert a relative path (assumed to be the path under AutoDQM_ML/) into an absolute path.

    :param relative_path: path under AutoDQM_ML/
    :type relative_path: str
    :return: absolute path
    :rtype: str
    """

    dir = os.path.dirname(__file__)
    subdirs = dir.split("/")

    base_path = ""
    for subdir in subdirs:
        base_path += subdir + "/"
        if subdir == "AutoDQM_ML":
            break

    return base_path + relative_path


def update_dict(original, new):
    """
    Update nested dictionary (dictionary possibly containing dictionaries)
    If a field is present in new and original, take the value from new.
    If a field is present in new but not original, insert this field 


    :param original: source dictionary
    :type original: dict
    :param new: dictionary to take new values from
    :type new: dict
    :return: updated dictionary
    :rtype: dict
    """

    updated = copy.deepcopy(original)

    for key, value in original.items():
        if key in new.keys():
            if isinstance(value, dict):
                updated[key] = update_dict(value, new[key])
            else:
                updated[key] = new[key]

    return updated


def do_cmd(cmd, returnStatus=False, dryRun=False):
    """

    """
    if dryRun:
        print("dry run: {}".format(cmd))
        status, out = 1, ""
    else:
        status, out = subprocess.getstatusoutput(cmd)
    if returnStatus:
        return status, out
    else:
        return out


def check_proxy():
    """
    Check if a valid grid proxy exists.

    :return: path to proxy if it exists, otherwise None
    :rtype: str
    """

    proxy = None
    bad_proxy = False
    proxy_info = do_cmd("voms-proxy-info").split("\n")
    for line in proxy_info:
        if "path" in line:
            proxy = line.split(":")[-1].strip()

        if "timeleft" in line:
            time_left = int(line.replace("timeleft", "").replace(":", "").strip())
            if not time_left > 0:
                bad_proxy = True

        if "Couldn't find a valid proxy." in line:
            bad_proxy = True

    if proxy is None or bad_proxy:
        return None

    else:
        return proxy


