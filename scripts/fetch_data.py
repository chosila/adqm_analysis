import os
import argparse

from autodqm_ml.data_prep.data_fetcher import DataFetcher
from autodqm_ml.utils import setup_logger

parser = argparse.ArgumentParser()

parser.add_argument(
   "--tag",
   help = "tag to identify output files",
   type = str,
   default = "test"
)
parser.add_argument(
   "--contents",
   help = "path to json file containing subsystems and histograms",
   type = str,
   default = "../metadata/histogram_lists/csc_emtf.json"
)
parser.add_argument(
    "--datasets",
    help = "path to json file containing specified datasets",
    type = str,
    default = "../metadata/dataset_lists/SingleMuon_UL1718.json"
)
parser.add_argument(
   "--debug",
   help = "run logger in DEBUG mode (INFO is default)",
   action = "store_true"
)
parser.add_argument(
    "--short",
    help = "run over only a few files (for debugging purposes)",
    action = "store_true"
)

args = parser.parse_args()

os.system("mkdir -p output/")

logger_mode = "DEBUG" if args.debug else "INFO"
log_file = "output/fetch_data_log_%s.txt" % args.tag
logger = setup_logger(logger_mode, log_file)

fetcher = DataFetcher(
        tag = args.tag,
        contents = args.contents,
        datasets = args.datasets,
        short = args.short
)

fetcher.run()
