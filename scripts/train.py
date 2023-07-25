import os
import json
import argparse

from autodqm_ml.utils import setup_logger
from autodqm_ml.algorithms.statistical_tester import StatisticalTester
from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.pca import PCA
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.utils import expand_path

parser = argparse.ArgumentParser()

# Required arguments
parser.add_argument(
    "--algorithm",
    help = "name of algorithm ('PCA' or 'Autoencoder' or 'StatisticalTester') to train with default options OR path to json filed specifying particular options for training a given algorithm.",
    type = str,
    required = True
)

# Optional arguments
parser.add_argument(
    "--output_dir",
    help = "output directory to place files in",
    type = str,
    required = False,
    default = None
)
parser.add_argument(
    "--tag",
    help = "tag to identify output files",
    type = str,
    required = False,
    default = None
)
parser.add_argument(
    "--input_file",
    help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
    type = str,
    required = False,
    default = None
)

parser.add_argument(
    "--low_stat_threshold",
    help = "Minimum number of entries required per histogram for training. If a histogram has less than the set minimum, the histogram will not be included in training.",
    type = int,
    required = False,
    default = 10000
)
parser.add_argument(
    "--train_highest_only",
    help = "If True, only trains on the runs with the highest stats, or the highest number of entries. The test set becomes the remaining runs.",
    type = bool,
    required = False,
    default = False
)


parser.add_argument(
    "--histograms",
    help = "csv list of histograms on which to train the ML algorithm. If multiple are supplied, for PCAs one PCA will be trained for each histogram, while for autoencoders, a single AutoEncoder taking each of the histograms as inputs will be trained.",
    type = str,
    required = False,
    default = None
)
# To be added when I figure out how to add both safely.
#parser.add_argument(
#    "--train_size",
#    help = "proportion of data to be used in model training (as opposed to model testing). Entering a number less than 0 does something weird, but I forgot what that is."
#    required = False,
#    default = 0.5,
#)

parser.add_argument(
    "--reference",
    help = "reference run number to use for comparisons with StatisticalTester",
    type = int,
    required = False,
    default = None
)
parser.add_argument(
    "--n_components",
    help = "dimension of latent space (number of principle components for PCA)",
    type = int,
    required = False,
    default = None
)
parser.add_argument(
    "--autoencoder_mode",
    help = "specify whether you want to train an autoencoder for each histogram ('individual') or a single autoencoder on all histograms ('simultaneous')",
    type = str,
    required = False,
    default = None
)
parser.add_argument(
    "--debug",
    help = "run logger in DEBUG mode (INFO is default)",
    required = False,
    action = "store_true"
)

args = parser.parse_args()

os.system("mkdir -p %s/" % args.output_dir)

logger_mode = "DEBUG" if args.debug else "INFO"
log_file = "%s/fetch_data_log_%s.txt" % (args.output_dir, args.tag)
logger = setup_logger(logger_mode, log_file)

if "json" in args.algorithm:
    if not os.path.exists(args.algorithm):
        algorithm_config_file = expand_path(args.algorithm)
    else:
        algorithm_config_file = args.algorithm

    with open(algorithm_config_file, "r") as f_in:
        config = json.load(f_in)

    # Add command line arguments to config
    for k,v in vars(args).items():
        if v is not None:
            config[k] = v # note: if you specify an argument both through command line argument and json, we give precedence to the version from command line arguments

else:
    config = vars(args)
    config["name"] = args.algorithm.lower()

if not config["name"] in ["autoencoder", "pca", "statistical_tester"]:
    message = "[train.py] Requested algorithm '%s' is not in the supported list of algorithms ['autoencoder', 'pca']." % (config["name"])
    logger.exception(message)
    raise RuntimeError()

if config["name"] == "pca":
    algorithm = PCA(**config)
elif config["name"] == "autoencoder":
    algorithm = AutoEncoder(**config)
elif config["name"] == "statistical_tester":
    algorithm = StatisticalTester(**config)

if args.input_file is None and "input_file" not in config.keys():
    message = "[train.py] An input file for training the ML algorithm was not supplied through CLI nor found in the json config file for the algorithm."
    logger.exception(message)
    raise RuntimeError()

if args.histograms is None and "histograms" not in config.keys():
    message = "[train.py] A list of histograms to train on was not supplied through CLI nor found in the json config file for the algorithm."
    logger.exception(message)
    raise RuntimeError()

if args.input_file is not None: #
    training_file = args.input_file
else:
    training_file = config["input_file"]

if (args.histograms is not None) and ('.json' not in args.histograms):
    histograms = {x : { "normalize" : True} for x in args.histograms.split(",")}
elif '.json' in args.histograms:
    histograms = json.load(open(args.histograms, 'r'))
    histograms = { i + '/' + j : {"normalize" : False} for i in histograms for j in histograms[i]}
elif isinstance(config["histograms"], str):
    histograms = {x : { "normalize" : True} for x in config["histograms"].split(",")}
elif isinstance(config["histograms"], dict):
    histograms = config["histograms"]
# take the metadata/histogram_list json as histogram list
else:
    logger.exception("[train.py] The `histograms` argument should either be a csv list of histogram names (str) or a dictionary (if provided through a json config).")
    raise RuntimeError()

# Load data
algorithm.load_data(
    file= training_file,
    histograms = histograms
)

# Train
if isinstance(algorithm, MLAlgorithm):
    algorithm.train()

# Predict
algorithm.predict()

# Save model and new df with score zipped in
algorithm.save()
