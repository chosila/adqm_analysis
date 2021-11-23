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
parser.add_argument(
    "--output_dir",
    help = "output directory to place files in",
    type = str,
    required = False,
    default = "output"
)
parser.add_argument(
    "--tag",
    help = "tag to identify output files",
    type = str,
    required = False,
    default = "test"
)
parser.add_argument(
    "--algorithm",
    help = "name of algorithm ('PCA' or 'Autoencoder' or 'StatisticalTester') to train with default options OR path to json filed specifying particular options for training a given algorithm.",
    type = str,
    required = True
)
parser.add_argument(
    "--input_file",
    help = "input file (i.e. output from fetch_data.py) to use for training the ML algorithm",
    type = str,
    required = False,
    default = None
)
parser.add_argument(
    "--histograms",
    help = "csv list of histograms on which to train the ML algorithm. If multiple are supplied, for PCAs one PCA will be trained for each histogram, while for autoencoders, a single AutoEncoder taking each of the histograms as inputs will be trained.",
    type = str,
    required = False,
    default = None
)
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
        algorithm_config_file = algo
    with open(algorithm_config_file, "r") as f_in:
        config = json.load(f_in)

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

if args.histograms is not None:
    histograms = {x : { "normalize" : True} for x in args.histograms.split(",")}
else:
    histograms = config["histograms"]

# Load data
algorithm.load_data(
    file = training_file,
    histograms = histograms
)

# Train
if isinstance(algorithm, MLAlgorithm):
    algorithm.train()

# Predict
algorithm.predict()

# Save model and new df with score zipped in
algorithm.save() 
