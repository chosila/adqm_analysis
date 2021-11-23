import os
import json
import argparse
import awkward
import numpy

from autodqm_ml.utils import setup_logger
from autodqm_ml.utils import expand_path
from autodqm_ml.plotting.plot_tools import make_original_vs_reconstructed_plot, make_sse_plot
from autodqm_ml.constants import kANOMALOUS, kGOOD

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help = "output directory to place files in",
        type = str,
        required = False,
        default = "output"
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
        help = "csv list of histograms to assess", 
        type = str,
        required = True,
        default = None
    )
    parser.add_argument(
        "--algorithms",
        help = "csv list of algorithmn names to assess",
        type = str,
        required = False,
        default = None
    )
    parser.add_argument(
        "--n_runs",
        help = "number of runs to make original/reconstructed plots for",
        type = int,
        required = False,
        default = 3
    )
    parser.add_argument(
        "--runs",
        help = "csv list of runs to make original/reconstructed plots for",
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

    return parser.parse_args()


def infer_algorithms(runs, histograms, algorithms):
    for histogram, info in histograms.items():
        for field in runs.fields:
            if field == histogram:
                histograms[histogram]["original"] = field
            elif histogram in field and "_score_" in field:
                algorithm = field.replace(histogram, "").replace("_score_", "")
                if algorithms is not None:
                    if algorithm not in algorithms:
                        continue

                if not algorithm in info["algorithms"].keys():
                    histograms[histogram]["algorithms"][algorithm] = { "score" : field }

                # Check if a reconstructed histogram also exists for algorithm
                reco = field.replace("score", "reco")
                if reco in runs.fields:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = reco
                else:
                    histograms[histogram]["algorithms"][algorithm]["reco"] = None

    return histograms


def main(args):
    os.system("mkdir -p %s/" % args.output_dir)

    logger_mode = "DEBUG" if args.debug else "INFO"
    log_file = "%s/fetch_data_log_%s.txt" % (args.output_dir, "assess")
    logger = setup_logger(logger_mode, log_file)


    histograms = { x : {"algorithms" : {}} for x in args.histograms.split(",") }

    runs = awkward.from_parquet(args.input_file)

    if args.algorithms is not None:
        algorithms = args.algorithms.split(",")
    else:
        algorithms = None

    histograms = infer_algorithms(runs, histograms, algorithms)
    for h, info in histograms.items():
        logger.debug("[assess.py] For histogram '%s', found the following anomaly detection algorithms:" % (h))
        for a, a_info in info["algorithms"].items():
            logger.debug("\t Algorithm '%s' with score in field '%s' and reconstructed histogram in field '%s'" % (a, a_info["score"], str(a_info["reco"])))
    

    # Print out runs with N highest sse scores for each histogram
    N = 5
    for h, info in histograms.items():
        for algorithm, algorithm_info in info["algorithms"].items():
            runs_sorted = runs[awkward.argsort(runs[algorithm_info["score"]], ascending=False)]
            logger.info("[assess.py] For histogram '%s', algorithm '%s', the mean +/- std anomaly score is: %.2e +/- %.2e." % (h, algorithm, awkward.mean(runs[algorithm_info["score"]]), awkward.std(runs[algorithm_info["score"]])))
            #logger.info("[assess.py] For histogram '%s', algorithm '%s', the runs with the highest anomaly scores are: " % (h, algorithm)) 
            logger.info("\t The runs with the highest anomaly scores are:")
            for i in range(N):
                logger.info("\t Run number : %d, Anomaly Score : %.2e" % (runs_sorted.run_number[i], runs_sorted[algorithm_info["score"]][i]))

    # Histogram of sse for algorithms
    for h, info in histograms.items():
        splits = {
                "train_label" : [("train", 0), ("test", 1)],
                "label" : [("anomalous", kANOMALOUS), ("good", kGOOD)]
        }
        for split, split_info in splits.items():
            for name, id in split_info:
                runs_set = runs[runs[split] == id]
                if len(runs_set) == 0:
                    logger.warning("[assess.py] For histogram '%s', no runs belong to the set '%s', skipping making a histogram of SSE for this." % (h, name))
                    continue
                recos = {}
                for algorithm, algorithm_info in info["algorithms"].items():
                    recos[algorithm] = { "score" : runs_set[algorithm_info["score"]] }
                h_name = h.replace("/", "").replace(" ", "")
                save_name = args.output_dir + "/" + h_name + "_sse_%s_%s.pdf" % (split, name)
                make_sse_plot(h_name, recos, save_name)


        #for set, id in zip(["train", "test"], [0, 1]):
        #    runs_set = runs[runs.train_label == id]
        #    recos = {}
        #    for algorithm, algorithm_info in info["algorithms"].items():
        #        recos[algorithm] = { "score" : runs_set[algorithm_info["score"]] }
        #    h_name = h.replace("/", "").replace(" ", "")
        #    save_name = args.output_dir + "/" + h_name + "_sse_%s.pdf" % set
        #    make_sse_plot(h_name, recos, save_name)

    # Plots of original/reconstructed histograms
    if args.runs is None:
        random_runs = True
        selected_runs_idx = numpy.random.choice(len(runs), size=args.n_runs, replace=False)
        selected_runs = runs.run_number[selected_runs_idx]
        logger.debug("[assess.py] An explicit list of runs was not given, so we will make plots for %d randomly chosen runs: %s" % (args.n_runs, str(selected_runs)))
    else:
        random_runs = False
        selected_runs = [int(x) for x in args.runs.split(",")]
        selected_runs_idx = runs.run_number < 0 # dummy all False
        for run in selected_runs:
            selected_runs_idx = selected_runs_idx | (runs.run_number == run)
        logger.debug("[assess.py] Will make plots for the %d specified runs: %s" % (len(selected_runs), str(selected_runs)))

    runs_trim = runs[selected_runs_idx]
    for h, info in histograms.items():
        for i in range(len(runs_trim)):
            run = runs_trim[i]
            run_number = run.run_number
            original = run[info["original"]]
            recos = {}
            for algorithm, algorithm_info in info["algorithms"].items():
                if algorithm_info["reco"] is None:
                    continue
                recos[algorithm] = { "reco" : run[algorithm_info["reco"]], "score" : run[algorithm_info["score"]]}
            h_name = h.replace("/", "").replace(" ", "")
            save_name = args.output_dir + "/" + h_name + "_Run%d.pdf" % run_number
            make_original_vs_reconstructed_plot(h_name, original, recos, run_number, save_name) 


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
