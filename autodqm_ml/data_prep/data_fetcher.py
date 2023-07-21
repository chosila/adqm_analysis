import json
import os
import uproot
import numpy
import pandas
import awkward
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

from autodqm_ml.utils import check_proxy, expand_path

EOS_PATH = "/eos/cms/store/group/comm_dqm/DQMGUI_data/"
HIST_PATH = "DQMData/Run {}/"

class DataFetcher():
    """
    Class to access DQM data on /eos through `xrootd`.
    :param contents: path to json file specifying the subsytems and histograms to grab 
    :type contents: str
    :param datasets: path to json file specifying the years, eras, runs, productions, and primary datasets to grab
    :type datasets: str
    :param short: flag to just run over a few files (for debugging)
    :type short: bool
    """
    def __init__(self, output_dir, contents, datasets, short = False):
        proxy = check_proxy()
        if proxy is None:
            message = "[DataFetcher : __init__] Unable to find a valid grid proxy, which is necessary to access DQM data on /eos with `xrootd`. Please create a valid grid proxy and rerun."
            logger.exception(message)
            raise RuntimeError()
        
        self.output_dir = output_dir

        if not os.path.exists(contents):
            contents = expand_path(contents)
        with open(contents, "r") as f_in:
            self.contents = json.load(f_in)

        if not os.path.exists(datasets):
            datasets = expand_path(datasets)
        with open(datasets, "r") as f_in:
            pds_and_datasets = json.load(f_in)

        if "primary_datasets" not in pds_and_datasets.keys():
            message = "[DataFetcher : __init__] The 'primary_datasets' field was not specified in input json '%s'! Please specify." % primary_datasets
            logger.exception(message)
            raise ValueError(message)
 
        if "years" not in pds_and_datasets.keys():
            message = "[DataFetcher : __init__] The 'years' field was not specified in input json '%s'! Please specify." % datasets
            logger.exception(message)
            raise ValueError(message) 
           
        self.pds = pds_and_datasets["primary_datasets"]
        self.datasets = pds_and_datasets["years"]

        for year, info in self.datasets.items():
            if "productions" not in info.keys():
                message = "[DataFetcher : __init__] For year '%s', the 'productions' field was not specified! Please specify." % (year)
                logger.exception(message)
                raise ValueError(message)

            for field in ["eras", "runs", "bad_runs", "good_runs"]:
                if field not in info.keys():
                    info[field] = None

        self.short = short


    def run(self):
        """
        Identify all specified DQM files,
        extract specified histograms from these files,
        write data to specified output format,
        and write a summary of the data fetching.
        """
        logger.info("[DataFetcher : run] Running DataFetcher to grab the following set of subsystems and histograms")
        for subsystem, info in self.contents.items():
            logger.info("\t Subsystem: %s" % subsystem)
            logger.info("\t Histograms:")
            for hist in info:
                logger.info("\t\t %s" % hist)
        logger.info("\t for the following years %s" % str(self.datasets.keys()))
        logger.info("\t and for the following primary datasets %s" % str(self.pds)) 

        logger.info("[DataFetcher : run] Grabbing histograms for the following years: %s" % str(self.datasets.keys()))
        for year, info in self.datasets.items():
            logger.info("Year: %s" % year)
            logger.info("\t productions: %s" % (str(info["productions"])))
            logger.info("\t specified eras: %s" % (str(info["eras"])))
            logger.info("\t specified runs: %s" % (str(info["runs"])))

            if info["bad_runs"] is not None:
                logger.info("\t The following runs will be labeled as 'bad' (with 'label' = 1): %s" % (str(info["bad_runs"]))) 
            if info["good_runs"] is not None:
                logger.info("\t The following runs will be labeled as 'good' (with 'label' = 0): %s" % (str(info["good_runs"])))

            if info["bad_runs"] is not None and info["good_runs"] is not None:
                logger.info("\t and all other runs will be labeled with 'label' = -1.")
            elif info["bad_runs"] is not None:
                logger.info("\t and all other runs will be assumed to be 'good' (with 'label' = 0).")
            elif info["good_runs"] is not None:
                logger.info("\t and all other runs will be assumed to be 'bad' (with 'label' = 1).")




        self.get_list_of_files()
        self.extract_data()
        self.write_data()
        self.write_summary()


    def get_list_of_files(self):
        """
        Grab list of all DQM files matching specifications.
        """
        self.files = { "all" : [] }

        for pd in self.pds:
            self.files[pd] = {}
            for year, info in self.datasets.items():
                path = self.construct_eos_path(EOS_PATH, pd, year)

                logger.info("[DataFetcher : get_files] Searching for directories and files matching the primary dataset '%s', the specified datasets %s under directory '%s'" % (pd, str(info), path))
                    
                files = self.get_files(path, year, info, self.short)

                logger.info("[DataFetcher : get_files] Grabbed %d files under path %s" % (len(files), path))
                logger.debug("[DataFetcher : get_files] Full list of files:")
                for file in files:
                    logger.debug("\t %s" % file)

                self.files[pd][year] = files
                self.files["all"] += files


    @staticmethod
    def construct_eos_path(base_path, pd, year):
        """
        Construct path to eos dqm files.
        :param base_path: base path to eos dqm files
        :type base_path: str
        :param pd: primary dataset
        :type pd: str
        :param year: year to get dqm files for
        :type year: str
        """
        # this is trivial now, but may get more complicated in Run 3, adding other subsystems, etc
        return base_path + ("Run%s/" % year) + pd + "/"


    @staticmethod
    def get_files(path, year, datasets, short = False):
        """
        Get all DQM files under a given eos path.
        :param path: path to eos dir
        :type path: str
        :param year: specified year to grab files for
        :type year: str
        :param datasets: dictionary of productions, eras, and runs to grab files for
        :type datasets: dict
        :param short: flag to grab only a couple files
        :type short: bool
        """
        files = []

        directories = os.popen("xrdfs root://eoscms.cern.ch/ ls %s" % path).read().split("\n")
        for dir in directories:
            if dir == "":
                continue
            if ".root" in dir: # this is already a root file
                file = dir
                if not any(prod in file for prod in datasets["productions"]): # check if file matches any of the specified productions
                    continue
                if datasets["eras"] is not None:
                    if not any(("Run" + year + era) in file for era in datasets["eras"]): # check if file matches any of the specified eras
                        continue
                if datasets["runs"] is not None:
                    if not any(run in file for run in datasets["runs"]): # check if file matches any of the specified runs
                        continue
                files.append(file)
            else: # this is a subdir or not a root file
                if datasets["runs"] is not None:
                    run_prefix = DataFetcher.get_run_prefix(dir)
                    if not any(run_prefix in run for run in datasets["runs"]): # check if any specified runs fall in the run range for this directory
                        continue
                files += DataFetcher.get_files(dir, year, datasets, short) # run recursively on subdirs

            if short:
                if len(files) > 5:
                    break 

        for idx, file in enumerate(files):
            if not file.startswith("root://eoscms.cern.ch/"):
                files[idx] = "root://eoscms.cern.ch/" + file # prepend xrootd accessor

        return files


    def extract_data(self):
        """
        Extract all requested histograms from list of files.
        """
        self.data = {}
        for pd in self.pds:
            self.data[pd] = {} 
            for year in self.datasets.keys():
                logger.info("[DataFetcher : extract_data] Loading histograms for pd '%s' and year '%s' from %d total files." % (pd, year, len(self.files[pd][year])))
                for file in tqdm(self.files[pd][year]):
                    run_number = DataFetcher.get_run_number(file)

                    label = -1 # unknown good/bad
                    if self.datasets[year]["bad_runs"] is not None:
                        if str(run_number) in self.datasets[year]["bad_runs"]:
                            label = 1 # bad/anomalous
                        elif self.datasets[year]["good_runs"] is None:
                            label = 0 # if only bad_runs was specified, mark everything not in bad_runs as good

                    if self.datasets[year]["good_runs"] is not None:
                        if str(run_number) in self.datasets[year]["good_runs"]:
                            label = 0 # good/not anomalous
                        elif self.datasets[year]["bad_runs"] is None:
                            label = 1 # if only good_runs was specified, mark everything not in good_runs as bad

                    logger.debug("[DataFetcher : load_data] Loading histograms from file %s, run %d" % (file, run_number))

                    histograms = self.load_data(file, run_number, self.contents) 
                    if not self.data[pd]:
                        self.data[pd] = histograms

                    if histograms is not None:
                        histograms["run_number"] = [run_number]
                        histograms["year"] = [year]
                        histograms["label"] = [label]
                        for k, v in histograms.items():
                            self.data[pd][k] += v
                        

    def load_data(self, file, run_number, contents): 
        """
        Load specified histograms from a given file.
        :param file: dqm file
        :type file: str
        :param run_number: run number for this file
        :type run_number: int
        :param contents: dictionary of subsystems : list of histograms to load data for
        :type contents: dict
        :param subsystem: name of subsystem
        :type subsystem: str
        :param histograms: list of histograms to load data for
        :type histograms: list of str
        :return: histogram names and contents
        :rtype: dict 
        """
        #hist_data = { "columns" : [], "data" : [] }
        hist_data = {}

        # Check if file is corrupt
        try:
            uproot.open(file)
        except:
            logger.warning("[DataFetcher : load_data] Problem loading file '%s', it might be corrupted. We will just skip this file." % file)
            return None

        with uproot.open(file) as f:
            if f is None:
                logger.warning("[DataFetcher : load_data] Problem loading file '%s', it might be corrupted. We will just skip this file." % file)
                return None

            for subsystem, histogram_list in contents.items(): 
                for hist in histogram_list:
                    histogram_path = DataFetcher.construct_histogram_path(HIST_PATH, run_number, subsystem, hist) 
                    hist_data[subsystem + "/" + hist] = [f[histogram_path].values()]

        logger.debug("[DataFetcher : load_data] Histogram contents:")
        for hist, data in hist_data.items():
            logger.debug("\t %s : %s" % (hist, data))

        return hist_data


    @staticmethod
    def get_run_prefix(directory):
        """
        For directories on /eos in the form 'R000NNNNxx/ or 000NNNNxx', return the run prefix NNNN
        :param directory: name of directory
        :type directory: str
        :return: run prefix
        :rtype: str
        """
        sub_dir = directory.split("/")[-1]
        if not (sub_dir.startswith("R000") or sub_dir.startswith("000") or sub_dir.endswith("xx")):
            message = "[DataFetcher : get_run_prefix] Directory '%s' with sub-directory '%s' was not in expected format." % (directory, sub_dir)
            logger.exception(message)
            raise ValueError(message)

        if sub_dir.startswith("R000"):
            return sub_dir.replace("R000", "").replace("xx", "")
        elif sub_dir.startswith("000"):
            return sub_dir[3:].replace("xx", "")
        else:
            return sub_dir.replace("xx", "")

    @staticmethod
    def get_run_number(file):
        """
        Return run number from a given file name.
        :param file: name of file
        :type file: str
        :return: run number
        :rtype: int
        """
        return int(file.split("/")[-1].split("__")[0][-6:])


    @staticmethod
    def construct_histogram_path(base_path, run_number, subsystem, histogram):
        """
        Construct path to histogram inside dqm file.
        :param base_path: base path inside dqm file
        :type base_path: str
        :param run_number: run number for this file
        :type run_number: int
        :param subsystem: name of subsystem
        :type subsystem: str
        :param histogram: name of histogram
        :type histogram: str
        """
        # this is trivial now, but may get more complicated
        return base_path.format(run_number) + subsystem + "/" + histogram


    def write_data(self):
        """
        Write dataframe -> parquet file for each primary dataset.
        The conversion from pandas -> awkward and then saving as parquet makes output files about an order of magnitude smaller than simply doing df.to_pickle.
        The resulting .parquet file can be read either as an awkward Array or a pandas dataframe with:
            array = awkward.from_parquet("file.parquet")
            df = pandas.read_parquet("file.parquet")
        """
        os.system("mkdir -p %s/" % self.output_dir) 

        for pd in self.pds:
            array = awkward.Array(self.data[pd])
            if array is not None:
                output_file = "%s/%s.parquet" % (self.output_dir, pd)
                logger.info("[DataFetcher : write_data] Writing histograms to output file '%s'" % (output_file))
                awkward.to_parquet(array, output_file) 


    def write_summary(self):
        """
        Write summary json of configuration.
        """
        return

