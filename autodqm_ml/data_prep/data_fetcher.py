import json
import os
import uproot
import numpy
import pandas

import logging
logger = logging.getLogger(__name__)

EOS_PATH = "/eos/cms/store/group/comm_dqm/DQMGUI_data/"
HIST_PATH = "DQMData/Run {}/"

class DataFetcher():
    """
    Class to access DQM data on /eos through `xrootd`.
    :param tag: tag to identify this DataFetcher and its output files
    :type tag: str
    :param contents: path to json file specifying the subsytems and histograms to grab 
    :type contents: str
    :param datasets: path to json file specifying the years, eras, runs, productions, and primary datasets to grab
    :type datasets: str
    :param short: flag to just run over a few files (for debugging)
    :type short: bool
    """
    def __init__(self, tag, contents, datasets, short = False):
        self.tag = tag

        with open(contents, "r") as f_in:
            self.contents = json.load(f_in)

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

            if "eras" not in info.keys():
                info["eras"] = None

            if "runs" not in info.keys():
                info["runs"] = None

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
                if len(files) > 2:
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
            self.data[pd] = pandas.DataFrame()
            for year in self.datasets.keys():
                for file in self.files[pd][year]:
                    run_number = DataFetcher.get_run_number(file)
                    logger.debug("[DataFetcher : load_data] Loading histograms from file %s, run %d" % (file, run_number))

                    histograms = self.load_data(file, run_number, self.contents) 

                    if histograms is not None:
                        columns = ["run_number", "pd", "year"] + histograms["columns"]
                        column_data = [[run_number, pd, year] + histograms["data"]]
                        df = pandas.DataFrame(column_data, columns = columns)
                        self.data[pd] = self.data[pd].append(df, ignore_index=True)

                    #if self.short:
                    #    if len(self.data[pd]) > 2:
                    #        continue
        

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
        hist_data = { "columns" : [], "data" : [] }

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
                    hist_data["data"].append(f[histogram_path].values())
                    hist_data["columns"].append(subsystem + "/" + hist)

        logger.debug("[DataFetcher : load_data] Histogram contents:")
        for hist, data in zip(hist_data["columns"], hist_data["data"]):
            logger.debug("\t %s : %s" % (hist, data))

        return hist_data


    @staticmethod
    def get_run_prefix(directory):
        """
        For directories on /eos in the form 'R000NNNNxx/', return the run prefix NNNN
        :param directory: name of directory
        :type directory: str
        :return: run prefix
        :rtype: str
        """
        sub_dir = directory.split("/")[-1]
        if not (sub_dir.startswith("R000") or sub_dir.endswith("xx")):
            message = "[DataFetcher : get_run_prefix] Directory '%s' with sub-directory '%s' was not in expected format." % (directory, sub_dir)
            logger.exception(message)
            raise ValueError(message)

        return sub_dir.replace("R000", "").replace("xx", "")

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
        Write dataframe -> pickle file for each primary dataset.
        """
        os.system("mkdir -p output/") 

        for pd in self.pds:
            df = self.data[pd]
            if df is not None:
                df.to_pickle("output/%s_%s.pkl" % (self.tag, pd))


    def write_summary(self):
        """
        Write summary json of configuration.
        """
        return

