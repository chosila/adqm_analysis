import json
import os
import uproot
import numpy
import pandas
import logging

EOS_PATH = "/eos/cms/store/group/comm_dqm/DQMGUI_data/"
HIST_PATH = "DQMData/Run {}/"

class DataFetcher():
    """
    Class to access DQM data on /eos through `xrootd`.
    :param tag: tag to identify this DataFetcher and its output files
    :type tag: str
    :param contents: path to json file specifying the subsytems and histograms to grab 
    :type contents: str
    :param datasets: csv list of primary datasets to grab data for
    :type datasets: str
    :param years: csv list of which years (and eras) to grab data for
    :type years: str
    :param short: flag to just run over a few files (for debugging)
    :type short: bool
    """
    def __init__(self, tag, contents, years, datasets, short = False):
        self.tag = tag

        with open(contents, "r") as f_in:
            self.contents = json.load(f_in)

        self.years = years.split(",")
        self.datasets = datasets.split(",")
        self.short = short

        print(__name__)
        self.logger = logging.getLogger(__name__)


    def run(self):
        """
        Identify all specified DQM files,
        extract specified histograms from these files,
        write data to specified output format,
        and write a summary of the data fetching.
        """
        self.logger.info("[DataFetcher : run] Running DataFetcher to grab the following set of subsystems and histograms")
        for subsystem, info in self.contents.items():
            self.logger.info("\t Subsystem: %s" % subsystem)
            self.logger.info("\t Histograms:")
            for hist in info:
                self.logger.info("\t\t %s" % hist)
        self.logger.info("\t for the following years %s" % str(self.years))
        self.logger.info("\t and for the following primary datasets %s" % str(self.datasets)) 

        self.get_list_of_files()
        self.extract_data()
        self.write_data()
        self.write_summary()


    def get_list_of_files(self):
        """
        Grab list of all DQM files matching specifications.
        """
        self.files = { "all" : [] }
        for pd in self.datasets:
            self.files[pd] = {}
            for year in self.years:
                path = self.construct_eos_path(EOS_PATH, pd, year)
                files = self.get_files(path, self.short)

                self.logger.info("[DataFetcher : get_files] Grabbed %d files under path %s" % (len(files), path))
                self.logger.debug("[DataFetcher : get_files] Full list of files:")
                for file in files:
                    self.logger.debug("\t %s" % file)

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
    def get_files(path, short = False):
        """
        Get all DQM files under a given eos path.
        :param path: path to eos dir
        :type path: str
        :param short: flag to grab only a couple files
        :type short: bool
        """
        files = []
        directories = os.popen("xrdfs root://eoscms.cern.ch/ ls %s" % path).read().split("\n")
        for dir in directories:
            if dir == "":
                continue
            if ".root" in dir: # this is already a root file
                files.append(dir)
            else: # this is a subdir or not a root file
                files += DataFetcher.get_files(dir) # run recursively on subdirs

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
        for pd in self.datasets:
            self.data[pd] = pandas.DataFrame()
            for year in self.years:
                for file in self.files[pd][year]:
                    run_number = DataFetcher.get_run_number(file)
                    self.logger.debug("[DataFetcher : load_data] Loading histograms from file %s, run %d" % (file, run_number))

                    histograms = self.load_data(file, run_number, self.contents) 

                    if histograms is not None:
                        columns = ["run_number", "pd", "year"] + histograms["columns"]
                        column_data = [[run_number, pd, year] + histograms["data"]]
                        df = pandas.DataFrame(column_data, columns = columns)
                        self.data[pd] = self.data[pd].append(df, ignore_index=True)

                    if self.short:
                        if len(self.data[pd]) > 2:
                            continue
        

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
            self.logger.info("[DataFetcher : load_data] ERROR loading file %s" % file)
            return None

        with uproot.open(file) as f:
            if f is None:
                self.logger.info("[DataFetcher : load_data] ERROR loading file %s" % file)
                return None

            for subsystem, histogram_list in contents.items(): 
                for hist in histogram_list:
                    histogram_path = DataFetcher.construct_histogram_path(HIST_PATH, run_number, subsystem, hist) 
                    hist_data["data"].append(f[histogram_path].values())
                    hist_data["columns"].append(subsystem + "/" + hist)

        self.logger.debug("[DataFetcher : load_data] Histogram contents:")
        for hist, data in zip(hist_data["columns"], hist_data["data"]):
            self.logger.debug("\t %s : %s" % (hist, data))

        return hist_data


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
        for pd in self.datasets:
            df = self.data[pd]
            if df is not None:
                df.to_pickle("output/%s_%s.pkl" % (self.tag, pd))


    def write_summary(self):
        """
        Write summary json of configuration.
        """
        return

