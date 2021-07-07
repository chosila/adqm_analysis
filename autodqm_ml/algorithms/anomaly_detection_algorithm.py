import pandas
from sklearn.model_selection import train_test_split

from autodqm_ml import utils
from autodqm_ml.data_formats.histogram import Histogram

import logging
logger = logging.getLogger(__name__)

DEFAULT_COLUMNS = ["run_number"] # columns which should always be read from input df

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name = "default", data_file = None, **kwargs):
        self.name = name

        if data_file is not None:
            self.data_file = utils.expand_path(data_file)
        else:
            self.data_file = None

        self.data_is_loaded = False
        self.data = {} # dictionary to store actual histogram data, of the form { histogram_name : { "X_train" : <data>, "X_test" : <data> }, }
        self.histogram_info = [] # list of Histogram objects (for getting histogram metadata)

        for key, value in kwargs.items():
            setattr(self, key, value)

        

    def load_data(self, file = None, histograms = {}, train_frac = 0.0):
        """

        """
        if self.data_is_loaded:
            return

        file = utils.expand_path(file)
        if file is not None:
            if self.data_file is not None:
                if not (file == self.data_file):
                    logger.warning("[AnomalyDetectionAlgorithm : load_data] Data file was previously set as '%s', but will be changed to '%s'." % (self.data_file, file)) 
                    self.data_file = file
            else:
                self.data_file = file

        if self.data_file is None:
            logger.exception("[AnomalyDetectionAlgorithm : load_data] No data file was provided to load_data and no data file was previously set for this instance, please specify the input data file.")
            raise ValueError()

        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loading training data from file '%s'" % (self.data_file))

        # Load dataframe
        df = pandas.read_pickle(self.data_file)

        # Keep only the necessary columns in dataframe
        self.histograms = list(histograms.keys())
        df = df[DEFAULT_COLUMNS + self.histograms] 

        # Extract actual histogram data
        for histogram, histogram_info in histograms.items():
            # Normalize (if specified in histograms dict)
            if "normalize" in histogram_info.keys():
                if histogram_info["normalize"]:
                    for i in range(len(df)): # TODO: come up with more efficient way than looping through df
                        h = Histogram(name = histogram, data = df[histogram][i])
                        h.normalize()
                        df[histogram][i] = h.data
                        if i == 0: 
                            self.histogram_info.append(h) 
                
            data = list(df[histogram].values)
          
            # Split into training/testing events (only relevant for ML methods)
            self.data[histogram] = {}
            if train_frac > 0:
                X_train, X_test = train_test_split(
                        data,
                        train_size = train_frac,
                        random_state = 0 # fix random state so each histogram gets the same test/train split
                )

                self.data[histogram]["X_train"] = X_train
                self.data[histogram]["X_test"] = X_test

                self.n_train = len(self.data[histogram]["X_train"])
                self.n_test = len(self.data[histogram]["X_test"])

            else:
                self.data[histogram]["X_train"] = None
                self.data[histogram]["X_test"] = data

                self.n_train = 0
                self.n_test = len(self.data[histogram]["X_test"])

        
        for column in DEFAULT_COLUMNS:
            self.data[column] = {}
            data = list(df[column].values)

            if train_frac > 0:
                col_train, col_test = train_test_split(
                        data,
                        train_size = train_frac,
                        random_state = 0 # fix random state so each histogram gets the same test/train split
                )

                self.data[column]["train"] = col_train
                self.data[column]["test"] = col_test

            else:
                self.data[column]["train"] = None
                self.data[column]["test"] = data 


        self.df = df
        logger.debug("[AnomalyDetectionAlgorithm : load_data] Loaded data for %d histograms with %d events in training set and %d events in testing set." % (len(self.histogram_info), self.n_train, self.n_test))

        self.data_is_loaded = True


    def evaluate(self, runs = None, reference = None, histograms = None, threshold = None, metadata = {}):
        """

        """

        if runs is None:
            runs = self.data["run_number"]["test"]

        if histograms is None:
            histograms = self.histograms

        h_ref = {}
        if reference is not None:
            for histogram in self.histograms:
                h_ref[histogram] = Histogram(
                        name = "%s_ref_Run%d" % (histogram, reference),
                        data = self.df[self.df["run_number"] == reference][histogram].iloc[0]
                )


        results = {}
        for run in runs:
            hists = []
            for histogram in self.histograms:
                h = Histogram(
                        name = histogram,
                        data = self.df[self.df["run_number"] == run][histogram].iloc[0],
                        reference = h_ref[histogram] if reference is not None else None
                )
                hists.append(h)
            results[run] = {x : y for x, y in self.evaluate_run(histograms = hists, threshold = threshold, metadata = metadata).items() if x in histograms}

        return results

    def evaluate_run(self, histograms, threshold, metadata):
        """

        """
        raise NotImplementedError()
