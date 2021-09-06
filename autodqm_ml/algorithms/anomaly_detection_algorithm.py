import pandas
import numpy
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

        

    def load_data(self, file = None, histograms = {}, train_frac = 0.0, remove_identical_bins = False, remove_low_stat = False):
        """
        Loads data from pickle file into ML class. 

        :param file: file containing data to be extracted. File output of fetch_data.py
        :type file: str
        :param histograms: names of histograms to be loaded. Must match histogram names used in fetch_data.py. Dictionary in the form {<histogram name> : {"normalize" : <bool>}}.
        :type histograms: dict. Default histograms = {}
        :param train_frac: fraction of dataset to be kept as training data. Must be between 0 and 1. 
        :type train_frac: float. Default train_frac = 0.0
        :param remove_identical_bins: removes bins that are identical throughout all runs. 
        :type remove_identical_bins: bool. Default remove_identical_bins = False.
        :param remove_low_stat: removes runs containing histograms with low stats. Low stat threshold is 10000 events.
        :type remove_low_stat: bool. remove_low_stat = False
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


        
        for histogram, histogram_info in histograms.items():
            # remove low stat hists if required
            # needs own loop in case the later hists also cuts df
            if remove_low_stat:
                mask = df[histogram].apply(numpy.sum) > 10000
                df = df[mask]
                df.reset_index(drop=True, inplace=True)


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
            
            # Remove identical bins if required
            if remove_identical_bins:
                # identify bad bins
                hd = numpy.stack(df[histogram].values)
                nbins = hd.shape[1]
                bad_bins = numpy.all(hd==numpy.tile(hd[0,:],hd.shape[0]).reshape(hd.shape), axis=0)
                good_bins = numpy.logical_not(bad_bins)
                bad_bins = numpy.arange(nbins)[bad_bins]
                good_bins = numpy.arange(nbins)[good_bins]
                # remove bad bins
                cleaned = hd[:, good_bins]
                data = numpy.split(cleaned, cleaned.shape[0])
                data = [x.flatten() for x in data]
                # modify df, so df used for eval matches in dimension
                df[histogram] = data 
            else:
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
