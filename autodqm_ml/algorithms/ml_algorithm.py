import logging
logger = logging.getLogger(__name__)

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

class MLAlgorithm(AnomalyDetectionAlgorithm):
    """
    Abstract base class for any ML-based anomaly detection algorithm,
    including PCA and Autoencoder.
    """
    def __init__(self, **kwargs):
        super(MLAlgorithm, self).__init__(**kwargs)
        self.model = None


    
    def predict(self, **kwargs):
        """

        """
        raise NotImplementedError()


    def plot_original_vs_reconstructed(self, histograms, N = None):
        """
        Make plots of original and reconstructed histograms for ML algorithm.
        Case 1:
            - user supplies a list of histogram names and a number N of runs to make plots for
        Case 2:
            - user supplies a list of Histogram objects to make plots for
        """
        # TODO

        #if N is not None: # we will randomly select N runs from test set to plot original vs. reconstructed
        #    original_hists = self.make_inputs(split = "test", N = N)
        #    reconstructed_hists = self.predict(original_hists)
        #
        #else: # user has supplied specific histograms to make plots for
        #    original_hists = self.make_inputs(histograms)
        #    reconstructed_hists = self.predict(original_hists)


    def make_plots(self, N = 1):
        """

        """
        raise NotImplementedError()


    def evaluate_with_model(self, histograms, threshold, metadata):
        """
        Abstract method to evaluate whether a run is anomalous for a specific ML algorithm.
        Should be implemented for any derived class.
        """
        raise NotImplementedError()


    def set_model(self, model_file, config = {}):
        """
        Load an autoencoder model from a file and set this as the algorithm to use during evaluate.

        :param model_file: file containing weights of a trained model
        :type model_file: str
        :param config: additional config options for this model
        :type config: dict
        """

        if self.model is not None:
            logger.warning("[ml_algorithm.py : load_model] A model has already been set for this ML Algorithm object. We will overwrite with the newly specified model, but please make sure this is intended.")

        self.model = self.load_model(model_file, **config)

        if self.model is None or not self.model:
            message = "[ml_algorithm.py : load_model] There was some problem loading model from file %s with options %s" % (model_file, str(config))
            logger.exception(message)
            raise ValueError(message)

        return True


    def load_model(self, model_file, **kwargs):
        """
        Abstract method to load an ML model from a file.

        :param model_file: file containing weights of a trained model
        :type model_file: str
        """
        raise NotImplementedError()


    def save_model(self, model_file, **kwargs):
        """
        Abstract method to load an ML model from a file.

        :param model_file: file containing weights of a trained model
        :type model_file: str
        """
        raise NotImplementedError() 



    def train(self, histograms, file, config):
        """
        Abstract method to train an ML model.

        """

        raise NotImplementedError()

