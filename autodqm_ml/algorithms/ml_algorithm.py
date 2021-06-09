import logging
logger = logging.getLogger(__name__)

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

class MLAlgorithm(AnomalyDetectionAlgorithm):
    """
    Abstract base class for any ML-based anomaly detection algorithm,
    including PCA and Autoencoder.
    """
    def __init__(self, name):
        super(MLAlgorithm, self).__init__(name)
        self.model = None


    def evaluate(self, histograms, threshold, metadata):
        if self.model is None:
            message = "[ml_algorithm.py : evaluate] No model has been set for this ML algorithm, cannot evaluate histograms!"
            logger.exception(message)
            raise ValueError(message)
        
        results = self.evaluate_with_model(histograms, threshold, metadata)
        return results


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


    def train(self, **kwargs):
        """
        Abstract method to train an ML model.

        """

        raise NotImplementedError()