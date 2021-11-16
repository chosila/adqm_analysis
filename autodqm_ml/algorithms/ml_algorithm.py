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


    def load_model(self, model_file):
        """
        Abstract method to load an ML model from a file.

        :param model_file: file containing weights of a trained model
        :type model_file: str
        """
        raise NotImplementedError()


    def save_model(self, model, model_file):
        """
        Abstract method to load an ML model from a file.

        :param model: ML model
        :type model: varies by ML algorithm
        :param model_file: file containing weights of a trained model
        :type model_file: str
        """
        raise NotImplementedError() 


    def train(self, **kwargs): 
        """
        Abstract method to train an ML model.
        """

        raise NotImplementedError()

