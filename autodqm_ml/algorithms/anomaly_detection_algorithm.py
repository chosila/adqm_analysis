import logging

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(__name__)

    def train(self):
        """
        Abstract function to "train" the algorithm.
        
        Only applicable for ML methods. Entails giving a set of histograms
        as training data, as well as any relevant training details for the algorithm.
        """
        raise NotImplementedError()
    
    def evaluate(self, histograms, threshold, metadata = {}): 
        """
        Abstract function to evaluate whether a run is anomalous.

        In the case of ks-test/pull-value test this entails giving a histogram
        along with a reference histogram (or sets of histograms),
        and a threshold for the p-value/chi^2.

        In the case of ML methods (pca, autoencoder, etc), this entails giving
        a histogram (or set of histograms), a trained model config file,
        and a threshold for the output of the ML method.
        :param histograms: list of Histograms to make a decision on
        :type histograms: Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        :param metadata: metadata for this algorithm (e.g. training config dict)
        :type metadata: dict
        """
        raise NotImplementedError()
