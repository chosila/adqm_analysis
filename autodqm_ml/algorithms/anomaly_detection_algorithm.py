import logging
logger = logging.getLogger(__name__)

class AnomalyDetectionAlgorithm():
    """
    Abstract base class for any anomaly detection algorithm,
    including ks-test, pull-value test, pca, autoencoder, etc.
    :param name: name to identify this anomaly detection algorithm
    :type name: str
    """

    def __init__(self, name):
        self.name = name

    def run(self, histograms, threshold, metadata = {}): 
        """
        :param histograms: list of Histograms to make a decision on
        :type histograms: list of autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        :param metadata: metadata for this algorithm (e.g. training config dict)
        :type metadata: dict
        """

        results = self.evaluate(histograms, threshold, metadata)

        # Check that results are in the proper format
        if not isinstance(results, dict):
            message = "The output of any AnomalyDetectionAlgorithm.evaluate method should be a dictionary, not %s as you have passed." % (str(type(results)))
            logger.exception(message)
            raise TypeError(message)

        # Check that there is an entry for each histogram
        names_in = set([histogram.name for histogram in histograms])
        names_out = set([str(name) for name in results.keys()])

        if not names_in == names_out:
            message = "The list of input histograms does not match the list of output histograms! Input: %s, Output: %s" % (names_in, names_out)
            logger.exception(message)
            raise AssertionError(message)

        # Check that each histogram has the proper results
        for name, result in results.items():
            if "decision" not in result or "score" not in result.keys():
                message = "Each histogram name should correspond to a dictionary with two entries, 'decision' and 'name', not %s as is the case for histogram %s" % (str(result.keys()), name)
                logger.exception(message)
                raise AssertionError(message)

        return results

    def evaluate(self, histograms, threshold, metadata):
        """
        Abstract function to evaluate whether a run is anomalous.

        In the case of ks-test/pull-value test this entails giving a list of histograms
        where each histogram in the list should have a reference histogram,
        and a threshold for the p-value/chi^2.

        In the case of ML methods (pca, autoencoder, etc), this entails giving
        a list of histograms, a trained model config file,
        and a threshold for the output of the ML method.

        Some algorithms perform per-histogram decisions and may have a different decision
        for each histogram in the list.
        Some algorithms may make global (multi-histogram) decisions and have the same decision
        for subsets of histograms which are considered simultaneously.
        In this case, a decision and score for each histogram should still be provided (for consitency
        with algorithms which make per-histogram decisions)
        

        :param histograms: list of Histograms to make a decision on
        :type histograms: Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        :param metadata: metadata for this algorithm (e.g. training config dict)
        :type metadata: dict
        :return: dictionary of histogram names with two entries for each histogram, decision (bool) and score (float)
        :rtype: dict
        """
        raise NotImplementedError()
