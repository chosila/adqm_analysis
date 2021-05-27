from scipy.stats import kstest

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

class StatisticalTester(AnomalyDetectionAlgorithm):
    """
    Class to perform statistical-based tests for judging anomalies.
    For 1d histograms, it will perform a ks-test.
    For 2d histograms, it will perform a pull-value test.
    """
    def evaluate(self, histograms, threshold, metadata = {}):
        results = {}

        for histogram in histograms:
            # Check that histogram has a reference
            if not hasattr(histogram, "reference") or histogram.reference is None:
                message = "[StatisticalTester : evaluate] Histogram %s does not have a valid reference." % (histogram.name)
                self.logger.exception(message)
                raise Exception(message)
 
            # Normalize histogram (if not already normalized)
            if metadata["normalize"]:
                if not histogram.is_normalized:
                    histogram.normalize()
                if not histogram.reference.is_normalized:
                    histogram.reference.normalize()
            
            if histogram.n_dim == 1:
                decision, score = self.ks_test(histogram, threshold)
            elif histogram.n_dim == 2:
                decision, score = self.pull_value_test(histogram, threshold)

            results[histogram.name] = {
                    "decision" : decision,
                    "score" : score
            }

        return results


    def ks_test(self, histogram, threshold):
        """
        Perform ks test on two 1d histograms.

        :param histogram: Histogram object
        :type histogram: autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        """

        ks_test_results = kstest(
                histogram.reference.data["values"],
                histogram.data["values"]
        )

        score = ks_test_results[0]
        decision = score > threshold

        return decision, score 

    def pull_value_test(self, histogram, threshold):
        """
        Perform pull value test on two 2d histograms.

        :param histogram: Histogram object
        :type histogram: autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        """

        # TODO
        return False, 0
