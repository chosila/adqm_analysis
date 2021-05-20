from scipy.stats import kstest

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

class KSTest(AnomalyDetectionAlgorithm):
    def evaluate(self, histograms, threshold, metadata = {}):
        results = { "anomalous" : [], "d_score" : [] } 
        for histogram in histograms:
            # Check that histogram is 1d
            if histogram.n_dim != 1:
                message = "[KSTest : evaluate] Histograms for KSTest must have n_dim = 1, but histogram %s has n_dim = %d." % (histogram.name, histogram.n_dim)
                self.logger.exception(message)
                raise Exception(message)

            # Check that histogram has a reference
            if not hasattr(histogram, "reference") or histogram.reference is None:
                message = "[KSTest : evaluate] Histogram %s does not have a valid reference." % (histogram.name)
                self.logger.exception(message)
                raise Exception(message)

            # Check that histogram has enough entries
            if histogram.n_entries < metadata["min_entries"]:
                message = "[KSTest : evaluate] Histograms for KSTest must have at least %d entries, but histogram %s has %d entries." % (metadata["min_entries"], histogram.name, histogram.n_entries)
                self.logger.exception(message)
                raise Exception(message)

            # Normalize histogram (if not already normalized)
            if metadata["normalize"]:
                if not histogram.is_normalized:
                    histogram.normalize()
                if not histogram.reference.is_normalized:
                    histogram.reference.normalize()
   
            # Calculate ks test 
            ks_test_results = kstest(
                    histogram.reference.data["values"],
                    histogram.data["values"]
            )
            ks_test_d_value = ks_test_results[0]

            # Make decision
            anomalous = ks_test_d_value > threshold
            results["d_score"].append(ks_test_d_value)
            results["anomalous"].append(anomalous)

        return results
