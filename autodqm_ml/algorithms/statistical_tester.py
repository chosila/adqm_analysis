import numpy
from scipy.stats import ks_2samp

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

import logging
logger = logging.getLogger(__name__)

class StatisticalTester(AnomalyDetectionAlgorithm):
    """
    Class to perform statistical-based tests for judging anomalies.
    For 1d histograms, it will perform a ks-test.
    For 2d histograms, it will perform a pull-value test.
    """

    def __init__(self, **kwargs):
        super(StatisticalTester, self).__init__(**kwargs)
        self.reference = kwargs.get("reference", None)

    def predict(self):
        if self.reference is None:
            self.reference = self.df.run_number[0]
            logger.warning("[StatisticalTester : predict] No reference run was provided, will use the first run in the df (%d) as the reference." % (self.reference))
        
        self.reference_hists = self.df[self.df.run_number == self.reference][0] 

        for histogram, info in self.histograms.items():
            score = numpy.zeros(len(self.df))
            for i in range(len(score)):
                if info["n_dim"] == 1:
                    d_value, p_value = self.ks_test(self.df[histogram][i], self.reference_hists[histogram])
                    score[i] = d_value
                elif info["n_dim"] == 2:
                    pull_value = self.pull_value_test(self.df[histogram][i], self.reference_hists[histogram])
                    score[i] = pull_value

            self.add_prediction(histogram, score)


    def ks_test(self, target, reference):
        """
        Perform ks test on two 1d histograms.
        """

        score, p_value = ks_2samp(
                numpy.array(target), 
                numpy.array(reference) 
        )

        return score, p_value


    def pull_value_test(self, target, reference):
        """
        Perform pull value test on two 2d histograms.

        :param histogram: Histogram object
        :type histogram: autodqm_ml.data_formats.histogram.Histogram
        :param threshold: value for which to declare a histogram anomalous
        :type threshold: float
        """

        # TODO
        return 0
