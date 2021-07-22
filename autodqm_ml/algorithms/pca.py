import pandas
import numpy

import logging
logger = logging.getLogger(__name__)

from sklearn import decomposition

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.data_formats.histogram import Histogram

class PCA(MLAlgorithm):
    """
    PCA-based anomaly detector
    """
    def load_model(self, model_file, **kwargs):
        """
        TODO
        """
        pass


    def save_model(self, model_file, **kwargs):
        """
        TODO
        """
        pass 

    
    def train(self, n_components = 2, config = {}):
        """

        """

        self.model = {}
        for histogram in self.histogram_info:
            name = histogram.name
            pca = decomposition.PCA(
                    n_components = n_components,
                    random_state = 0, # fixed for reproducibility
            )

            inputs = self.data[name]["X_train"]

            logger.debug("[PCA : train] Training PCA with %d principal components for histogram '%s' with %d training examples." % (n_components, name, len(inputs)))
            pca.fit(inputs)
            self.model[name] = pca


    def evaluate_run(self, histograms, threshold = None, reference = None, metadata = {}):
        if threshold is None:
            threshold = 0.00001 # FIXME hard-coded for now

        results = {}

        for histogram in histograms:
            # Get original histogram
            original_hist = numpy.array(histogram.data).reshape(1, -1)

            # Transform to latent space
            transformed_hist = self.model[histogram.name].transform(original_hist)

            # Reconstruct latent representation back in original space
            reconstructed_hist = self.model[histogram.name].inverse_transform(transformed_hist)

            # Take mean squared error between original and reconstructed histogram
            sse = numpy.mean(
                    numpy.square(original_hist - reconstructed_hist)
            )

            results[histogram.name] = {
                    "score" : sse,
                    "decision" : sse > threshold
            }

        return results

    

