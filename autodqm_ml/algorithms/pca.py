import pandas
import numpy
import json

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
        self.model = {}
        for histogram in self.histogram_info:
            name = histogram.name
            filename = name.split('/')[-1]
            pca = decomposition.PCA(
                    #n_components = n_components,
                    random_state = 0, # fixed for reproducibility                                                                                                   
            )
            pcaParams = json.load(open(f'{model_file}/{filename}.json','r'))
            pca.components_ = numpy.array(pcaParams['components_'])
            pca.explained_variance_ = numpy.array(pcaParams['explained_variance_'])
            pca.explained_variance_ratio_ = numpy.array(pcaParams['explained_variance_ratio_'])
            pca.singular_values_ = numpy.array(pcaParams['singular_values_'])
            pca.mean_ = numpy.array(pcaParams['mean_'])
            pca.n_components_ = numpy.array(pcaParams['n_components_'])
            pca.n_features_ = numpy.array(pcaParams['n_features_'])
            pca.n_samples_ = numpy.array(pcaParams['n_samples_'])
            pca.noise_variance_ = numpy.array(pcaParams['noise_variance_'])
            self.model[name] = pca


    def save_model(self, model_file, **kwargs):
        for histogram in self.histogram_info:
            name = histogram.name
            pca = self.model[name]
            filename = name.split('/')[-1]
            pcaParams = {
                'name' : filename,
                'components_' : pca.components_.tolist(),
                'explained_variance_' : pca.explained_variance_.tolist(),
                'explained_variance_ratio_' : pca.explained_variance_ratio_.tolist(),
                'singular_values_' : pca.singular_values_.tolist(),
                'mean_' : pca.mean_.tolist(),
                'n_components_' : pca.n_components_,
                'n_features_' : pca.n_features_, 
                'n_samples_' : pca.n_samples_, 
                'noise_variance_' : pca.noise_variance_
            }
            json.dump(pcaParams, open(f'{model_file}/{filename}.json','w'),indent=4)
        
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

            #------------------------------------
            #pca.components_ = numpy.zeros_like(pca.components_)
            # newpca = decomposition.PCA(
            #         n_components = n_components,
            #         random_state = 0, # fixed for reproducibility                                                             
            # )
            # newpca.components_ = pca.components_ 
            # newpca.explained_variance_ = pca.explained_variance_ 
            # newpca.explained_variance_ratio_ = pca.explained_variance_ratio_
            # newpca.singular_values_ = pca.singular_values_
            # newpca.mean_ = pca.mean_
            # newpca.n_components_ = pca.n_components_
            # newpca.n_features_ = pca.n_features_
            # newpca.n_samples_ = pca.n_samples_
            # newpca.noise_variance_ = pca.noise_variance_
            # 
            # self.model[name] = newpca
            #-----------------------------------

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

    

