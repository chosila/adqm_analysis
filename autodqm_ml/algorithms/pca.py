import os
import pandas
import numpy
import json
import matplotlib.pyplot as plt
import awkward
#from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from sklearn import decomposition
from pathlib import Path

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.plotting.plot_tools import plot1D, plotMSESummary

DEFAULT_OPT = {
        "n_components" : 2
}

class PCA(MLAlgorithm):
    """
    PCA-based anomaly detector
    """
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)

        if not hasattr(self, "n_components"):
            self.n_components = DEFAULT_OPT["n_components"]

    def load_model(self, model_file):
        """
        Load PCA model from pickle file

        :param model_file: folder containing PCA pickles
        :type model_file: str
        
        """
        with open(model_file, "r") as f_in:
            pcaParams = json.load(f_in)

        pca = decomposition.PCA(random_state = 0)

        pca.components_ = numpy.array(pcaParams['components_'])
        pca.explained_variance_ = numpy.array(pcaParams['explained_variance_'])
        pca.explained_variance_ratio_ = numpy.array(pcaParams['explained_variance_ratio_'])
        pca.singular_values_ = numpy.array(pcaParams['singular_values_'])
        pca.mean_ = numpy.array(pcaParams['mean_'])
        pca.n_components_ = numpy.array(pcaParams['n_components_'])
        pca.n_features_ = numpy.array(pcaParams['n_features_'])
        pca.n_samples_ = numpy.array(pcaParams['n_samples_'])
        pca.noise_variance_ = numpy.array(pcaParams['noise_variance_'])

        return pca

    def save_model(self, pca, model_file):
        """
        Save a trained PCA model

        :param pca: trained pca
        :type pca: `sklearn.decomposition.PCA`
        :param model_file: folder name to place trained PCA pickles
        :type model_file: str
        """
        os.system("mkdir -p %s" % self.output_dir)
        pcaParams = {
                'name' : model_file.split("/")[-1].replace(".json", ""),
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

        with open(model_file, "w") as f_out:
            json.dump(pcaParams, f_out, indent = 4, sort_keys = True)       
 

    def train(self):
        """
        Trains new PCA models using loaded data. Must call pca.load_data() before training. 
        """

        self.model = {}

        for histogram, histogram_info in self.histograms.items():
            model_file = "%s/pca_%s_%s.json" % (self.output_dir, histogram_info["name"], self.tag)
            if os.path.exists(model_file):
                logger.warning("[PCA : train] A trained PCA already exists for histogram '%s' with tag '%s' at file '%s'. We will load the saved model from the file rather than retraining. If you wish to retrain, please provide a new tag or delete the old outputs." % (histogram, self.tag, model_file))
                self.model[histogram] = self.load_model(model_file)
                continue

            pca = decomposition.PCA(
                    n_components = self.n_components,
                    random_state = 0, # fixed for reproducibility
            )
            
            input = self.df[self.df.train_label == 0][histogram]

            logger.debug("[PCA : train] Training PCA with %d principal components for histogram '%s' with %d training examples." % (self.n_components, histogram, len(input)))

            pca.fit(input)
            self.model[histogram] = pca
            
            logger.debug("[PCA : train] Saving trained PCA to file '%s'." % (model_file))
            self.save_model(pca, model_file)

    
    def predict(self, **kwargs):
        """

        """
        for histogram, histogram_info in self.histograms.items():
            pca = self.model[histogram]
            
            # Grab the original histograms and transform to latent space
            original_hist = self.df[histogram]
            original_hist_transformed = pca.transform(original_hist)

            # Reconstruct histogram from latent space representation
            reconstructed_hist = pca.inverse_transform(original_hist_transformed)
            
            # Calculate sse
            sse = awkward.sum(
                    (original_hist - reconstructed_hist) ** 2,
                    axis = -1
            )

            self.add_prediction(histogram, sse, reconstructed_hist)

