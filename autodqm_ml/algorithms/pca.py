import pandas
import numpy
import json
import matplotlib.pyplot as plt
#from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from sklearn import decomposition
from pathlib import Path

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.plotting.plot_tools import plot1D

class PCA(MLAlgorithm):
    """
    PCA-based anomaly detector
    """
    def load_model(self, model_file, **kwargs):
        """
        Load PCA model from pickle file

        :param model_file: folder containing PCA pickles
        :type model_file: str
        
        """
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
            ## delcaire model instance
            self.model[name] = pca


    def save_model(self, model_file, **kwargs):
        """
        Save a trained PCA model

        :param model_file: folder name to place trained PCA pickles
        :type model_file: str
        """
        Path(model_file).mkdir(parents=True, exist_ok=True)
        for histogram in self.histogram_info:
            name = histogram.name
            pca = self.model[name]
            ## used for nameing json file, and labeling the file inside json
            filename = name.split('/')[-1]
            ## info that is going into JSON file
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
        Trains new PCA models using loaded data. Must call pca.load_data() before training. 

        :param n_components: number of components to keep. If None, all components are kept. 
        :type n_components: int, default = 2
        
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



    def plot(self, runs, histograms=None, threshold=None):
        """
        Plots reconstructed histograms on top of original histograms. If the SSE between the plotted histograms are above the threshold, SSE plots will also be made.Either pca.train() or pca.load_model() must be called before plot. 
        
        :param runs: Runs to be plotted
        :type runs: list of int
        :param histograms: names of histograms to be plotted. Must match the names used by load_model/train. If histograms = None, all trained histograms in the pca class will be plotted
        :type histograms: list of str. Default histograms = None
        :param threshold: threshold to identify histogram as anomalous. If None, threshold will be set to 0.00001. 
        :type threshold: float, Default threshold = None
        """
        # threshold hardcoded for now
        if threshold==None:
            threshold = 0.00001

        if histograms==None:
            histograms = self.histograms


        # loop over list of histograms to plot
        for histogram in histograms:#self.histograms:
            # loop over runs to plot
            for run in runs:                
                h = Histogram(
                            name = histogram,
                            data = self.df[self.df["run_number"] == run][histogram].iloc[0],
                    )
                
                original_hist = numpy.array(h.data).reshape(1, -1)
                
                # Transform to latent space                                                                                                                         
                transformed_hist = self.model[h.name].transform(original_hist)
                
                # Reconstruct latent representation back in original space                                                                                          
                reconstructed_hist = self.model[h.name].inverse_transform(transformed_hist)
                
                # plot1D takes array of shape (n,), but original and reco have shape (n,1)
                plot1D(original_hist.flatten(), reconstructed_hist.flatten(), run, h.name, 'pca', threshold)


 
