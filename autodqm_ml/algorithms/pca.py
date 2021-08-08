import pandas
import numpy
import json
import matplotlib.pyplot as plt
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from sklearn import decomposition
from pathlib import Path

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
            ## delcaire model instance
            self.model[name] = pca


    def save_model(self, model_file, **kwargs):
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



    def plot(self, runs, reference, histograms=None, threshold=None):
        # threshold hardcoded for now
        if threshold==None:
            threshold = 0.00001

        if histograms==None:
            histograms = self.histograms

        # loop over list of histograms to plot
        for histogram in histograms:#self.histograms:
            # reference histogram
            h_ref = Histogram(
                    name = "%s_ref_Run%d" % (histogram, reference),
                    data = self.df[self.df["run_number"] == reference][histogram].iloc[0]
            )

            # loop over runs to plot
            for run in runs:
                fig,ax=plt.subplots()
                fig2,ax2 = plt.subplots()
                h = Histogram(
                            name = histogram,
                            data = self.df[self.df["run_number"] == run][histogram].iloc[0],
                            reference = h_ref if reference is not None else None
                    )
                
                original_hist = numpy.array(h.data).reshape(1, -1)
                
                # Transform to latent space                                                                                                                         
                transformed_hist = self.model[h.name].transform(original_hist)
                
                # Reconstruct latent representation back in original space                                                                                          
                reconstructed_hist = self.model[h.name].inverse_transform(transformed_hist)
                
                # Take mean squared error between original and reconstructed histogram                                                                                  
                sse = numpy.mean(
                        numpy.square(original_hist - reconstructed_hist)
                )
                
                # for bin edges
                binedges = numpy.linspace(0, 1, original_hist.shape[-1])
                width = binedges[1]-binedges[0]
                # plot 
                ax.bar(binedges, original_hist[0], alpha=0.5, label='original', width=width)
                ax.bar(binedges, reconstructed_hist[0], alpha=0.5, label='reconstructed', width=width)
                plotname = h.name.split('/')[-1]
                ax.set_title(f'{plotname} {run}')
                ax.legend(loc='best')
                # create directory to save plot
                Path(f'plots/{run}').mkdir(parents=True, exist_ok=True)
                fig.savefig(f'plots/{run}/{plotname}.png')
                fig.clf()
                
                if sse > threshold: 
                    ax2.bar(binedges, numpy.square(original_hist[0] - reconstructed_hist[0]), alpha=0.5, width=width)
                    ax2.set_title(f'SSE {plotname} {run}')
                    fig2.savefig(f'plots/{run}/SSE-{plotname}.png')
                    fig2.clf()


 
