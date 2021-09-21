import pandas
import numpy

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.plotting.plot_tools import plot1D, plotMSESummary

#from autodqm_ml.plotting.plots import plot_original_vs_reconstructed

class AutoEncoder(MLAlgorithm):
    """
    Autoencoder base class.
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


    def train(self, n_epochs = 1000, batch_size = 128, config = {}):
        """

        """
        inputs, outputs = self.make_inputs(split = "train")
        inputs_val, outputs_val = self.make_inputs(split = "test")

        self.model = AutoEncoder_DNN(self.histogram_info, **config).model()
       
        self.model.compile(
                optimizer = keras.optimizers.Adam(), 
                loss = keras.losses.MeanSquaredError()
        )

        self.model.fit(
                inputs,
                outputs,
                validation_data = (inputs_val, outputs_val),
                callbacks = [keras.callbacks.EarlyStopping(patience = 3)],
                epochs = n_epochs,
                batch_size = batch_size
        )

    
    def predict(self, batch_size = 1024):
        inputs, outputs = self.make_inputs(split = "test")
        return self.model.predict(inputs, batch_size = batch_size)


    def evaluate_run(self, histograms, threshold = None, reference = None, metadata = {}):
        if threshold is None:
            threshold = 0.00001 # FIXME hard-coded for now

        inputs, outputs = self.make_inputs(histograms = histograms)
        pred = self.model.predict(inputs, batch_size = 1024)

        sse = self.model.evaluate(inputs, outputs, batch_size = 1024) 

        results = {}
        for idx, histogram in enumerate(self.histogram_info):
            score = sse[idx+1] 
            results[histogram.name] = {
                    "score" : score,
                    "decision" : score > threshold
            }

        results["global"] = {
                "score" : sse[0],
                "decision" : sse[0] > threshold
        }

        return results


    def plot(self, runs, histograms = None, threshold = None):
        """
        Plots reconstructed histograms on top of original histograms. If MSE between plotted histograms are ablove the threshold, SE plots will be constructed for the histogram

        :param runs: Runs to be plotted
        :type runs: list of int
        :param histograms: names of histograms to be plotted. Must match the names used by load_model/train. If histograms = None, all trained histograms in the pca class will be plotted
        :type histograms: list of str. Default histograms = None
        :param threshold: threshold to identify histogram as anomalous. If None, threshold will be set to 0.00001. 
        :type threshold: float, Default threshold = None
        """
        if threshold==None:
            threshold = 0.0001
        if histograms==None:
            histograms = self.histograms 
            
        original_hists = []
        reconstructed_hists = []
        for run in runs:
            hists = []
            for histogram in histograms:
                h = Histogram(
                    name = histogram, 
                    data = self.df[self.df["run_number"] == run][histogram].iloc[0]
                    )
                hists.append(h)
            inputs, outputs = self.make_inputs(histograms = hists)
            pred = self.model.predict(inputs, batch_size = 1024)
            
            #sse = self.model.evaluate(inputs, outputs, batch_size = 1024)
            
            inputslist = list(inputs.values())
            
            for i,x in enumerate(pred):
                # plot1D takes (n, ) shape so need to flatten
                original_hist = inputslist[i].numpy().flatten()
                original_hists.append(original_hist)
                # pred[i] has the shape (1,n,1) while iputslist[i] has shape (1,n), so reshape
                reconstructed_hist = x[:,:,0].flatten()
                reconstructed_hists.append(reconstructed_hist)
                plot1D(original_hist, reconstructed_hist, run, histograms[i], self.name, threshold)
        plotMSESummary(original_hists, reconstructed_hists, threshold, histograms, runs, self.name)
        

    def make_inputs(self, split = None, histograms = None, N = None):
        """

        """
        inputs = {}
        outputs = {}

        if split is None and histograms is not None:
            names_in = [h.name for h in histograms]
            if not names_in == self.histograms:
                logger.exception("[AutoEncoder : make_inputs] List of histograms given (%s) does not match the list of histograms this AutoEncoder was created with (%s)." % (names_in, self.histograms))
                raise ValueError()

            hists = histograms

        elif split is not None and histograms is None:
            hists = self.histogram_info


        for histogram in hists:
            if split is not None and histograms is None:
                data = self.data[histogram.name]["X_%s" % split]
            
            elif split is None and histograms is not None:
                data = [histogram.data]

            data = tf.convert_to_tensor(data)
            inputs["input_" + histogram.name_] = data
            outputs["output_" + histogram.name_] = data

        return inputs, outputs


class AutoEncoder_DNN(keras.models.Model):
    """
    Model defined through the Keras Model Subclassing API: https://www.tensorflow.org/guide/keras/custom_layers_and_models
    An AutoEncoder instance owns a single AutoEncoder_DNN, which is the actual implementation of the DNN.

    :param histogram_info: 
    :type histogram_info: 
    :param n_hidden_layers: number of hidden layers in encoder/decoder
    :type n_hidden_layers: int
    :param n_nodes: number of nodes per hidden layer
    :type n_nodes: int
    :param n_latent_dim: dimensionality of latent space
    :type n_latent_dim: int
    """
    def __init__(self, histogram_info, n_hidden_layers = 2, n_nodes = 100, n_latent_dim = 20, n_filters = 16, **kwargs):
        super(AutoEncoder_DNN, self).__init__()

        self.n_histograms = len(histogram_info)
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes = n_nodes
        self.n_latent_dim = n_latent_dim
        self.n_filters = n_filters

        self.inputs = []
        self.outputs = []
        self.encoders = []
        self.decoders = []
        for histogram in histogram_info: 
            input, encoder = self.build_encoder(histogram)
            self.inputs.append(input)
            self.encoders.append(encoder)

        encoder_outputs = [x for x in self.encoders]

        if self.n_histograms > 1:
            encoders_merged = keras.layers.concatenate(encoder_outputs)
        else:
            encoders_merged = encoder_outputs[0]

        layer = encoders_merged
        for i in range(self.n_hidden_layers):
            layer = keras.layers.Dense(
                    units = self.n_nodes,
                    activation = "relu",
                    name = "hidden_%d" % i,
            )(layer)

        latent_representation = keras.layers.Dense(
                units = self.n_latent_dim,
                activation = None,
                name = "latent_representation"
        )(layer)
        
        for histogram in histogram_info:
            output = self.build_decoder(histogram, latent_representation)
            self.outputs.append(output)


    def model(self):
        model = keras.models.Model(inputs = self.inputs, outputs = self.outputs)
        model.summary()
        return model


    def build_encoder(self, histogram):
        input = keras.layers.Input(
                shape = histogram.shape + (1,), 
                name = "input_%s" % histogram.name_
        )

        layer = input
        for i in range(self.n_hidden_layers):
            name = "encoder_%d_%s" % (i, histogram.name_)
            if histogram.n_dim == 1:
                layer = keras.layers.Conv1D(
                        filters = self.n_filters,
                        kernel_size = 3,
                        strides = 1,
                        activation = "relu",
                        name = name 
                )(layer)
            elif histogram.n_dim == 2:
                layer = keras.layers.Conv2D(
                        filters = self.n_filters,
                        kernel_size = 3,
                        strides = 1,
                        activation = "relu",
                        name = name
                )(layer)
        
        encoder = keras.layers.Flatten()(layer)
        return input, encoder


    def build_decoder(self, histogram, input):
        n_output_units = histogram.n_bins * self.n_filters 
        layer = keras.layers.Dense(
                units = n_output_units,
                activation = "relu",
                name = "decoder_input_%s" % histogram.name_
        )(input)
        target_shape = histogram.shape + (self.n_filters,) 
        layer = keras.layers.Reshape(target_shape = target_shape)(layer)

        for i in range(self.n_hidden_layers):
            if i == (self.n_hidden_layers - 1):
                activation = "relu"
                n_filters = 1
                name = "output_%s" % (histogram.name_)
            else:
                activation = "relu"
                n_filters = 32
                name = "decoder_%d_%s" % (i, histogram.name_)

            if histogram.n_dim == 1:
                layer = keras.layers.Conv1DTranspose(
                        filters = n_filters,
                        kernel_size = 3,
                        strides = 1,
                        padding = "same",            
                        activation = activation,
                        name = name 
                )(layer)
            elif histogram.n_dim == 2:
                layer = keras.layers.Conv2DTranspose(
                        filters = n_filters,
                        kernel_size = 3,
                        strides = 1,
                        padding = "same", 
                        activation = activation,
                        name = name
                )(layer)

        output = layer
        return output
