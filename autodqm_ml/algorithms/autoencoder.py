import pandas
import numpy

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.data_formats.histogram import Histogram

class AutoEncoder(MLAlgorithm):
    """
    Autoencoder base class.
    """
    def evaluate_with_model(self, histograms, threshold, metadata):
        results = {}

        return results


    def load_model(self, model_file, **kwargs):
        """

        """


    def train(self, histograms, file, config, n_epochs = 10, batch_size = 128):
        """

        """
        self.training_file = file

        inputs, outputs = self.load_data(histograms)

        self.model = AutoEncoder_DNN(self.histogram_info, **config).model()

        self.model.compile(
                optimizer = keras.optimizers.Adam(), 
                loss = keras.losses.MeanSquaredError()
        )

        self.model.fit(
                inputs,
                outputs,
                epochs = n_epochs,
                batch_size = batch_size
        )

        # TODO: save model


    def load_data(self, histograms):
        """
        1. Load input file
        2. Get metadata for each of the histograms
        3. Create arrays

        """
        df = pandas.read_pickle(self.training_file)

        # Get metadata from 0th entry in dataframe
        self.histogram_info = []
        for histogram in histograms:
            h = Histogram(
                name = histogram,
                data = df[histogram][0]
            )
            h.name_ = h.name.replace("/", "_").replace(" ", "") # tf doesn't like the "/"s or spaces
            print(h.name_)
            self.histogram_info.append(h)
            logger.debug("[AutoEncoder : load_data] Found histogram '%s' in input file '%s', with details: shape: %s, n_dim: %d" % (histogram, self.training_file, str(h.shape), h.n_dim))
            

        # Create arrays
        inputs = {}
        outputs = {}
        for histogram in self.histogram_info:
            # Normalize
            for i in range(len(df)): # TODO: come up with more efficient way than looping through df
                h = Histogram(name = "dummy", data = df[histogram.name][i])
                h.normalize()
                df[histogram.name][i] = h.data
            data = list(df[histogram.name].values)
            data = tf.convert_to_tensor(data) # TODO: figure out how to deal with 2d histograms where n_bins_x != n_bins_y
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
    def __init__(self, histogram_info, n_hidden_layers = 2, n_nodes = 200, n_latent_dim = 20, n_filters = 16, **kwargs):
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
            activation = None if i == (n_hidden_layers - 1) else "relu" # no activation on last step
            units = n_nodes if i == (n_hidden_layers - 1) else n_latent_dim
            name = "latent_representation" if i == (n_hidden_layers - 1) else "hidden_%d" % i
            layer = keras.layers.Dense(
                    units = units,
                    activation = activation,
                    name = name 
            )(layer)

        latent_representation = layer
        
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
                activation = None
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


    def build_2d_encoder(self, histogram):
        """
        TODO
        """
        return

    
    def build_2d_decoder(self, histogram):
        """
        TODO
        """
        return



