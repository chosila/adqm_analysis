import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from autodqm_ml.algorithms.anomaly_detection_algorithm import AnomalyDetectionAlgorithm

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


    def train(self, config, n_epochs = 10, batch_size = 1024):
        """

        """
        self.model = AutoEncoder_DNN(**config)
        inputs = self.load_training_data(**config)
        self.model.fit(
                inputs,
                epochs = n_epochs,
                batch_size = batch_size
        )


    def load_training_data(self, **kwargs):
        """

        """

        return inputs

class AutoEncoder_DNN(keras.Model)
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
    def __init__(self, histogram_info, n_hidden_layers = 3, n_nodes = 50, n_latent_dim = 4, **kwargs):
        super(AutoEncoder_DNN, self).__init__()

        self.n_histograms = n_histograms
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes = n_nodes
        self.n_latent_dim = n_latent_dim

        self.encoders = []
        self.decoders = []
        for histogram in histogram_info:
            if histogram["n_dim"] == 1:
                self.encoders.append(self.build_1d_encoder(histogram))
                self.decoders.append(self.build_1d_decoder(histogram))
            elif histogram["n_dim"] == 2:
                self.encoders.append(self.build_2d_encoder(histogram))
                self.decoders.append(self.build_2d_decoder(histogram))

        encoder_outputs = [x for x in self.encoders]
        encoders_merged = keras.layers.Concatenate()(encoder_outputs)
        
        layer = encoders_merged
        for i in range(n_hidden_layers):
            layer = keras.layers.Dense(
                    units = n_nodes,
                    activation = "relu",
                    name = "hidden_%d" % i
            )(layer)

        


    def call(self, x):
        """

        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


    def build_1d_encoder(self, histogram):
        input = keras.layers.Input(
                shape = histogram["shape"],
                name = "input_%s" % histogram["name"]
        )

        layer = input
        for i in range(n_hidden_layers):
            layer = keras.layers.Conv1D(
                    filters = 32,
                    kernel_size = 3,
                    strides = 1,
                    activation = "relu",
                    name = "encoder_%d_%s" % (i, histogram["name"])
            )(layer)
        
        output = keras.layers.Flatten()(layer)
        return output 

    def build_1d_decoder(self, histogram):
        input = keras.layers.Input(
                shape = (self.n_latent_dim,)
        )

        n_output_units = histogram["shape"][0] * 32
        layer = keras.layers.Dense(
                units = n_output_units,
                activation = "relu",
                name = "input_%s" % histogram["name"]
        )(input)
        target_shape = histogram["shape"] + (32,)
        layer = keras.layers.Reshape(target_shape = target_shape)(layer)

        for i in range(n_hidden_layers-1):
            activation = None if i == (n_hidden_layers - 1) else "relu" # no activation on last step
            layer = keras.layers.Conv1DTranspose(
                    filters = 32,
                    kernel_size = 3,
                    strides = 1,
                    padding = "same",            
                    activation = activation,
                    name = "decoder_%d_%s" % (i, histogram["name"])
            )

        output = layer
        return output 

    def build_encoder(self, histogram):
        """

        """
        input = keras.layers.Input(shape = histogram["shape"])




    def build_decoder(self):
        self.decoder = None




