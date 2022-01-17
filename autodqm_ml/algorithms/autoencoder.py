import os
import pandas
import numpy
import json
import awkward

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml import utils

DEFAULT_OPT = {
        "n_hidden_layers" : 2,
        "n_nodes" : 25,
        "n_components" : 3,
        "kernel_1d" : 3,
        "kernel_2d" : 3,
        "n_filters" : 8
}

class AutoEncoder(MLAlgorithm):
    """
    Autoencoder base class.
    """
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)

        self.config = utils.update_dict(
                original = DEFAULT_OPT,
                new = self.__dict__
        )


    def load_model(self, model_file):
        """

        """
        model = keras.models.load_model(model_file)
        return model


    def save_model(self, model, model_file):
        """

        """
        model.save(model_file)


    def train(self, n_epochs = 1000, batch_size = 128):
        """

        """
        model_file = "%s/autoencoder_%s.h5" % (self.output_dir, self.tag)
        if os.path.exists(model_file):
            logger.warning("[AutoEncoder : train] A trained AutoEncoder alread exists with tag '%s' at file '%s'. We will load the saved model from the file rather than retraining. If you wish to retrain please provide a new tag or delete the old outputs." % (self.tag, model_file))
            self.model = self.load_model(model_file)
            return

        inputs, outputs = self.make_inputs(split = "train")
        inputs_val, outputs_val = self.make_inputs(split = "test")

        self.model = AutoEncoder_DNN(self.histograms, **self.config).model()
       
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
        self.save_model(self.model, model_file)

    
    def predict(self, batch_size = 1024):
        inputs, outputs = self.make_inputs(split = "all")
        pred = self.model.predict(inputs, batch_size = batch_size)

        idx = 0
        for histogram, histogram_info in self.histograms.items():
            original_hist = self.df[histogram]
            if len(self.histograms.items()) >= 2:
                reconstructed_hist = awkward.flatten(awkward.from_numpy(pred[idx]), axis = -1) 
            else:
                reconstructed_hist = awkward.flatten(awkward.from_numpy(pred), axis = -1)

            sse = awkward.sum(
                    (original_hist - reconstructed_hist) ** 2,
                    axis = -1
            )

            self.add_prediction(histogram, sse, reconstructed_hist)
            idx += 1


    def make_inputs(self, split = None):
        """

        """
        inputs = {}
        outputs = {}

        if split == "train":
            cut = self.df.train_label == 0
        elif split == "test":
            cut = self.df.train_label == 1
        else:
            cut = self.df.train_label >= 0

        df = self.df[cut]

        for histogram, info in self.histograms.items():
            data = tf.convert_to_tensor(df[histogram])
            inputs["input_" + info["name"]] = data
            outputs["output_" + info["name"]] = data

        return inputs, outputs


class AutoEncoder_DNN(keras.models.Model):
    """
    Model defined through the Keras Model Subclassing API: https://www.tensorflow.org/guide/keras/custom_layers_and_models
    An AutoEncoder instance owns a single AutoEncoder_DNN, which is the actual implementation of the DNN.

    """
    def __init__(self, histograms, **kwargs): 
        super(AutoEncoder_DNN, self).__init__()

        self.n_histograms = len(histograms.keys())

        self.__dict__.update(kwargs)

        self.inputs = []
        self.outputs = []
        self.encoders = []
        self.decoders = []
        for histogram, info in histograms.items(): 
            input, encoder = self.build_encoder(histogram, info)
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
                units = self.n_components,
                activation = None,
                name = "latent_representation"
        )(layer)
        
        for histogram, info in histograms.items(): 
            output = self.build_decoder(histogram, info, latent_representation)
            self.outputs.append(output)


    def model(self):
        model = keras.models.Model(inputs = self.inputs, outputs = self.outputs)
        model.summary()
        return model


    def build_encoder(self, histogram, info):
        input = keras.layers.Input(
                shape = info["shape"] + (1,), 
                name = "input_%s" % info["name"]
        )

        layer = input
        for i in range(self.n_hidden_layers):
            name = "encoder_%d_%s" % (i, info["name"])
            if info["n_dim"] == 1:
                layer = keras.layers.Conv1D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_1d,
                        strides = 1,
                        activation = "relu",
                        name = name 
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_2d,
                        strides = 1,
                        activation = "relu",
                        name = name
                )(layer)
        
        encoder = keras.layers.Flatten()(layer)
        return input, encoder


    def build_decoder(self, histogram, info, input):
        n_output_units = info["n_bins"] * self.n_filters 
        layer = keras.layers.Dense(
                units = n_output_units,
                activation = "relu",
                name = "decoder_input_%s" % info["name"] 
        )(input)
        target_shape = info["shape"] + (self.n_filters,) 
        layer = keras.layers.Reshape(target_shape = target_shape)(layer)

        for i in range(self.n_hidden_layers):
            if i == (self.n_hidden_layers - 1):
                activation = "relu"
                n_filters = 1
                name = "output_%s" % (info["name"])
            else:
                activation = "relu"
                n_filters = self.n_filters 
                name = "decoder_%d_%s" % (i, info["name"])

            if info["n_dim"] == 1:
                layer = keras.layers.Conv1DTranspose(
                        filters = n_filters,
                        kernel_size = self.kernel_1d,
                        strides = 1,
                        padding = "same",            
                        activation = activation,
                        name = name 
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2DTranspose(
                        filters = n_filters, 
                        kernel_size = self.kernel_2d,
                        strides = 1,
                        padding = "same", 
                        activation = activation,
                        name = name
                )(layer)

        output = layer
        return output
