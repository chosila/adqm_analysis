import os
import pandas
import numpy
import json
import awkward
import copy

import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.keras as keras

from datetime import datetime
from autodqm_ml.constants import kGOOD, kANOMALOUS
from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml import utils
from autodqm_ml.plotting.plot_tools import make_training_plots
DEFAULT_OPT = {
        "batch_size" : 128,
        "val_batch_size" : 1024,
        "learning_rate" : 0.001,
        "loss" : "mse",
        "n_epochs" : 1000,
        "early_stopping" : True,
        "early_stopping_rounds" : 3,
        "n_conv_layers": 1,
        "tied_dense": True,
        "n_hidden_layers": 1, 
        "n_nodes": 15,
        "n_components" : 3,
        "kernel_1d" : 3,
        "kernel_2d" : 3,
        "strides_1d" : 1,
        "strides_2d" : 1,
        "pooling" : False,
        "pooling_kernel_1d" : 2,
        "pooling_kernel_2d" : 2,
        "pooling_stride_1d" : None,
        "pooling_stride_2d" : None,
        "decoder_conv_layers" : True,
        "dropout" : 0.0,
        "batch_norm" : False,
        "n_filters" : 12,
        "retain_best":False,
        "lr_plateau_decay": False,
        "lr_plateau_decay_rounds": 5,
        "lr_plateau_decay_factor": 0.1,
        "lr_plateau_threshold": 0,
        "overwrite":False
}

class MirrorAE(MLAlgorithm):
    """
    Autoencoder base class.

    :param config: dictionary with hyperparameters for autoencoder training. Any hyperparameters not specified will be taken from the default values in `DEFAULT_OPT`
    :type config: dict
    :param mode: string to specify whether you want to train an autoencoder for each histogram ("individual") or a single autoencoder on all histograms ("simultaneous")
    :type mode: str
    """
    def __init__(self, **kwargs):
        super(MirrorAE, self).__init__(**kwargs)

        self.config = utils.update_dict(
                original = DEFAULT_OPT,
                new = kwargs.get('config', {})
        )

        self.mode = kwargs.get('autoencoder_mode', 'individual')
        if self.mode is None:
            self.mode = "individual"

        if not self.mode in ["individual", "simultaneous"]:
            logger.exception("[AutoEncoder : __init__] mode '%s' is not a recognized option for AutoEncoder. Currently available modes are 'individual' (default) and 'simultaneous'." % (self.mode))
            raise ValueError()
        self.models = {}

        logger.debug("[AutoEncoder : __init__] Constructing AutoEncoder with the following training options and hyperparameters:")
        for param, value in self.config.items():
            logger.debug("\t %s : %s" % (param, str(value)))


    def load_model(self, model_file):
        """

        """
        custom_layers = {'DenseTied': DenseTied}
        with keras.utils.custom_object_scope(custom_layers):
            model = keras.models.load_model(model_file)
        return model


    def save_model(self, model, model_file):
        """

        """
        logger.debug("[AutoEncoder : save_model] Saving trained autoencoder to file '%s'." % (model_file))
        model.save(model_file)


    def train(self):
        """

        """
        if self.mode == "simultaneous":
            self.models = { None : None }
            logger.debug("[AutoEncoder : train] Mode selected as 'simultaneous', meaning a single autoencoder will be trained simultaneously on all histograms. Use 'individual' if you wish to train one autoencoder for each histogram.")
        elif self.mode == "individual":
            self.models = { k : None for k,v in self.histograms.items() } #copy.deepcopy(self.histograms)
            logger.debug("[AutoEncoder : train] Mode selected as 'individual', meaning one autoencoder will be trained for each histogram. Use 'simultaneous' if you wish to train a single autoencoder for all histograms.")

        for histogram, histogram_info in self.models.items():
            if histogram is None:
                model_file = "%s/autoencoder_%s.h5" % (self.output_dir, self.tag)
            else:
                model_file = "%s/autoencoder_%s_%s.h5" % (self.output_dir, histogram, self.tag)

            if os.path.exists(model_file):
                self.models[histogram] = self.load_model(model_file)
                if self.config['overwrite']:
                    logger.warning("[AutoEncoder : train] Overwrite has been turned on and a trained AutoEncoder already exists with tag '%s' at file '%s'. We will load the saved model from the file and continue to train it." % (self.tag, model_file))
                    model = self.models[histogram]
                else:
                    logger.warning("[AutoEncoder : train] A trained AutoEncoder already exists with tag '%s' at file '%s'. We will load the saved model from the file rather than retraining. If you wish to retrain please provide a new tag or delete the old outputs." % (self.tag, model_file))
                    continue

            inputs, outputs = self.make_inputs(split = "train", histogram_name = histogram)
            inputs_val, outputs_val = self.make_inputs(split = "test", histogram_name = histogram)

            if histogram is None:
                hist_name = str(list(self.models.keys()))
            else:
                hist_name = histogram
            logger.debug("[AutoEncoder : train] Training autoencoder with %d dimensions in latent space for histogram(s) '%s' with %d training examples." % (self.config["n_components"], hist_name, len(list(inputs.values())[0])))

            if self.mode == "simultaneous":
                histograms = self.histograms
            elif self.mode == "individual":
                histograms = { histogram : self.histograms[histogram] }

            if not os.path.exists(model_file):
                 model = AutoEncoder_DNN(histograms, **self.config).model()
                 if self.config["overwrite"]:
                     logger.warning("[AutoEncoder : train] Overwrite has been turned on but no existing model was found with tag '%s' at file '%s'. We will create and train a new model." % (self.tag, model_file))
            model.summary()
            model.compile(
                    optimizer = keras.optimizers.Adam(learning_rate = self.config["learning_rate"]),
                    loss = self.config["loss"],
                    metrics = ['mse']
            )

            callbacks = []
            if self.config["early_stopping"]:
                callbacks.append(keras.callbacks.EarlyStopping(monitor = 'val_mse', patience = self.config["early_stopping_rounds"], restore_best_weights = self.config['retain_best']))
            if self.config["lr_plateau_decay"]:
                callbacks.append(keras.callbacks.ReduceLROnPlateau(patience = self.config["lr_plateau_decay_rounds"], factor = self.config["lr_plateau_decay_factor"], min_delta = self.config["lr_plateau_threshold"]))

            history = model.fit(
                    inputs,
                    outputs,
                    validation_data = (inputs_val, outputs_val),
                    callbacks = callbacks,
                    epochs = self.config["n_epochs"],
                    batch_size = self.config["batch_size"]
            )
            if self.stats_output_dir:
               if not os.path.isdir(self.stats_output_dir):
                   os.mkdir(self.stats_output_dir)
               history = pandas.DataFrame(history.history)
               runlabel = 'run_' + datetime.now().strftime('%H%M%S%m%d')
               logger.info("[AutoEncoder : train] Saving training statistics in '%s'." % (self.stats_output_dir))
               history.to_csv(self.stats_output_dir + runlabel + '_history.csv')
               history.to_parquet(self.stats_output_dir + runlabel + '_history.parquet')
               make_training_plots(history, histogram, self.stats_output_dir + runlabel + '_plots.png')



            self.save_model(model, model_file)
            self.models[histogram] = model


    def predict(self, batch_size = 1024):
        for histogram, model in self.models.items():
            inputs, outputs = self.make_inputs(split = "all", histogram_name = histogram)

            predictions = model.predict(inputs, batch_size = batch_size)
            if self.mode == "simultaneous" and self.n_histograms >= 2:
                predictions = { name : pred for name, pred in zip(model.output_names, predictions) }
            else:
                predictions = { model.output_names[0] : predictions }

            for name, pred in predictions.items():
                hist_name = self.histogram_name_map[name.replace("output_", "")] # shape [n_runs, histogram dimensions, 1]
                original_hist = self.df[hist_name] # shape [n_runs, histogram dimensions]

                reconstructed_hist = awkward.flatten( # change shape from [n_runs, histogram dimensions, 1] -> [n_runs, histogram dimensions]
                        awkward.from_numpy(pred),
                        axis = -1
                )
                #reconstructed_hist = pred
                sse = awkward.sum( # perform sum along inner-most axis, i.e. first histogram dimension
                        (original_hist - reconstructed_hist) ** 2,
                        axis = -1
                )

                # For 2d histograms, we need to sum over one more axis to get a single SSE score for each run
                if self.histograms[hist_name]["n_dim"] == 2:
                    sse = awkward.sum(sse, axis = -1) # second histogram dimension

                self.add_prediction(hist_name, sse, reconstructed_hist)

    def make_inputs(self, split = None, histogram_name = None):
        """

        """
        inputs = {}
        outputs = {}

        for histogram, info in self.histograms.items():
            if histogram_name is not None: # self.mode == "individual", i.e. separate autoencoder for each histogram
                if not histogram == histogram_name: # only grab the relevant histogram for this autoencoder
                    continue
            if 'CSC' in histogram_name:
                label_field = 'CSC_label'
            elif 'emtf' in histogram_name:
                label_field = 'EMTF_label'
            else:
                label_field = None

            if label_field and len(numpy.unique(self.df[label_field])) > 1: #Don't Include Anomalous Runs in Training
                if split == "train":
                    cut = [self.df.train_label[i] == 0 and self.df[label_field][i] == kGOOD for i in range(len(self.df))]
                elif split == "test":
                    cut = [self.df.train_label[i] == 0 and self.df[label_field][i] == kGOOD for i in range(len(self.df))]
                elif split == "all":
                    cut = self.df.run_number >= 0
                else:
                    cut = [l == kGOOD for l in self.df[label_field]]
            else:
                if split == "train":
                    cut = self.df.train_label == 0
                elif split == "test":
                    cut = self.df.train_label == 1
                else:
                    cut = self.df.run_number >= 0 # dummy all True cut

            df = self.df[cut]
            data = tf.convert_to_tensor(df[histogram])
            inputs["input_" + info["name"]] = data
            outputs["output_" + info["name"]] = data

        return inputs, outputs


class AutoEncoder_DNN():
    """
    An AutoEncoder instance owns AutoEncoder_DNN(s), which is the actual implementation of the DNN.
    """
    def __init__(self, histograms, **kwargs):
        self.n_histograms = len(histograms.keys())

        self.__dict__.update(kwargs)
        self.inputs = []
        self.outputs = []
        self.encoders = []
        self.decoders = []
        self.coupled = {}
        self.conv_sizes = {}
        for histogram, info in histograms.items():
            input, encoder = self.build_encoder(histogram, info)
            self.inputs.append(input)
            self.encoders.append(encoder)
            output = self.build_decoder(histogram, info, encoder)
            self.outputs.append(output)


    def model(self):
        model = keras.models.Model(
                inputs = self.inputs,
                outputs = self.outputs,
                name = "autoencoder"
        )
        #model.summary()
        return model


    def build_encoder(self, histogram, info):        
        input = keras.layers.Input(
                shape = info["shape"] + (1,),
                name = "input_%s" % info["name"]
        )
        self.conv_sizes[histogram] = [tf.shape(input)._inferred_value[1:]]
        layer = input
        for i in range(self.n_conv_layers):
            name = "encoder_%d_%s" % (i, info["name"])
            if info["n_dim"] == 1:
                layer = keras.layers.Conv1D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_1d,
                        strides = self.strides_1d,
                        activation = "relu",
                        name = name
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2D(
                        filters = self.n_filters,
                        kernel_size = self.kernel_2d,
                        use_bias = False,
                        strides = self.strides_2d,
                        activation = "relu",
                        name = name
                )(layer)
            self.conv_sizes[histogram].append(tf.shape(layer)._inferred_value[1:])
            if self.batch_norm:
                layer = keras.layers.BatchNormalization(name = name + "_batch_norm")(layer)
            if self.dropout > 0:
                layer = keras.layers.Dropout(self.dropout, name = name + "_dropout")(layer)

        if self.n_hidden_layers >= 1:    
            layer = keras.layers.Flatten()(layer)
            self.coupled[histogram] = []
        
            for i in range(self.n_hidden_layers-1):
                dense = keras.layers.Dense(
                        units = self.n_nodes,
                        activation = "relu",
                        use_bias = False,
                        name = "Encoder_Hidden_%d" % i,
                )
                if self.tied_dense:
                    self.coupled[histogram].append(dense)
                layer = dense(layer)

            dense = keras.layers.Dense(
                    units = self.n_components,
                    activation = "relu",
                    use_bias = False,
                    name = "Encoder_Output"
            )
            if self.tied_dense:
                self.coupled[histogram].append(dense)
            encoder = dense(layer)
        
        else:
            encoder = layer

        return input, encoder


    def build_decoder(self, histogram, info, input):
        layer = input
        if not self.tied_dense:
            self.coupled[histogram] = [None]*self.n_hidden_layers
        if self.n_hidden_layers >= 1:
            self.coupled[histogram].reverse()
            for i in range(len(self.coupled[histogram]) - 1):
                layer = DenseTied(
                        units = self.n_nodes,
                        activation = "relu",
                        use_bias = False,
                        name = "Decoder_Hidden_%d" % i,
                        tied_to = self.coupled[histogram][i]
                )(layer)
            units = 1
            for m in self.conv_sizes[histogram][len(self.conv_sizes[histogram]) - 1]:
                units *= m
            layer = DenseTied(
                    units = units,
                    activation = "relu",
                    use_bias = False,
                    name = "Decoder_Hidden_%d" % (len(self.coupled[histogram]) - 1),
                    tied_to = self.coupled[histogram][len(self.coupled[histogram]) - 1]
            )(layer)
        
        self.conv_sizes[histogram].reverse()
        if self.n_conv_layers >= 1:
            name = "decoder_reshape"
            target_shape = self.conv_sizes[histogram][0]
        else:
            target_shape = info["shape"] + (1,)
            print(target_shape)
            name = "output_%s" % (info["name"])
        if self.n_hidden_layers >= 1:
            layer = keras.layers.Reshape(target_shape = target_shape, name = name)(layer)
        for i in range(self.n_conv_layers):
            if i == (self.n_conv_layers - 1):
                activation = "relu"
                n_filters = 1
                name = "output_%s" % (info["name"])
                batch_norm = False
                dropout = 0
            else:
                activation = "relu"
                n_filters = self.n_filters
                name = "decoder_%d_%s" % (i, info["name"])
                batch_norm = self.batch_norm
                dropout = self.dropout
            cur_size = tf.shape(layer)._inferred_value[1:]
            if info["n_dim"] == 1:
                k = self.kernel_1d
                s = self.strides_1d
            elif info["n_dim"] == 2:
                k = self.kernel_2d
                s = self.strides_2d
            l = self.conv_sizes[histogram][i+1]
            out_pads = [l[j] - (cur_size[j] - 1)*s - k for j in range(info["n_dim"])]
            if info["n_dim"] == 1:
                layer = keras.layers.Conv1DTranspose(
                        filters = n_filters,
                        kernel_size = self.kernel_1d,
                        strides = self.strides_1d,
                        use_bias = False,
                        output_padding = out_pads, 
                        activation = activation,
                        name = name
                )(layer)
            elif info["n_dim"] == 2:
                layer = keras.layers.Conv2DTranspose(
                        filters = n_filters,
                        kernel_size = self.kernel_2d,
                        strides = self.strides_2d,
                        output_padding = out_pads,
                        activation = activation,
                        use_bias = False,
                        name = name
                )(layer)
            if batch_norm:
                layer = keras.layers.BatchNormalization(name = name + "_batch_norm")(layer)
            if dropout > 0:
                layer = keras.layers.Dropout(self.dropout, name = name + "_dropout")(layer)
           
        output = layer
        return output

class DenseTied(keras.layers.Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.config_terms = {
                             'units':units,
                             'activation':activation,
                             'use_bias':use_bias,
                             'kernel_initializer':kernel_initializer,
                             'bias_initializer':bias_initializer,
                             'kernel_regularizer':kernel_regularizer,
                             'bias_regularizer':bias_regularizer,
                             'activity_regularizer':activity_regularizer,
                             'kernel_constraint':kernel_constraint,
                             'bias_constraint':bias_constraint,
                             'tied_to':tied_to
        }
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = keras.layers.InputSpec(min_ndim=2)
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update(self.config_terms)
        return config 

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = tf.Variable(tf.transpose(self.tied_to.kernel), trainable = False)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = tf.tensordot(inputs, self.kernel, axes = 1)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
