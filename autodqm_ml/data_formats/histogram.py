import numpy

import logging
logger = logging.getLogger(__name__)

class Histogram():
    """
    Class to store histogram data, relevant metadata, and
    perform any necessary normalization or preprocessing.
    :param name: name to identify this histogram
    :type name: str
    :param data: histogram data
    :type data: dict
    :param reference: optional, reference histogram to be compared against
    :type reference: Histogram
    """

    def __init__(self, name, data, reference = None):
        self.name = name
        self.data = data
        self.reference = reference

        self.shape = self.data.shape
        self.n_dim = len(self.data.shape)
        self.n_bins = 1.
        for x in self.shape:
            self.n_bins *= x
        if not (self.n_dim == 1 or self.n_dim == 2):
            message = "[Histogram : __init__] Only 1d and 2d histograms are supported. For histogram '%s', we found shape %s, from which we infer %d dimensions." % (self.name, str(self.shape), self.n_dim)
            logger.exception(message)
            raise ValueError(message)

        self.n_entries = numpy.sum(self.data) 
        self.is_normalized = False
    

    def normalize(self):
        """
        Normalize histogram values by the number of entries.
        """
        if self.is_normalized:
            return

        if self.n_entries <= 0:
            message = "[Histogram : normalize] Histogram must have > 0 entries to normalize but histogram %s has %d entries." % (self.name, self.n_entries)
            logger.exception(message)
            raise Exception(message)

        self.data = self.data / float(self.n_entries)
        self.is_normalized = True
