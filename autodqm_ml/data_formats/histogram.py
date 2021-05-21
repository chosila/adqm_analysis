import numpy

from autodqm_ml.utils import setup_logger

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
    :param logger: logger to print out various levels of diagnostic info 
    :type logger: logging.getLogger()
    """

    def __init__(self, name, data, reference = None, logger = None):
        self.name = name
        self.data = data
        self.reference = reference
        self.logger = logger

        if self.logger is None:
            self.logger = setup_logger("INFO")

        self.n_dim = self.data["values"].ndim
        self.n_entries = numpy.sum(self.data["values"]) # FIXME: are all DQM histograms occupancies?
        self.is_normalized = False


    def normalize(self):
        """
        Normalize histogram values by the number of entries.
        """
        if self.is_normalized:
            return

        if self.n_entries <= 0:
            message = "[Histogram : normalize] Histogram must have > 0 entries to normalize but histogram %s has %d entries." % (self.name, self.n_entries)
            self.logger.exception(message)
            raise Exception(message)

        self.data["values"] = self.data["values"] / self.n_entries
        self.is_normalized = True
