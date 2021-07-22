from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")

histograms = [
        'L1T//Run summary/L1TStage2EMTF/emtfTrackEta',
        'L1T//Run summary/L1TStage2EMTF/emtfTrackPhi'
]

file = "../scripts/output/CSC_EMTF_InitialList_4May2021_SingleMuon_short.pkl"

autoencoder = AutoEncoder(
        name = "autoencoder"
)

autoencoder.train(
        histograms = histograms,
        file = file,
        config = {},
)
