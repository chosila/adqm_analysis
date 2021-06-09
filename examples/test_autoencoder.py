import pandas
file = "../scripts/output/CSC_EMTF_InitialList_4May2021_SingleMuon_short.pkl"
df = pandas.read_pickle(file)

from autodqm_ml.utils import setup_logger
logger = setup_logger("DEBUG")

from autodqm_ml.algorithms.autoencoder import AutoEncoder

names = [
               #'CSC//Run summary/CSCOfflineMonitor/Occupancy/hORecHits',
               #'CSC//Run summary/CSCOfflineMonitor/Occupancy/hOSegments',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeCombinedSerial',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeVsTOF',
               #'CSC//Run summary/CSCOfflineMonitor/Segments/hSTimeVsZ',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackBX',
               'L1T//Run summary/L1TStage2EMTF/emtfTrackEta',
               #'L1T//Run summary/L1TStage2EMTF/emtfTrackOccupancy',
               'L1T//Run summary/L1TStage2EMTF/emtfTrackPhi',
               'L1T//Run summary/L1TStage2EMTF/emtfTrackQualityVsMode',
]

a = AutoEncoder(name = "test")
a.train(histograms = names, file = file, config = {}, n_epochs = 1000, batch_size = 100)


