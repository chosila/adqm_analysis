from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.statistical_tester import StatisticalTester
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.algorithms.pca import PCA

from autodqm_ml.utils import setup_logger
logger = setup_logger("DEBUG")

training_file = "scripts/output/test_9Jun2021_SingleMuon.pkl"
histograms = {
        'L1T//Run summary/L1TStage2EMTF/emtfTrackEta' : { "normalize" : True },
        'L1T//Run summary/L1TStage2EMTF/emtfTrackPhi' : { "normalize" : True },
        #'L1T//Run summary/L1TStage2EMTF/emtfTrackQualityVsMode' : { "normalize" : True },
}

s = StatisticalTester("my_stat_tester")
p = PCA("my_pca")
a = AutoEncoder("my_autoencoder")


for x in [s, p, a]:
    x.load_data(
            file = training_file,
            histograms = histograms,
            train_frac = 0.5
    )

    if isinstance(x, MLAlgorithm):
        x.train()


test_runs = a.data["run_number"]["test"]
test = test_runs[0:10]
ref = test_runs[10]

results = {}
for x in [s, p, a]:
    results[x.name] = x.evaluate(
            runs = test,
            reference = ref,
            histograms = ['L1T//Run summary/L1TStage2EMTF/emtfTrackEta']
    )


for run in test:
    logger.info("Run: %d" % run)
    for x in [s, p, a]:
        logger.info("Algorithm: %s, results: %s" % (x.name, results[x.name][run]))

"""
for x in [s, p, a]:
    results = x.evaluate(
            runs = test,
            reference = ref,
            histograms = ['L1T//Run summary/L1TStage2EMTF/emtfTrackEta'],

    )
"""
