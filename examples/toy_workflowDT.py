from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.statistical_tester import StatisticalTester
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.algorithms.pca import PCA
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_model', dest='load_model', type=bool, default=False, help='T/F load model')     
args = parser.parse_args()

from autodqm_ml.utils import setup_logger
logger = setup_logger("INFO")


training_file = 'scripts/output/test_SingleMuon.pkl' #"scripts/output/test_9Jun2021_SingleMuon.pkl"
histograms = {
    "DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/T0_FromSegm_W-1_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/h4DSegmNHits_W-1_St1_Sec1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/T0_FromSegm_W-1_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/VDrift_FromSegm_W-1_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel1/Sector1/Station1/h4DSegmNHits_W1_St1_Sec1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel1/Sector1/Station1/T0_FromSegm_W1_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel1/Sector1/Station1/VDrift_FromSegm_W1_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel0/Sector1/Station1/h4DSegmNHits_W0_St1_Sec1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel0/Sector1/Station1/T0_FromSegm_W0_Sec1_St1" : { "normalize" : True },
    "DT/Run summary/02-Segments/Wheel0/Sector1/Station1/VDrift_FromSegm_W0_Sec1_St1" : { "normalize" : True },
}

p = PCA("my_pca")
a = AutoEncoder("my_autoencoder")


for x in [p]:
    x.load_data(
        file = training_file,
        histograms = histograms,
        train_frac = 0.5,
        remove_identical_bins = True,
        remove_low_stat = True
    )

    if args.load_model:
        x.load_model(model_file='models')
    else:
        x.train()
        x.save_model(model_file='models')
        

test_runs = p.data["run_number"]["test"]
test = test_runs[0:10]
ref = test_runs[10]


results = {}
for x in [p]:
    results[x.name] = x.evaluate(
            runs = test,
            reference = ref,
            histograms = list(histograms.keys())
    )
    
    x.plot(runs = test, 
           histograms = list(histograms.keys())
           )


for run in test:
    logger.info("Run: %d" % run)
    for x in [p]:
        logger.info("Algorithm: %s, results: %s" % (x.name, results[x.name][run]))

