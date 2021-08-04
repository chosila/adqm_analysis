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
    
    "DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/h4DSegmNHits_W-1_St1_Sec1" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector1/Station2/h4DSegmNHits_W-1_St2_Sec1" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector1/Station3/h4DSegmNHits_W-1_St3_Sec1" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector1/Station4/h4DSegmNHits_W-1_St4_Sec1" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector2/Station1/h4DSegmNHits_W-1_St1_Sec2" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector2/Station2/h4DSegmNHits_W-1_St2_Sec2" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector2/Station3/h4DSegmNHits_W-1_St3_Sec2" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector2/Station4/h4DSegmNHits_W-1_St4_Sec2" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector3/Station1/h4DSegmNHits_W-1_St1_Sec3" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector3/Station2/h4DSegmNHits_W-1_St2_Sec3" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector3/Station3/h4DSegmNHits_W-1_St3_Sec3" : { "normalize" : True },
#     "DT/Run summary/02-Segments/Wheel-1/Sector3/Station4/h4DSegmNHits_W-1_St4_Sec3" : { "normalize" : True }
}

#s = StatisticalTester("my_stat_tester")
p = PCA("my_pca")
a = AutoEncoder("my_autoencoder")


for x in [p]:#[s, p, a]:
    x.load_data(
            file = training_file,
            histograms = histograms,
            train_frac = 0.5
    )

    #if isinstance(x, MLAlgorithm) and isinstance(x, PCA):
    #    x.train(n_components=3)
    #elif isinstance(x, MLAlgorithm):
    if args.load_model:
        x.load_model(model_file='models')
    else:
        x.train()
        x.save_model(model_file='models')


test_runs = p.data["run_number"]["test"]#a.data["run_number"]["test"]
test = test_runs[0:10]
ref = test_runs[10]

results = {}
for x in [p]:#[s, p, a]:
    results[x.name] = x.evaluate(
            runs = test,
            reference = ref,
            histograms = list(histograms.keys())#['DT/Run summary/02-Segments/Wheel-1/Sector1/Station1/h4DSegmNHits_W-1_St1_Sec1']
    )


for run in test:
    logger.info("Run: %d" % run)
    for x in [p]:#[s, p, a]:
        logger.info("Algorithm: %s, results: %s" % (x.name, results[x.name][run]))

