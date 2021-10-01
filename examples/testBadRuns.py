from autodqm_ml.algorithms.ml_algorithm import MLAlgorithm
from autodqm_ml.algorithms.autoencoder import AutoEncoder
from autodqm_ml.algorithms.pca import PCA

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--test', dest='test', type=bool, default=False, help='Bool: whether running script as train or test')     
args = parser.parse_args()

from autodqm_ml.utils import setup_logger
logger = setup_logger("INFO")


wheels = [-2]#,0,1]
secs = [1]#,5,10]
sts = [1]#,2,3,4]

t0  = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/T0_FromSegm_W{w}_Sec{sec}_St{st}' for w,sec,st in zip(wheels,secs,sts)]
h4d = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/h4DSegmNHits_W{w}_St{st}_Sec{sec}' for w,sec,st in zip(wheels,secs,sts)]
vdrift = [f'DT/Run summary/02-Segments/Wheel{w}/Sector{sec}/Station{st}/VDrift_FromSegm_W{w}_Sec{sec}_St{st}' for w,sec,st in zip(wheels,secs,sts)]
histnames = t0 + h4d + vdrift 

histograms = {histname:{'normalize':True} for histname in histnames}
training_file = 'scripts/output/test_SingleMuon.pkl'
testing_file = 'scripts/output/bad_dt_SingleMuon.pkl'

p = PCA('my_pca')

if not args.test:
    p.load_data(
        file = training_file,
        histograms = histograms,
        train_frac = 0.8, 
        remove_identical_bins = True,
        remove_low_stat = True
    )
    p.train()
    p.save_model(model_file='models')

else: 
    p.load_data(
        file = testing_file,
        histograms = histograms, 
        train_frac = 1,
        remove_identical_bins = True,
        remove_low_stat = False
    )
    p.load_model(model_file='models')

test_runs = p.data["run_number"]["test"]

test = test_runs[0:10]

results = p.evaluate(
    runs = test_runs,
    histograms = list(histograms.keys())
)
p.plot(
    runs = test_runs,
    histograms = list(histograms.keys())
) 

for run in test:
    logger.info("Run: %d" % run)
    logger.info("Algorithm: %s, results: %s" % (p.name, results[run]))
