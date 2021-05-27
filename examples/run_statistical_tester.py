import pandas

from autodqm_ml.data_formats.histogram import Histogram
from autodqm_ml.algorithms.statistical_tester import StatisticalTester
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")

df = pandas.read_pickle("../scripts/output/CSC_EMTF_InitialList_4May2021_SingleMuon_short.pkl")

hist_name = "L1T//Run summary/L1TStage2EMTF/emtfTrackEta"
ref_run = 296168

ref_hist = Histogram(
        name = hist_name.replace("/", "_") + "_" + str(ref_run) + "_" + "ref",
        data = { "values" : df.loc[df["run_number"] == ref_run][hist_name][0] },
        logger = logger
)

histograms = []
for i in range(len(df)):
    run = df["run_number"][i]
    
    histograms.append(Histogram(
            name = hist_name.replace("/", "_") + "_" + str(run),
            data = { "values" : df[hist_name][i] },
            reference = ref_hist
    ))


statistical_tester = StatisticalTester(
        name = "stat_tester"
)
metadata = { "normalize" : True , "min_entries" : 100 }

results = statistical_tester.run(
        histograms = histograms,
        threshold = 0.09,
        metadata = metadata
)

for hist, result in results.items():
    anomalous = "Anomalous" if result["decision"] else "Not anomalous"
    logger.info("Histogram: %s, score: %s, decision: %s" % (hist, result["score"], anomalous))
