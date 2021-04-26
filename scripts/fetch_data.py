from autodqm_ml.data_prep.data_fetcher import DataFetcher
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")
fetcher = DataFetcher(
        contents = "../autodqm_ml/data_prep/metadata/contents_example.json",
        years = "2017,2018",
        datasets = "SingleMuon",
        logger = logger
)

fetcher.run()

