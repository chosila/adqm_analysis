from autodqm_ml.data_prep.data_fetcher import DataFetcher
from autodqm_ml.utils import setup_logger

logger = setup_logger("DEBUG", "output/log.txt")
fetcher = DataFetcher(
        tag = "test", # will identify output files
        contents = "metadata/contents_example.json",
        datasets = "metadata/datasets_example.json",
        short = True
)

fetcher.run()

