import sys
from io import StringIO
from dvc import api
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(name)s : %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger(__name__)

logging.info("Fetching data...")

finantial_dataset = api.read(
    path="dataset/finantials.csv", remote="dataset-tracker", encoding="utf8"
)
movies_dataset = api.read(
    path="dataset/movies.csv", remote="dataset-tracker", encoding="utf8"
)
opening_gorss_dataset = api.read(
    path="dataset/opening_gross.csv", remote="dataset-tracker", encoding="utf8"
)

finantial_data = pd.read_csv(StringIO(finantial_dataset))
movies_data = pd.read_csv(StringIO(movies_dataset))
opening_gorss_data = pd.read_csv(StringIO(opening_gorss_dataset))

numeric_column_mask = (movies_data.dtypes == float) | (movies_data.dtypes == int)
numeric_column = [
    column for column in numeric_column_mask.index if numeric_column_mask[column]
]

movies_data = movies_data[numeric_column + ["movie_title"]]
finantial_data = finantial_data[["movie_title", "production_budget", "worldwide_gross"]]
finantial_and_movies_data = pd.merge(
    finantial_data, movies_data, on="movie_title", how="left"
)
full_movie_data = pd.merge(
    opening_gorss_data, finantial_and_movies_data, on="movie_title", how="left"
)

full_movie_data = full_movie_data.drop(["gross", "movie_title"], axis=1)

full_movie_data.to_csv("dataset/full_data.csv", index=False)

logger.info("data already fetch and prepared!")
