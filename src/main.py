from src.utils.datasets import fetch_dataset
from src.modelling.keyword_extraction import extract_top_keywords
import paths


if __name__ == '__main__':
   data = fetch_dataset(paths.the20news_dataset)
   documents = data[:, 1]
   keywords, transformer = extract_top_keywords(documents)
   print(keywords)
