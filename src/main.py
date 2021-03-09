from src.utils.datasets import fetch_dataset
from src.modelling.NMF_keyword_extraction import extract_top_keywords
from collections import defaultdict
from src.utils.text import split_document
import paths

if __name__ == '__main__':
    data = fetch_dataset(paths.the20news_dataset)
    documents = data[:, 1]
    for doc in documents:
        sentences = split_document(doc)
        nodes = extract_top_keywords(sentences)
        print(nodes)
        exit()
