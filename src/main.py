from src.utils.datasets import fetch_dataset
from src.modelling.SBERT_transformer import get_sentence_transformer
from src.processing.document_network import extract_network
import paths

if __name__ == '__main__':
    data = fetch_dataset(paths.the20news_dataset)
    documents = data[:, 1]
    sentence_transformer = get_sentence_transformer()
    for doc in documents:
        nodes, edges = extract_network(doc, sentence_transformer)
        exit()
