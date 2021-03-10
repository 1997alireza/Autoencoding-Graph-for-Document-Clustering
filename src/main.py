from src.utils.datasets import fetch_dataset
from src.modelling.NMF_keyword_extraction import extract_top_keywords
from src.utils.text import split_document
from src.modelling.SBERT_transformer import get_sentence_transformer
from src.processing.edge_weighting import sentence_similarity_edge
import numpy as np
import paths

if __name__ == '__main__':
    data = fetch_dataset(paths.the20news_dataset)
    documents = data[:, 1]
    sentence_transformer = get_sentence_transformer()
    for doc in documents:
        sentences = split_document(doc)
        embeddings = sentence_transformer.encode(sentences)
        keyword_sents = extract_top_keywords(sentences)
        nodes = []  # a list of {'keyword', 'feature', 'sentences_idx'}
        for keyword in keyword_sents:
            sentences_idx = keyword_sents[keyword]
            average_embeddings = np.sum([embeddings[idx] for idx in sentences_idx], axis=0) / len(sentences_idx)
            nodes.append({'keyword': keyword, 'feature': average_embeddings, 'sentences_idx': sentences_idx})

        edges = []  # a list of (node_number_i, node_number_j, weight)

        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    edge_weight = sentence_similarity_edge(node_i['sentences_idx'], node_j['sentences_idx'],
                                                                 embeddings)
                    edges.append((i, j, edge_weight))
        print(nodes)
        exit()
