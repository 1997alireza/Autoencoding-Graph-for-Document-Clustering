from src.modelling.NMF_keyword_extraction import extract_top_keywords
from src.utils.text import split_document
from src.processing.edge_weighting import sentence_similarity_edge
import numpy as np


def extract_network(doc, sentence_transformer):
    """

    :param doc: string of the document
    :param sentence_transformer: SBERT transformer
    :return: nodes (a list of nodes including{'keyword', 'feature'});
             node['feature'] is the average of its sentences' SBERT embeddings,
             and edges (a 2d numpy array containing weights of the graph's edges);
             each weight is a float in the range [-1, 1])
    """
    sentences = split_document(doc)
    embeddings = sentence_transformer.encode(sentences)
    keyword_sents = extract_top_keywords(sentences)
    nodes = []  # a list of {'keyword', 'feature'}
    nodes_sentences_idx = []  # a list of indexes of related sentences to each node
    for keyword in keyword_sents:
        sentences_idx = keyword_sents[keyword]
        average_embeddings = np.sum([embeddings[idx] for idx in sentences_idx], axis=0) / len(sentences_idx)
        nodes.append({'keyword': keyword, 'feature': average_embeddings})
        nodes_sentences_idx.append(sentences_idx)

    edges = np.zeros([len(nodes), len(nodes)], dtype=float)

    for i, node_i_sentences_idx in enumerate(nodes_sentences_idx):
        for j, node_j_sentences_idx in enumerate(nodes_sentences_idx):
            edge_weight = sentence_similarity_edge(node_i_sentences_idx, node_j_sentences_idx, embeddings)
            edges[i, j] = edge_weight

    return nodes, edges
