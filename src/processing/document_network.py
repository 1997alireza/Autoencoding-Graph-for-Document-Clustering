from src.modelling.NMF_keyword_extraction import extract_top_keywords
from src.utils.text import split_document
from src.processing.edge_weighting import sentence_similarity_edge
from src.modelling.SBERT_transformer import get_sentence_transformer
import numpy as np


def create_network(documents):
    """

    :param documents: list of documents, each document is taken as a string

    :return: nodes (a list of nodes including{'keyword', 'feature'});
             node['feature'] is the average of its sentences' SBERT embeddings,
             and adjacency (a 2d numpy array containing weights of the graph's adjacency);
             each weight is a float in the range [-1, 1]) computed based on sentence set similarity,
             and doc_to_node_mapping which is a mapping from documnet index to its related nodes
    """
    sentence_transformer = get_sentence_transformer()
    documents_sentences = [split_document(doc) for doc in documents]

    embeddings = [sentence_transformer.encode(sentences) for sentences in documents_sentences]
    # a 2d list of embeddings of each sentence in each document

    keyword_sents = extract_top_keywords(documents_sentences)
    nodes = []  # a list of {'keyword', 'feature'}
    nodes_sentences_idx_tuple = []  # [(i,j)] a list of indexes of related sentences to each node
    doc_to_node_mapping = [[] for _ in range(len(documents_sentences))]
    # doc_to_node_mapping[i] is a list containing the indexes of the nodes related to the i-th document

    for keyword in keyword_sents:
        sentences_idx_tuple = keyword_sents[keyword]
        average_embeddings = \
            np.sum([embeddings[doc_idx][sent_idx] for doc_idx, sent_idx in sentences_idx_tuple], axis=0)\
            / len(sentences_idx_tuple)
        nodes.append({'keyword': keyword, 'feature': average_embeddings})
        nodes_sentences_idx_tuple.append(sentences_idx_tuple)

        node_idx = len(nodes) - 1
        for sentence_doc_id, _ in sentences_idx_tuple:
            doc_to_node_mapping[sentence_doc_id].append(node_idx)

    adjacency = np.zeros([len(nodes), len(nodes)], dtype=float)

    for i, node_i_sentences_idx_tuple in enumerate(nodes_sentences_idx_tuple):
        for j, node_j_sentences_idx_tuple in enumerate(nodes_sentences_idx_tuple):
            edge_weight = sentence_similarity_edge(node_i_sentences_idx_tuple, node_j_sentences_idx_tuple, embeddings)
            adjacency[i, j] = edge_weight

    return nodes, adjacency, doc_to_node_mapping
