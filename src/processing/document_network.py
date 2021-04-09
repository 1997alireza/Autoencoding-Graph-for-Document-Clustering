from src.modelling.NMF_keyword_extraction import extract_top_keywords
from src.utils.text import split_document
from src.processing.edge_weighting import sentence_similarity_edge
from src.modelling.SBERT_transformer import get_sentence_transformer
from src.utils.datasets import fetch_dataset
import paths
import numpy as np
import pickle
from src.utils.text import preprocess


def create_network(documents, dataset_name=None):
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

    doc_to_node_mapping = [[] for _ in range(len(documents_sentences))]
    # doc_to_node_mapping[i] is a list containing the indexes of the nodes related to the i-th document

    keyword_sents = extract_top_keywords(documents_sentences, dataset_name=dataset_name)
    nodes = []  # a list of {'keyword', 'feature'}
    adjacency = np.zeros([len(keyword_sents), len(keyword_sents)], dtype=float)

    for node_idx, keyword in enumerate(keyword_sents):
        print('node {}/{}'.format(node_idx, len(keyword_sents)))
        sentences_idx_tuple = keyword_sents[keyword]
        embeddings_list = sentence_transformer.encode(
            [documents_sentences[doc_idx][sent_idx] for doc_idx, sent_idx in sentences_idx_tuple])
        average_embeddings = np.sum(embeddings_list, axis=0) / len(sentences_idx_tuple)

        # node feature
        nodes.append({'keyword': keyword, 'feature': average_embeddings})

        # node adjacency vector
        for other_node_idx, other_keyword in enumerate(keyword_sents):
            print('--inner loop node {}/{}'.format(other_node_idx, len(keyword_sents)))
            other_sentences_idx_tuple = keyword_sents[other_keyword]
            other_embeddings_list = sentence_transformer.encode(
                [documents_sentences[doc_idx][sent_idx] for doc_idx, sent_idx in other_sentences_idx_tuple])
            edge_weight = sentence_similarity_edge(embeddings_list, other_embeddings_list)
            adjacency[node_idx, other_node_idx] = edge_weight

        for doc_idx, _ in sentences_idx_tuple:
            doc_to_node_mapping[doc_idx].append(node_idx)

    print('nodes\' features and adjacency vector are computed')

    print('documents network is created')
    return nodes, adjacency, doc_to_node_mapping


def get_documents_network(dataset_path=paths.the20news_dataset):
    """

    :param dataset_path: paths.reuters_dataset or paths.the20news_dataset
    :return: nodes, adjacency, doc_to_node_mapping; same as create_network function,
             and documents_labels; a list of each document's label
    """

    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    graph_file_path = paths.models + 'keyword_correlation_graph/' + dataset_name + '.pkl'
    data = fetch_dataset(dataset_path)
    try:
        nodes, adjacency, doc_to_node_mapping = pickle.load(open(graph_file_path, 'rb'))
        documents_labels = data[:, 0]
        print('documents network are loaded')
    except FileNotFoundError:
        documents_labels = data[:, 0]
        documents = data[:, 1]
        documents = [preprocess(doc) for doc in documents]
        nodes, adjacency, doc_to_node_mapping = create_network(documents, dataset_name)
        pickle.dump((nodes, adjacency, doc_to_node_mapping), open(graph_file_path, 'wb'))

    return nodes, adjacency, doc_to_node_mapping, documents_labels
