import os
import sys
import tensorflow as tf

import paths
from src.processing.GAE_to_KCG import GAE
from src.processing.document_embedding import extract_embeddings
from src.processing.embedding_clustering import cluster_embeddings, cluster_embeddings_wo_ignored
from src.utils.datasets import name_of_dataset, number_of_labels
from src.utils.metrics import accuracy, adjusted_mutual_info


class HiddenPrints:
    def __init__(self, verbose):
        self._verbose = verbose

    def __enter__(self):
        if not self._verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            tf.get_logger().setLevel('WARNING')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def run(dataset_path, clustering_method, big_graph=False, verbose=False):
    """

    :param dataset_path: paths.reuters_dataset or paths.the20news_dataset
    :param clustering_method: 'spectral' or 'kmeans' or 'deep'
    :param big_graph: if True leads to a bg graph with at most 70 nodes, otherwise with at most 50 nodes
    :param verbose: if is True ignore prints other than the evaluation results

    evaluating the document clustering approach using AMI and Accuracy metrics
    """
    with HiddenPrints(verbose):
        gae = GAE(dataset_path, big_graph)
        doc2emb = extract_embeddings(gae)
        clustering_labels = cluster_embeddings(doc2emb,
                                               document_num=len(gae.documents_labels),
                                               n_labels=number_of_labels(name_of_dataset(dataset_path)),
                                               method=clustering_method)
        acc = accuracy(gae.documents_labels, clustering_labels)
        ami = adjusted_mutual_info(gae.documents_labels, clustering_labels)

    print('ACC={}, AMI={}'.format(acc, ami))

    with HiddenPrints(verbose):
        clustering_labels_wo_ignored, true_labels_wo_ignored = \
            cluster_embeddings_wo_ignored(doc2emb,
                                          gae.documents_labels,
                                          n_labels=number_of_labels(name_of_dataset(dataset_path)),
                                          method=clustering_method)
        acc = accuracy(true_labels_wo_ignored, clustering_labels_wo_ignored)
        ami = adjusted_mutual_info(true_labels_wo_ignored, clustering_labels_wo_ignored)

    print('\nWHEN WE DO NOT CONSIDER IGNORED DOCUMENTS')
    print('ACC={}, AMI={}'.format(acc, ami))


if __name__ == '__main__':
    dataset_path = paths.reuters_dataset
    clustering_method = 'spectral'
    print("\nCLUSTERING METHOD <{}> on dataset <{}>".format(clustering_method, name_of_dataset(dataset_path)))
    run(dataset_path, clustering_method, big_graph=True, verbose=True)

    # ignored documents:
    # retuers - small graph: 3391/7884 (43%)
    # retuers - big graph: 1793/7884 (22%)
    # the20news - small graph: 5458/18692 (29%)
    # the20news - big graph: 3580/18692 (19%)
