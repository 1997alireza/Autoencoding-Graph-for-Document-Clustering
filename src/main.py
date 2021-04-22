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

    # ---On small graph (50 keywords)---

    # <Spectral Clustering>
    # reuters (with 3391/7884 (43%) ignored documents in the graph):
    # ACC=0.35223, AMI=0.12527
    # WO IGNORED DOCUMENTS: ACC=0.30091, AMI=0.18111

    # the20news (with 5458/18692 (29%) ignored documents in the graph):
    # ACC=0.10849, AMI=0.05515
    # WO IGNORED DOCUMENTS: ACC=0.13079, AMI=0.07574

    # <K-Means>
    # reuters (with 3391/7884 (43%) ignored documents in the graph):
    # ACC=0.33916, AMI=0.13302
    # WO IGNORED DOCUMENTS: ACC=0.25662, AMI=0.18046

    # the20news (with 5458/18692 (29%) ignored documents in the graph):
    # ACC=0.11079, AMI=0.06023
    # WO IGNORED DOCUMENTS: ACC=0.12619, AMI=0.07791

    # <Deep Clustering>
    # reuters (with 3391/7884 (43%) ignored documents in the graph):
    # ACC=0.38661, AMI=0.04415
    # WO IGNORED DOCUMENTS: ACC=0.34787, AMI=0.10119

    # the20news (with 5458/18692 (29%) ignored documents in the graph):
    # ACC=0.10004, AMI=0.04059
    # WO IGNORED DOCUMENTS: ACC=0.08244, AMI=0.04647

    # ---On Big graph (70 keywords)---

    # <Spectral Clustering> ?
    # the20news (with 3580/18692 (29%) ignored documents in the graph):
    # ACC=0.11251, AMI=0.07183
    # WO IGNORED DOCUMENTS: ACC=0.1244, AMI=0.08739
