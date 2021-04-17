from sklearn.cluster import SpectralClustering, KMeans
from src.modelling.deep_clustering.clustering_model import DeepClusteringModel
import numpy as np


def cluster_embeddings(doc2emb, document_num, n_labels, method):
    """

    :param doc2emb:
    :param document_num:
    :param n_labels:
    :param method: determine the clustering method which can be 'spectral' or 'kmeans' or 'deep'
    :return:
    """
    assert method in ['spectral', 'kmeans', 'deep']

    embeddings = np.array(list(doc2emb.values()))

    if len(doc2emb) == document_num:
        # all documents are considered in the graph and have embedding
        n_clusters = n_labels
    else:
        # we use the last label for ignored documents
        n_clusters = n_labels - 1
        
    labels = do_clustering(embeddings, n_clusters, method)

    doc2label = {}
    for i, doc_id in enumerate(doc2emb.keys()):
        doc2label[doc_id] = labels[i]

    if len(doc2emb) != document_num:
        for i in range(document_num):
            if i not in doc2label:
                # i-th document is ignored in the graph
                doc2label[i] = n_labels-1  # the last label is for ignored documents

    return np.array(sorted(doc2label.items()))[:, 1]


def cluster_embeddings_wo_ignored(doc2emb, true_labels, n_labels, method):
    """
    clustering without considering the ignored documents in the graph
    :param doc2emb: 
    :param true_labels: 
    :param n_labels:
    :param method: determine the clustering method which can be 'spectral' or 'kmeans' or 'deep'
    :return: 
    """
    assert method in ['spectral', 'kmeans', 'deep']
    
    embeddings = np.array(list(doc2emb.values()))

    labels = do_clustering(embeddings, n_labels, method)

    clustering_labels_wo_ignored, true_labels_wo_ignored = [], []

    for i, doc_id in enumerate(doc2emb.keys()):
        clustering_labels_wo_ignored.append(labels[i])
        true_labels_wo_ignored.append(true_labels[doc_id])

    return clustering_labels_wo_ignored, true_labels_wo_ignored


def do_clustering(x, n_clusters, method):
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(x)
        return clustering.labels_
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
        return clustering.labels_
    else:  # method == 'deep'
        deep_clustering = DeepClusteringModel(data_size=x.shape[1], n_clusters=n_clusters)
        deep_clustering.train(x)
        return deep_clustering.clusters(x)
