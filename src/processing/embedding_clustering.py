from sklearn.cluster import SpectralClustering
import numpy as np


def cluster_embeddings(doc2emb, document_num, n_labels):
    embeddings = np.array(list(doc2emb.values()))

    if len(doc2emb) == document_num:
        # all documents are considered in the graph and have embedding
        n_clusters = n_labels
    else:
        # we use the last label for ignored documents
        n_clusters = n_labels - 1
    clustering = SpectralClustering(n_clusters=n_clusters, random_state=0).fit(embeddings)

    doc2label = {}
    for i, doc_id in enumerate(doc2emb.keys()):
        doc2label[doc_id] = clustering.labels_[i]

    if len(doc2emb) != document_num:
        for i in range(document_num):
            if i not in doc2label:
                # i-th document is ignored in the graph
                doc2label[i] = n_labels-1  # the last label is for ignored documents

    return np.array(sorted(doc2label.items()))[:, 1]


def cluster_embeddings_wo_ignored(doc2emb, true_labels, n_labels):
    """clustering without considering the ignored documents in the graph"""
    embeddings = np.array(list(doc2emb.values()))

    clustering = SpectralClustering(n_clusters=n_labels, random_state=0).fit(embeddings)

    clustering_labels_wo_ignored, true_labels_wo_ignored = [], []

    for i, doc_id in enumerate(doc2emb.keys()):
        clustering_labels_wo_ignored.append(clustering.labels_[i])
        true_labels_wo_ignored.append(true_labels[doc_id])

    return clustering_labels_wo_ignored, true_labels_wo_ignored
