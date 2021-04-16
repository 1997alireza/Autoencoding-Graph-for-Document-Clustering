"""
Calculating the embedding of documents using their the GAE's latent feature of their related keywords in KCG
We use Global Average Pooling over the related node sequence
"""

import numpy as np
from collections import defaultdict


def extract_embeddings(gae):
    """
    Note: some documents may have not any related nodes in the graph, so they are ignored and do not have embedding
    :param gae: The trained graph autoencoder on KCG
    :return: dictionary mapping from document id to its embedding
    """
    doc2emb = defaultdict(np.ndarray)  # doc2emb[doc id] = embedding of document
    ignored_documents = []  # documents whose sentences are not assigned to the graph, and do not have embedding
    nodes_latent_feature = [gae.latent_feature(node_id) for node_id in range(len(gae.nodes))]

    for doc_id in range(len(gae.doc_to_node_mapping)):
        related_nodes = gae.doc_to_node_mapping[doc_id]
        if len(related_nodes) == 0:
            ignored_documents.append(doc_id)
        else:
            latent_features = [nodes_latent_feature[node_id] for node_id in related_nodes]
            doc2emb[doc_id] = np.mean(latent_features, axis=0)  # Global Average Pooling over the node sequence

    # all ignored documents will be classified into one cluster
    print('documents\' embeddings are extracted. ignored documents: {}/{}'
          .format(len(ignored_documents), len(gae.doc_to_node_mapping)))
    # 43% for reuters dataset

    return doc2emb


if __name__ == '__main__':
    import paths
    extract_embeddings(paths.reuters_dataset)
