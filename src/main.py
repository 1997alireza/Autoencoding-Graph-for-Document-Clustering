from src.processing.GAE_on_KCG import GAE
from src.processing.document_embedding import extract_embeddings
from src.processing.embedding_clustering import cluster_embeddings, cluster_embeddings_wo_ignored
from src.utils.datasets import name_of_dataset, number_of_labels
import paths
from src.utils.metrics import accuracy, adjusted_mutual_info


if __name__ == '__main__':
    dataset_path = paths.reuters_dataset
    gae = GAE(dataset_path)
    doc2emb = extract_embeddings(gae)

    clustering_method = 'spectral'  # in ['spectral', 'kmeans', 'deep']
    print("\nCLUSTERING METHOD <{}>".format(clustering_method))
    clustering_labels = cluster_embeddings(doc2emb,
                                           document_num=len(gae.documents_labels),
                                           n_labels=number_of_labels(name_of_dataset(dataset_path)),
                                           method=clustering_method)
    acc = accuracy(gae.documents_labels, clustering_labels)
    ami = adjusted_mutual_info(gae.documents_labels, clustering_labels)
    print('ACC={}, AMI={}'.format(acc, ami))

    clustering_labels_wo_ignored, true_labels_wo_ignored = \
        cluster_embeddings_wo_ignored(doc2emb,
                                      gae.documents_labels,
                                      n_labels=number_of_labels(name_of_dataset(dataset_path)),
                                      method=clustering_method)
    acc = accuracy(true_labels_wo_ignored, clustering_labels_wo_ignored)
    ami = adjusted_mutual_info(true_labels_wo_ignored, clustering_labels_wo_ignored)

    print('\nWHEN WE DO NOT CONSIDER IGNORED DOCUMENTS')
    print('ACC={}, AMI={}'.format(acc, ami))

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
    # ACC=0.39625, AMI=0.00969
    # WO IGNORED DOCUMENTS: ACC=0.3757, AMI=0.1086

    # the20news (with 5458/18692 (29%) ignored documents in the graph):
    # ACC=0.09796, AMI=0.03916
    # WO IGNORED DOCUMENTS: ACC=0.11146, AMI=0.0587
