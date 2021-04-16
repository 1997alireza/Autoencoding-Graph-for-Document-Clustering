from src.processing.GAE_on_KCG import GAE
from src.processing.document_embedding import extract_embeddings
from src.processing.embedding_clustering import cluster_embeddings, cluster_embeddings_wo_ignored
from src.utils.datasets import name_of_dataset, number_of_labels
import paths
from src.utils.metrics import accuracy
from sklearn.metrics.cluster import adjusted_mutual_info_score


if __name__ == '__main__':
    dataset_path = paths.reuters_dataset
    gae = GAE(dataset_path)
    doc2emb = extract_embeddings(gae)
    clustering_labels = cluster_embeddings(doc2emb,
                                           document_num=len(gae.documents_labels),
                                           n_labels=number_of_labels(name_of_dataset(dataset_path)))
    acc = accuracy(gae.documents_labels, clustering_labels)
    ami = adjusted_mutual_info_score(gae.documents_labels, clustering_labels)
    print('ACC={}, AMI={}'.format(acc, ami))

    clustering_labels_wo_ignored, true_labels_wo_ignored = cluster_embeddings_wo_ignored(
        doc2emb, gae.documents_labels, n_labels=number_of_labels(name_of_dataset(dataset_path)))
    acc = accuracy(true_labels_wo_ignored, clustering_labels_wo_ignored)
    ami = adjusted_mutual_info_score(true_labels_wo_ignored, clustering_labels_wo_ignored)

    print('\nWHEN WE DO NOT CONSIDER IGNORED DOCUMENTS')
    print('ACC={}, AMI={}'.format(acc, ami))

    # reuters (with 3391/7884 (43%) ignored documents in the graph):
    # ACC=0.35223236935565705, AMI=0.12527552602756395
    # WO IGNORED DOCUMENTS: ACC=0.3009125306031605, AMI=0.18111226509277029

    # the20news (with 5458/18692 (29%) ignored documents in the graph):
    # ACC=0.1084956130965119, AMI=0.055153756779170594
    # WO IGNORED DOCUMENTS: ACC=0.13079945594680364, AMI=0.07574611751816518
