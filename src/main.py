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

