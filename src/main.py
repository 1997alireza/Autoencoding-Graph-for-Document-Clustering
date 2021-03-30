from src.utils.datasets import fetch_dataset
from src.processing.document_network import create_network
import paths

if __name__ == '__main__':
    data = fetch_dataset(paths.the20news_dataset)
    labels = data[:, 0]
    documents = data[:, 1]
    nodes, adjacency, doc_to_node_mapping = create_network(documents)
