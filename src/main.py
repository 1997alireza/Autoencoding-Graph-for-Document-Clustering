from src.processing.document_network import get_documents_network
import paths

if __name__ == '__main__':
    nodes, adjacency, doc_to_node_mapping, documents_labels = get_documents_network(paths.reuters_dataset)
    # TODO: some documents may have not any related nodes in the graph
    print(nodes)
