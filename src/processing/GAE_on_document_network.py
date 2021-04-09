from src.modelling.LoNGAE.train_lp_with_feats import run
from .document_network import get_documents_network
import paths
from datetime import datetime
import numpy as np


def train():
    time_zero = datetime.now()
    nodes, adjacency, doc_to_node_mapping, documents_labels = get_documents_network(paths.the20news_dataset)
    nodes_features = np.array([node['feature'] for node in nodes])
    print('delta T: ', datetime.now() - time_zero)
    time_zero = datetime.now()
    run(adjacency, nodes_features)
    print('delta T: ', datetime.now() - time_zero)
