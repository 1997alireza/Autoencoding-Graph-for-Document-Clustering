from src.modelling.LoNGAE.train_lp_with_feats import run
from .document_network import create_all_document_networks
import paths
from datetime import datetime


def train():
    time_zero = datetime.now()
    networks, labels = create_all_document_networks(paths.the20news_dataset)
    print('delta T: ', datetime.now() - time_zero)
    time_zero = datetime.now()

    run(actors_adjacency, actors_feature, node_features_weight, evaluate_lp=True)
    print('delta T: ', datetime.now() - time_zero)