from src.modelling.LoNGAE.train_lp_with_feats import run
from .KCG import get_documents_kcg
import paths
import numpy as np
from src.utils.datasets import name_of_dataset
from tensorflow import keras
from src.modelling.LoNGAE.models.ae import autoencoder_with_node_features


class GAE:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.nodes, self._adjacency, self.doc_to_node_mapping, self.documents_labels = get_documents_kcg(self.dataset_path)
        self._nodes_features = np.array([node['feature'] for node in self.nodes])

        try:
            self._encoder, self._ae = self._load_models()
        except OSError:
            self._train()

    def _train(self, validate=False):
        self._encoder, self._ae = run(self._adjacency, self._nodes_features,
                                      saving_directory=paths.models + 'graph_ae/big_{}/'.format(name_of_dataset(self.dataset_path)),
                                      validate=validate)

    def validate(self):
        self._train(validate=True)

    def _load_models(self):
        """

        :raise OSError: when the model file is not found
        :return:
        """
        directory_path = paths.models + 'graph_ae/big_{}/'.format(name_of_dataset(self.dataset_path))

        encoder = keras.models.load_model(directory_path + 'encoder.keras')
        _, ae = autoencoder_with_node_features(self._adjacency.shape[1], self._nodes_features.shape[1])
        ae.load_weights(directory_path + 'autoencoder_weights.h5')

        return encoder, ae

    def latent_feature(self, node_id):
        adj = self._adjacency[node_id]
        feat = self._nodes_features[node_id]
        adj_aug = np.concatenate((adj, feat))
        return self._encoder.predict(adj_aug.reshape(1, -1))[0]  # prediction on one sample
