from sklearn.cluster import KMeans
from .clustering_layer import ClusteringLayer
from .ae_model import ClusteringAutoEncoderModel
from keras.models import Model
from keras.optimizers import SGD
from src.utils.metrics import accuracy, adjusted_mutual_info
import numpy as np


class DeepClusteringModel:
    def __init__(self, data_size, n_clusters):
        self.ae = ClusteringAutoEncoderModel(data_size)
        self.n_clusters = n_clusters

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(self.ae.encoder.output)
        self.model = Model(inputs=self.ae.encoder.input, outputs=clustering_layer)

    @staticmethod
    def _target_distribution(q):
        """return new target distribution using the given old distribution"""
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def train(self, x, true_labels=None):
        """
        Note: true_labels only are used to evaluate the model, not for training
        :param x:
        :param true_labels:
        :return:
        """
        print('training autoencoder for deep clustering...')
        ae_losses = self.ae.train(x)

        print('constructing deep clustering model...')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(self.ae.encoder.predict(x))  # training k-means on latent space of the input data
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        self.model.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        self._optimize(x, true_labels)

    def _optimize(self, x, true_labels):
        print('optimizing deep clustering model...')
        # hyper parameters
        maxiter = 8000
        update_interval = 140
        log_interval = int(maxiter/20)
        batch_size = 64

        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:  # update the target distribution every <update_interval> steps
                q = self.model.predict(x)
                p = DeepClusteringModel._target_distribution(q)  # update the auxiliary target distribution p

            idx = list(range(index * batch_size, min((index + 1) * batch_size, x.shape[0])))
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])

            if ite % log_interval == 0:
                print('--optimizing deep clustering model on step {}/{}\nloss={}'.format(ite, maxiter, loss))
                if true_labels is not None:
                    # evaluate the clustering performance
                    y_pred = q.argmax(1)
                    acc = accuracy(true_labels, y_pred)
                    ami = adjusted_mutual_info(true_labels, y_pred)
                    print('validation: ACC={}, AMI={}'.format(acc, ami))

            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    def clusters(self, x):
        clusters_probs = self.model.predict(x)
        return np.argmax(clusters_probs, axis=-1)
