from sklearn.cluster import KMeans
from .clustering_layer import ClusteringLayer
from keras.models import Model
from keras.optimizers import SGD
from src.utils.metrics import accuracy, adjusted_mutual_info
import numpy as np


class ClusteringModel:
    def __init__(self, encoder, n_clusters):
        self.encoder = encoder
        self.n_clusters = n_clusters

        clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
        self.model = Model(inputs=encoder.input, outputs=clustering_layer)
        self.input_data = None
        self.true_labels = None

    def initialize(self, x, true_labels):  # TODO: x is not embedding, it's the input of encoder
        self.input_data = np.array(x)  # we need to keep the input data for further usage in the class
        self.true_labels = true_labels
        # Initialize cluster centers using k-means.
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        kmeans.fit(self.encoder.predict(x))
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        self.model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

    @staticmethod
    def _target_distribution(q):
        """return new target distribution using the given old distribution"""
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def optimize(self):
        # hyper parameters
        maxiter = 8000
        update_interval = 140
        batch_size = 20

        index = 0
        x = self.input_data
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:  # update the target distribution every <update_interval> steps
                q = self.model.predict(x)
                p = ClusteringModel._target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                acc = accuracy(self.true_labels, y_pred)
                ami = adjusted_mutual_info(self.true_labels, y_pred)
                print('--optimizing deep clustering model; ACC={}, AMI={}'.format(acc, ami))

            idx = list(range(index * batch_size, min((index + 1) * batch_size, x.shape[0])))
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            if ite % update_interval == 0:
                print('--optimizing deep clustering model; loss={}'.format(loss))
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    def clusters(self):
        return self.model.predict(self.input_data)
