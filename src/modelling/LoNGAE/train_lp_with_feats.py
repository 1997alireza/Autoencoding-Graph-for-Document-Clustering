"""
This script trains a model based on the symmetrical autoencoder
architecture with parameter sharing. The model performs link
prediction using latent features learned from local graph topology
and available node features. The following datasets have node features:
{protein, metabolic, conflict, cora, citeseer, pubmed}

Usage: python train_lp_with_feats.py <dataset_str> <gpu_id>
"""

import numpy as np
from keras import backend as K
import random
from src.utils.mathematical import MSE
from .utils import generate_data, batch_data
from .utils_gcn import split_adjacency_data
from .models.ae import autoencoder_with_node_features
import paths

import os
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# TODO: validate using evaluate=True
def run(adj, feats, evaluate=False):
    """

    :param adj: adjacency matrix as a 2d numpy array
    :param feats: node features as a 2d numpy array
    :param evaluate: if it's True, the evaluation on link prediction task is done.
    (if the evaluation is being done, the model would not be trained on all of the provided data)
    :return:
    """
    adj = (adj + 1.0) * .5  # mapping edge weights from [-1, 1] to [0, 1]

    if evaluate:
        print('\nPreparing test split...\n')
        # splitting 10% of data for validation
        data_number = adj.shape[0]
        # TODO: check data_number is 50? if it is remove validation_batch_size
        val_inds = random.sample(range(0, data_number), int(data_number/10))
        val_mask = np.zeros((data_number,), dtype=bool)
        val_mask[val_inds] = True
        adj_val = adj[val_mask]
        feats_val = feats[val_mask]
        adj = adj[~val_mask]
        feats = feats[~val_mask]

    print('\nCompiling autoencoder model...\n')

    encoder, ae = autoencoder_with_node_features(adj.shape[1], feats.shape[1])

    print(ae.summary())

    # Specify some hyperparameters
    epochs = 50  # TODO: change it for the final test, and remove link prediction evaluation part
    train_batch_size = 20
    val_batch_size = 256

    print('\nFitting autoencoder model...\n')

    aug_adj = np.hstack((adj, feats))
    training_data = generate_data(aug_adj, adj, feats, shuffle=True)
    b_data = batch_data(training_data, train_batch_size)
    num_iters_per_train_epoch = aug_adj.shape[0] / train_batch_size
    for e in range(epochs):
        print('\nEpoch {:d}/{:d}'.format(e + 1, epochs))
        print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
        curr_iter = 0
        train_loss = []
        for batch_aug_adj, batch_adj, batch_f in b_data:
            # Each iteration/loop is a batch of train_batch_size samples
            # TODO: check if [np.hstack((batch_adj, batch_f))] == [batch_aug_adj] => remove batch_aug_adj
            loss = ae.train_on_batch([batch_aug_adj], [batch_adj, batch_f])
            total_loss = loss[0]
            # when we have multiple losses, train_on_batch returns a list [total_loss, loss1, loss2, ...]
            train_loss.append(total_loss)
            curr_iter += 1
            if curr_iter >= num_iters_per_train_epoch:
                break

        train_loss = np.asarray(train_loss)
        train_loss = np.mean(train_loss, axis=0)
        print('Avg. training loss: {:s}'.format(str(train_loss)))

        if not evaluate:
            encoder.save(paths.models + 'graph_ae/encoder.keras')
            ae.save_weights(paths.models + 'graph_ae/autoencoder_weights.h5')
            # we couldn't save the autoencoder model itself because of the DenseTied layer
            print('\nTrained model is saved.')

        # if evaluate:  #TODO, maybe we can use test_on_batch function, also use val_batch_size for memory efficiency
        #     print('\nEvaluating val set on link prediction...')
        #     outputs, predictions = [], []
        #     for step in range(int(aug_adj.shape[0] / val_batch_size + 1)):
        #         low = step * val_batch_size
        #         high = low + val_batch_size
        #         batch_aug_adj = aug_adj[low:high]
        #         if batch_aug_adj.shape[0] == 0:
        #             break
        #         decoded_lp = ae.predict_on_batch([batch_aug_adj])[0]
        #         outputs.append(decoded_lp)
        #     decoded_lp = np.vstack(outputs)
        #     predictions.extend(decoded_lp[test_r, test_c])
        #     predictions.extend(decoded_lp[test_c, test_r])
        #     predictions = np.array(predictions)
        #     print('Link prediction val MSE: {:6f}'.format(MSE(labels[non_zero_labels_idx], predictions[non_zero_labels_idx])))

    print('\nAll done.')
