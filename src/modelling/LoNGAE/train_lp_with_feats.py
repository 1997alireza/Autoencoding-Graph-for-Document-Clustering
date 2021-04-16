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
from .utils import generate_data, batch_data
from .models.ae import autoencoder_with_node_features
import plotly.express as px


import os
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def run(adj, feats, validate=False, saving_directory=None):
    """

    :param adj: adjacency matrix as a 2d numpy array
    :param feats: node features as a 2d numpy array
    :param validate: if it's True, the evaluation on link prediction task is done.
    (if the evaluation is being done, the model would not be trained on all of the provided data)
    :param saving_directory: directory to save models into
    :return:
    """
    adj = (adj + 1.0) * .5  # mapping edge weights from [-1, 1] to [0, 1]

    if validate:
        print('\nPreparing test split...\n')
        # splitting 10% of data for validation
        data_number = adj.shape[0]
        val_inds = random.sample(range(0, data_number), int(data_number/10))
        val_mask = np.zeros((data_number,), dtype=bool)
        val_mask[val_inds] = True
        adj_val = adj[val_mask]
        feats_val = feats[val_mask]
        aug_adj_val = np.hstack((adj_val, feats_val))
        adj = adj[~val_mask]
        feats = feats[~val_mask]

    print('\nCompiling autoencoder model...\n')

    encoder, ae = autoencoder_with_node_features(adj.shape[1], feats.shape[1])

    print(ae.summary())

    # Specify some hyperparameters
    epochs = 60  # based on validation on 150 epochs, we use early stopping with total 60 epochs


    train_batch_size = 20

    print('\nFitting autoencoder model...\n')

    train_losses, val_losses = [], []  # in all epochs
    training_data = generate_data(adj, feats, shuffle=True)
    b_data = batch_data(training_data, train_batch_size)
    num_iters_per_train_epoch = adj.shape[0] / train_batch_size
    for e in range(epochs):
        print('\nEpoch {:d}/{:d}'.format(e + 1, epochs))
        print('Learning rate: {:6f}'.format(K.eval(ae.optimizer.lr)))
        curr_iter = 0
        train_loss = []
        for batch_adj, batch_f in b_data:
            # Each iteration/loop is a batch of train_batch_size samples
            loss = ae.train_on_batch([np.hstack((batch_adj, batch_f))], [batch_adj, batch_f])
            total_loss = loss[0]
            # when we have multiple losses, train_on_batch returns a list [total_loss, loss1, loss2, ...]
            train_loss.append(total_loss)
            curr_iter += 1
            if curr_iter >= num_iters_per_train_epoch:
                break

        train_loss = np.asarray(train_loss)
        train_loss = np.mean(train_loss, axis=0)
        print('Avg. training loss: {:s}'.format(str(train_loss)))
        train_losses.append(train_loss)

        if validate:
            val_loss = ae.test_on_batch([aug_adj_val], [adj_val, feats_val])
            total_val_loss = val_loss[0]
            val_losses.append(total_val_loss)
            print('Validation loss: {:s}'.format(str(total_val_loss)))

        elif saving_directory is not None:  # save models if we are not validating
            encoder.save(saving_directory + 'encoder.keras')
            ae.save_weights(saving_directory + 'autoencoder_weights.h5')
            # we couldn't save the autoencoder model itself because of the DenseTied layer
            print('\nTrained model is saved in epoch {}.'.format(e))

    print('Training losses during {} epochs: {}'.format(epochs, train_losses))

    fig = px.line(train_losses, labels={'index': 'epochs', 'value': 'training loss'})
    # fig.update_layout(line_color='#ff0000')
    fig['data'][0]['line']['color'] = "#ff0000"
    fig.show()

    if validate:
        print('Validation losses during {} epochs: {}'.format(epochs, val_losses))

        fig = px.line(val_losses, labels={'index': 'epochs', 'value': 'validation loss'})
        fig.show()

    print('\nAll done.')
    return encoder, ae
