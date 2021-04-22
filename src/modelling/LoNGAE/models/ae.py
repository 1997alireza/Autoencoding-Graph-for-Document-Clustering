import numpy as np
from keras.layers import Input, Dense, Dropout, Lambda, Activation
from keras.models import Model
from keras import optimizers
from keras import backend as K
from ..layers.custom import DenseTied


def mvn(tensor):
    """Per row mean-variance normalization."""
    epsilon = 1e-6
    mean = K.mean(tensor, axis=1, keepdims=True)
    std = K.std(tensor, axis=1, keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    return mvn


def autoencoder_with_node_features(adj_row_length, features_length):
    h = adj_row_length
    w = adj_row_length + features_length

    kwargs = dict(
        use_bias=True,
        kernel_initializer='glorot_normal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None,
        trainable=True,
    )

    data = Input(shape=(w,), dtype=np.float32, name='data')

    noisy_data = Dropout(rate=0.5, name='drop1')(data)

       
    ### First set of encoding transformation ###
    encoded = Dense(256, activation='relu',
            name='encoded1', **kwargs)(noisy_data)
    encoded = Lambda(mvn, name='mvn1')(encoded)

    ### Second set of encoding transformation ###
    encoded = Dense(128, activation='relu',
            name='encoded2', **kwargs)(encoded)
    encoded = Lambda(mvn, name='mvn2')(encoded)
    encoded = Dropout(rate=0.5, name='drop2')(encoded)

    # the encoder model maps an input to its encoded representation
    encoder = Model([data], encoded)
    encoded1 = encoder.get_layer('encoded1')
    encoded2 = encoder.get_layer('encoded2')

    ### First set of decoding transformation ###
    decoded = DenseTied(256, tie_to=encoded2, transpose=True,
            activation='relu', name='decoded2')(encoded)
    decoded = Lambda(mvn, name='mvn3')(decoded)

    ### Second set of decoding transformation - reconstruction ###
    decoded = DenseTied(w, tie_to=encoded1, transpose=True,
            activation='linear', name='decoded1')(decoded)

    # output related to node features
    decoded_feats = Lambda(lambda x: x[:, h:],
                        name='decoded_feats')(decoded)

    # output related to adjacency
    decoded_adj_logits = Lambda(lambda x: x[:, :h],
                                name='decoded_adj_logits')(decoded)

    decoded_adj = Activation(activation='sigmoid', name='decoded_adj')(decoded_adj_logits)


    autoencoder = Model(
            inputs=[data], outputs=[decoded_adj, decoded_feats]
    )

    # compile the autoencoder
    adam = optimizers.Adam(lr=0.01, decay=0.0)

    autoencoder.compile(
        optimizer=adam,
        loss={'decoded_adj': 'mean_squared_error',
              'decoded_feats': 'mean_squared_error'},
        loss_weights={'decoded_adj': 1.0, 'decoded_feats': 1.0}
    )

    return encoder, autoencoder
