import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model


class ClusteringAutoEncoderModel:

    def __init__(self, data_size):
        input_data = Input(shape=(data_size,), dtype=np.float32, name='data')

        noisy_data = Dropout(rate=0.3, name='drop')(input_data)

        encoded = Dense(1024, activation='relu', name='encoded1',)(noisy_data)
        encoded = Dense(32, activation='relu', name='encoded2')(encoded)

        # the encoder model maps an input to its encoded representation
        encoder = Model(input_data, encoded)

        decoded = Dense(256, activation='relu', name='decoded2')(encoded)
        decoded = Dense(data_size, activation='linear', name='decoded1')(decoded)

        autoencoder = Model(input_data, decoded)

        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        self.encoder = encoder
        self.autoencoder = autoencoder

    def train(self, data):
        return self.autoencoder.fit(data, data,
                                    epochs=50,
                                    batch_size=256,
                                    shuffle=True,
                                    verbose=2)

