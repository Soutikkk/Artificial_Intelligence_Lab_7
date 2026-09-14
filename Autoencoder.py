import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Create sample data
X = np.random.rand(1000, 10)

# Input layer
input_layer = Input(shape=(10,))

# Encoder
encoded = Dense(5, activation='relu')(input_layer)

# Decoder
decoded = Dense(10, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile
autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X, X, epochs=10, batch_size=32, verbose=1)

print("Autoencoder training completed!")
