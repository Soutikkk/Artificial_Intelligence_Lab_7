import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize images to [-1, 1]
x_train = x_train.astype("float32") / 127.5 - 1
x_train = np.expand_dims(x_train, axis=-1)

# Generator
generator = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(100,)),
    layers.Dense(784, activation="tanh"),
    layers.Reshape((28, 28, 1))
])

# Discriminator
discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile discriminator
discriminator.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Connect Generator + Discriminator
discriminator.trainable = False

gan = tf.keras.Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Train GAN
for epoch in range(10):
    noise = np.random.normal(0, 1, (64, 100))

    # Generate fake images
    fake_images = generator.predict(noise, verbose=0)

    # Select real images
    real_images = x_train[np.random.randint(0, len(x_train), 64)]

    # Train discriminator
    discriminator.trainable = True
    discriminator.train_on_batch(
        real_images, np.ones((64, 1))
    )
    discriminator.train_on_batch(
        fake_images, np.zeros((64, 1))
    )

    # Train generator through GAN
    discriminator.trainable = False
    gan.train_on_batch(
        noise, np.ones((64, 1))
    )

    print(f"Epoch {epoch + 1}/10 completed")

print("GAN training completed!")
