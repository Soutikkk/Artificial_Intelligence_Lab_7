import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Training data: sequences of numbers
X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
])

# Output: next number in the sequence
y = np.array([4, 5, 6, 7])

# RNN expects data in the form:
# (samples, time steps, features)
X = X.reshape((4, 3, 1))

# Create the RNN model
model = Sequential([
    SimpleRNN(10, activation='relu', input_shape=(3, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Test the RNN
test = np.array([5, 6, 7]).reshape((1, 3, 1))

prediction = model.predict(test, verbose=0)

print("Input sequence:", [5, 6, 7])
print("Predicted next number:", prediction[0][0])
