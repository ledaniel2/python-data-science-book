import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense

# Load the text data
text = 'This is a sample text for creating a simple RNN using TensorFlow.'
# Preprocess the text data
chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}

# Create input and output sequences
seq_length = 10
X_data = []
y_data = []

for i in range(0, len(text) - seq_length):
    X_data.append([char_to_int[c] for c in text[i:i + seq_length]])
    y_data.append(char_to_int[text[i + seq_length]])

# Reshape the input data to (samples, time steps, features) and one-hot encode the output data
X = np.reshape(X_data, (len(X_data), seq_length, 1)) / float(len(chars))
y = tf.keras.utils.to_categorical(y_data)
# Create a simple RNN
model = tf.keras.models.Sequential()
model.add(SimpleRNN(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X, y, epochs=100, batch_size=16)
# Function to predict the next character in a sequence
def predict_next_char(input_seq):
    int_to_char = {i: c for i, c in enumerate(chars)}
    x = np.reshape([char_to_int[c] for c in input_seq], (1, len(input_seq), 1)) / float(len(chars))
    y_prob = model.predict(x)[0]
    return int_to_char[np.argmax(y_prob)]

# Test the prediction function
input_seq = 'This is a '
print(predict_next_char(input_seq))  # Output: 's'
