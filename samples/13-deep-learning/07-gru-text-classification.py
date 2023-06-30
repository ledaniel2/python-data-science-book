import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Set the maximum number of words to consider in the vocabulary
max_words = 10000

# Load the Reuters dataset
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words)

# Set the maximum sequence length
max_sequence_length = 100

# Pad sequences to the same length
X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

# Convert the labels to categorical format
num_classes = max(y_train) + 1
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Create a Sequential model
model = tf.keras.Sequential()

# Add an Embedding layer with an input dimension of 10000 (vocabulary size), an output dimension of 32, and an input length of 100 (sequence length)
model.add(layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_sequence_length))

# Add a GRU layer with 128 units
model.add(layers.GRU(units=128))

# Add a fully connected layer with 64 neurons and a ReLU activation function
model.add(layers.Dense(units=64, activation='relu'))

# Add an output layer with 2 neurons (one for each class) and a softmax activation function
model.add(layers.Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
