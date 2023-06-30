import tensorflow as tf
from tensorflow.keras import layers
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Sequential model
model = tf.keras.Sequential()

# Add an input layer with 8 neurons and a ReLU activation function
model.add(layers.Dense(8, activation='relu', input_shape=(10,)))

# Add a hidden layer with 16 neurons and a ReLU activation function
model.add(layers.Dense(16, activation='relu'))

# Add an output layer with 1 neuron and a sigmoid activation function
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
