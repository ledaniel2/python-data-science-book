import numpy as np

# Creating NumPy arrays from lists
list_array = np.array([1, 2, 3, 4, 5])
print("Array from a list:", list_array)

# Creating NumPy arrays from tuples
tuple_array = np.array((6, 7, 8, 9, 10))
print("Array from a tuple:", tuple_array)

# Creating NumPy arrays using built-in functions
zeros_array = np.zeros(5)
print("Array of zeros:", zeros_array)

ones_array = np.ones(5)
print("Array of ones:", ones_array)

range_array = np.arange(1, 11, 2)
print("Array with a range of values:", range_array)
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Element-wise addition
C = A + B
print("Element-wise addition:", C)

# Element-wise subtraction
D = A - B
print("Element-wise subtraction:", D)

# Element-wise multiplication
E = A * B
print("Element-wise multiplication:", E)

# Element-wise division
F = A / B
print("Element-wise division:", F)

# Matrix multiplication
G = np.dot(A, B)
print("Matrix multiplication:", G)
# Create a 1D array
array_1d = np.array([1, 2, 3, 4, 5])

# Indexing a 1D array
print("First element:", array_1d[0])
print("Last element:", array_1d[-1])

# Slicing a 1D array
print("First three elements:", array_1d[:3])
print("Last two elements:", array_1d[-2:])

# Create a 2D array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexing a 2D array
print("Element at position (1,2):", array_2d[1, 2])

# Slicing a 2D array
print("First two rows:\n", array_2d[:2, :])
print("Last column:\n", array_2d[:, -1])
# Create a 1D array
array_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Reshape the 1D array into a 3x3 2D array
array_2d = array_1d.reshape(3, 3)
print("Reshaped 2D array:\n", array_2d)

# Flatten the 2D array back into a 1D array
array_flattened = array_2d.flatten()
print("Flattened 1D array:", array_flattened)

# Concatenate two arrays
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = np.concatenate((A, B))
print("Concatenated array:", C)

# Stack arrays vertically
D = np.vstack((A, B))
print("Vertical stack:\n", D)

# Stack arrays horizontally
E = np.hstack((A, B))
print("Horizontal stack:", E)
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([1, 2, 3])

# Add B to each row of A
C = A + B
print("Broadcasted addition:\n", C)

# Multiply each row of A by B
D = A * B
print("Broadcasted multiplication:\n", D)
