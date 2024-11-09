import numpy as np

# Create a simple NumPy array
array: np.ndarray = np.array([1, 2, 3, 4, 5])
array = np.random.rand(5,6,7)
# Inspect the attributes of the array
print("Array:", array)
print("Shape:", array.shape)
print("Data type:", array.dtype)
print("Size:", array.size)
print("Number of dimensions:", array.ndim)

# Access elements using indexing
print("First element:", array[0])
print("Last element:", array[-1])

# Perform some manipulations
# Reshape the array
reshaped_array = array.reshape((5, 1))
print("Reshaped array:\n", reshaped_array)

# Slice the array
sliced_array = array[1:4]
print("Sliced array:", sliced_array)

# Perform element-wise operations
squared_array = array ** 2
print("Squared array:", squared_array)