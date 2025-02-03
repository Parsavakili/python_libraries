import numpy as np

# 1. Array Creation
print("1. Array Creation")
arr = np.array([1, 2, 3])  # Create array from list
zeros_arr = np.zeros((3, 3))  # 3x3 array of zeros
ones_arr = np.ones((2, 2))  # 2x2 array of ones
empty_arr = np.empty((2, 3))  # 2x3 uninitialized array
arange_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
eye_arr = np.eye(3)  # 3x3 identity matrix
random_arr = np.random.rand(2, 2)  # 2x2 array of random values

print("Array:", arr)
# Output: Array: [1 2 3]

print("Zeros array:\n", zeros_arr)
# Output: Zeros array:
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

print("Ones array:\n", ones_arr)
# Output: Ones array:
# [[1. 1.]
#  [1. 1.]]

print("Empty array:\n", empty_arr)
# Output: Empty array:
# [[0. 0. 0.]
#  [0. 0. 0.]] (values may vary)

print("Arange array:", arange_arr)
# Output: Arange array: [0 2 4 6 8]

print("Linspace array:", linspace_arr)
# Output: Linspace array: [0.   0.25 0.5  0.75 1.  ]

print("Identity matrix:\n", eye_arr)
# Output: Identity matrix:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

print("Random array:\n", random_arr)
# Output: Random array:
# [[0.123 0.456]
#  [0.789 0.321]] (values will vary)

print()

# 2. Array Attributes
print("2. Array Attributes")
arr_2d = np.array([[1, 2], [3, 4]])
print("Shape:", arr_2d.shape)  # (2, 2)
# Output: Shape: (2, 2)

print("Data type:", arr_2d.dtype)  # int64
# Output: Data type: int64

print("Size:", arr_2d.size)  # 4
# Output: Size: 4

print("Number of dimensions:", arr_2d.ndim)  # 2
# Output: Number of dimensions: 2

print()

# 3. Array Manipulation
print("3. Array Manipulation")
reshaped_arr = np.arange(6).reshape(2, 3)  # Reshape array
flattened_arr = arr_2d.flatten()  # Flatten array
transposed_arr = arr_2d.transpose()  # Transpose array
arr1 = np.array([1, 2])
arr2 = np.array([3, 4])
concatenated_arr = np.concatenate((arr1, arr2))  # Concatenate arrays
split_arr = np.split(concatenated_arr, 2)  # Split array
vstacked_arr = np.vstack((arr1, arr2))  # Vertical stack
hstacked_arr = np.hstack((arr1, arr2))  # Horizontal stack

print("Reshaped array:\n", reshaped_arr)
# Output: Reshaped array:
# [[0 1 2]
#  [3 4 5]]

print("Flattened array:", flattened_arr)
# Output: Flattened array: [1 2 3 4]

print("Transposed array:\n", transposed_arr)
# Output: Transposed array:
# [[1 3]
#  [2 4]]

print("Concatenated array:", concatenated_arr)
# Output: Concatenated array: [1 2 3 4]

print("Split array:", split_arr)
# Output: Split array: [array([1, 2]), array([3, 4])]

print("Vertical stack:\n", vstacked_arr)
# Output: Vertical stack:
# [[1 2]
#  [3 4]]

print("Horizontal stack:", hstacked_arr)
# Output: Horizontal stack: [1 2 3 4]

print()

# 4. Mathematical Operations
print("4. Mathematical Operations")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("Element-wise addition:", arr1 + arr2)
# Output: Element-wise addition: [5 7 9]

print("Element-wise multiplication:", arr1 * arr2)
# Output: Element-wise multiplication: [ 4 10 18]

print("Sum of array:", np.sum(arr1))
# Output: Sum of array: 6

print("Mean of array:", np.mean(arr1))
# Output: Mean of array: 2.0

print("Standard deviation:", np.std(arr1))
# Output: Standard deviation: 0.816496580927726

print("Dot product:", np.dot(arr1, arr2))
# Output: Dot product: 32

print("Sine of array:", np.sin(arr1))
# Output: Sine of array: [0.84147098 0.90929743 0.14112001]

print("Cosine of array:", np.cos(arr1))
# Output: Cosine of array: [ 0.54030231 -0.41614684 -0.9899925 ]

print()

# 5. Indexing and Slicing
print("5. Indexing and Slicing")
arr = np.array([1, 2, 3, 4])
print("Indexing:", arr[0])  # 1
# Output: Indexing: 1

print("Slicing:", arr[1:3])  # [2, 3]
# Output: Slicing: [2 3]

print("Boolean indexing:", arr[arr > 2])  # [3, 4]
# Output: Boolean indexing: [3 4]

print()

# 6. Linear Algebra
print("6. Linear Algebra")
mat = np.array([[1, 2], [3, 4]])
inv_mat = np.linalg.inv(mat)  # Inverse of matrix
det = np.linalg.det(mat)  # Determinant of matrix
eigenvalues, eigenvectors = np.linalg.eig(mat)  # Eigenvalues and eigenvectors

print("Matrix:\n", mat)
# Output: Matrix:
# [[1 2]
#  [3 4]]

print("Inverse of matrix:\n", inv_mat)
# Output: Inverse of matrix:
# [[-2.   1. ]
#  [ 1.5 -0.5]]

print("Determinant of matrix:", det)
# Output: Determinant of matrix: -2.0000000000000004

print("Eigenvalues:", eigenvalues)
# Output: Eigenvalues: [-0.37228132  5.37228132]

print("Eigenvectors:\n", eigenvectors)
# Output: Eigenvectors:
# [[-0.82456484 -0.41597356]
#  [ 0.56576746 -0.90937671]]

print()

# 7. File I/O
print("7. File I/O")
np.save('array.npy', arr)  # Save array to binary file
loaded_arr = np.load('array.npy')  # Load array from binary file
np.savetxt('array.txt', arr)  # Save array to text file
loaded_txt_arr = np.loadtxt('array.txt')  # Load array from text file

print("Loaded array from .npy file:", loaded_arr)
# Output: Loaded array from .npy file: [1 2 3 4]

print("Loaded array from .txt file:", loaded_txt_arr)
# Output: Loaded array from .txt file: [1. 2. 3. 4.]

print()

# 8. Miscellaneous Functions
print("8. Miscellaneous Functions")
arr = np.array([1, 2, 3, 4])
indices = np.where(arr > 2)  # Indices where condition is true
unique_elements = np.unique([1, 2, 2, 3])  # Unique elements
sorted_arr = np.sort([3, 1, 2])  # Sorted array

print("Indices where arr > 2:", indices)
# Output: Indices where arr > 2: (array([2, 3]),)

print("Unique elements:", unique_elements)
# Output: Unique elements: [1 2 3]

print("Sorted array:", sorted_arr)
# Output: Sorted array: [1 2 3]