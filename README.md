# Tensor Calculator
This is a simple Python module for basic tensor operations using the PyTorch library. It provides various static methods to perform operations such as creating tensors, arithmetic operations, statistical calculations, and mathematical transformations.

# Installation
To use this module, you need to have PyTorch installed. You can install PyTorch using pip:
       *pip install torch
# Usage
python:
from tensor_calculator import TensorCalculator

# Creating Tensors
zeros_tensor = TensorCalculator.all_zeros((3, 3))
ones_tensor = TensorCalculator.all_ones((2, 2))
random_tensor = TensorCalculator.random_tensor((4, 4))

# Basic Tensor Operations
tensor_sum = TensorCalculator.add_tensors(zeros_tensor, ones_tensor)
tensor_product = TensorCalculator.multiply_tensors(zeros_tensor, ones_tensor)

# Statistical Calculations
tensor_mean = TensorCalculator.tensor_mean(random_tensor)
tensor_std = TensorCalculator.tensor_std(random_tensor)
tensor_max = TensorCalculator.tensor_max(random_tensor)
tensor_min = TensorCalculator.tensor_min(random_tensor)

# Mathematical Transformations
tensor_exp = TensorCalculator.tensor_exp(random_tensor)
tensor_log = TensorCalculator.tensor_log(random_tensor)
tensor_sigmoid = TensorCalculator.tensor_sigmoid(random_tensor)
tensor_relu = TensorCalculator.tensor_relu(random_tensor)

# Tensor Transpose and Reshape
transposed_tensor = TensorCalculator.tensor_transpose(random_tensor)
reshaped_tensor = TensorCalculator.tensor_reshape(random_tensor, (2, 8))
List of Available Methods
all_zeros(size): Creates a tensor of zeros of the specified size.
all_ones(size): Creates a tensor of ones of the specified size.
random_tensor(size): Creates a tensor of random values of the specified size.
add_tensors(tensor1, tensor2): Adds two tensors element-wise.
multiply_tensors(tensor1, tensor2): Multiplies two tensors element-wise.
tensor_mean(tensor): Computes the mean of the elements in the tensor.
tensor_std(tensor): Computes the standard deviation of the elements in the tensor.
tensor_max(tensor): Finds the maximum value in the tensor.
tensor_min(tensor): Finds the minimum value in the tensor.
tensor_exp(tensor): Computes the exponential of each element in the tensor.
tensor_log(tensor): Computes the natural logarithm of each element in the tensor.
tensor_sigmoid(tensor): Applies the sigmoid function to each element in the tensor.
tensor_relu(tensor): Applies the ReLU function to each element in the tensor.
tensor_transpose(tensor): Transposes the tensor.
tensor_reshape(tensor, shape): Reshapes the tensor to the specified shape.

Feel free to use this module to perform various tensor operations efficiently.
