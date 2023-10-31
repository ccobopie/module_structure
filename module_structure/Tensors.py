
import torch

class TensorOperations:

    @staticmethod
    def all_zeros(size):
        return torch.zeros(size)

    @staticmethod
    def all_ones(size):
        return torch.ones(size)

    @staticmethod
    def random_tensor(size):
        return torch.rand(size)

    @staticmethod
    def add_tensors(tensor1, tensor2):
        return torch.add(tensor1, tensor2)

    @staticmethod
    def multiply_tensors(tensor1, tensor2):
        return torch.mul(tensor1, tensor2)

    @staticmethod
    def tensor_mean(tensor):
        return torch.mean(tensor)

    @staticmethod
    def tensor_std(tensor):
        return torch.std(tensor)

    @staticmethod
    def tensor_max(tensor):
        return torch.max(tensor)

    @staticmethod
    def tensor_min(tensor):
        return torch.min(tensor)

    @staticmethod
    def tensor_exp(tensor):
        return torch.exp(tensor)

    @staticmethod
    def tensor_log(tensor):
        return torch.log(tensor)

    @staticmethod
    def tensor_sigmoid(tensor):
        return torch.sigmoid(tensor)

    @staticmethod
    def tensor_relu(tensor):
        return torch.relu(tensor)

    @staticmethod
    def tensor_transpose(tensor):
        return torch.t(tensor)

    @staticmethod
    def tensor_reshape(tensor, shape):
        return torch.reshape(tensor, shape)


#-----------------------------------------------------------------------------------
# Example usage
size = (2, 3)  # Example size for the tensors

# Creating tensors
zeros_tensor = TensorOperations.all_zeros(size)
ones_tensor = TensorOperations.all_ones(size)
random_tensor = TensorOperations.random_tensor(size)

# Performing operations
sum_tensor = TensorOperations.add_tensors(zeros_tensor, ones_tensor)
mul_tensor = TensorOperations.multiply_tensors(ones_tensor, random_tensor)
tensor_mean = TensorOperations.tensor_mean(random_tensor)
tensor_std = TensorOperations.tensor_std(random_tensor)
tensor_max = TensorOperations.tensor_max(random_tensor)
tensor_min = TensorOperations.tensor_min(random_tensor)
tensor_exp = TensorOperations.tensor_exp(random_tensor)
tensor_log = TensorOperations.tensor_log(random_tensor)
tensor_sigmoid = TensorOperations.tensor_sigmoid(random_tensor)
tensor_relu = TensorOperations.tensor_relu(random_tensor)
tensor_transpose = TensorOperations.tensor_transpose(random_tensor)
tensor_reshape = TensorOperations.tensor_reshape(random_tensor, (3, 2))

# Printing the results
print("Zeros Tensor:")
print(zeros_tensor)
print("\nOnes Tensor:")
print(ones_tensor)
print("\nRandom Tensor:")
print(random_tensor)
print("\nSum of Zeros and Ones Tensors:")
print(sum_tensor)
print("\nElement-wise Multiplication of Ones and Random Tensors:")
print(mul_tensor)
print("\nMean of Random Tensor:")
print(tensor_mean)
print("\nStandard Deviation of Random Tensor:")
print(tensor_std)
print("\nMaximum Value of Random Tensor:")
print(tensor_max)
print("\nMinimum Value of Random Tensor:")
print(tensor_min)
print("\nExponentiation of Random Tensor:")
print(tensor_exp)
print("\nLogarithm of Random Tensor:")
print(tensor_log)
print("\nSigmoid of Random Tensor:")
print(tensor_sigmoid)
print("\nReLU of Random Tensor:")
print(tensor_relu)
print("\nTranspose of Random Tensor:")
print(tensor_transpose)
print("\nReshaped Random Tensor:")
print(tensor_reshape)


# Example usage
size = (2, 3)  # Example size for the tensors

# Creating tensors
zeros_tensor = TensorOperations.all_zeros(size)
ones_tensor = TensorOperations.all_ones(size)
random_tensor = TensorOperations.random_tensor(size)

# Performing operations
sum_tensor = TensorOperations.add_tensors(zeros_tensor, ones_tensor)
mul_tensor = TensorOperations.multiply_tensors(ones_tensor, random_tensor)

# Printing the results
print("Zeros Tensor:")
print(zeros_tensor)
print("\nOnes Tensor:")
print(ones_tensor)
print("\nRandom Tensor:")
print(random_tensor)
print("\nSum of Zeros and Ones Tensors:")
print(sum_tensor)
print("\nElement-wise Multiplication of Ones and Random Tensors:")
print(mul_tensor)
