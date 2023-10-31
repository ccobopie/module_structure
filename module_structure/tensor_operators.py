
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
